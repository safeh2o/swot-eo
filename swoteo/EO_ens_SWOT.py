import base64
import datetime
import io
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import fsolve, minimize
from scipy.stats import ks_2samp, levene
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from xlrd.xldate import xldate_as_datetime
from yattag import Doc

np.random.seed(0)
from . import EO_functions

plt.rcParams["figure.autolayout"] = True

# np.random.seed(19231520)

class EO_Ensemble:
    def __init__(self, inputtime, savepath, inputpath, scenario):
        """Methods:
        'Average': Take the straight average for each prediction
        'Linear Weighted Average': Take the average of each ensemble prediction based on linear calibration data error
        'Quadratic Weighted Error': Take average of each ensemble prediction based on quadratic calibration data error
        'Best Member': take the best member"""
        scenario_map = {
            "optimumDecay": "Optimum Decay",
            "minDecay": "Minimum Decay",
            "maxDecay": "Maximum Decay",
        }
        self.scenario = scenario_map[scenario]
        self.ensemble_size = 100
        self.inputpath = inputpath
        self.savepath = savepath

        self.version = "1.8.1"

        self.inputtime = int(inputtime)

        self.decay_equations = ["First Order", "Power Decay", "Parallel First Order"]

        self.f_dict = {
            "Power Decay": EO_functions.power_law_predict,
            "First Order": EO_functions.first_order_predict,
            "Parallel First Order": EO_functions.parallel_first_order_predict,
        }

        self.convergence_checker = {
            "Power Decay": True,
            "First Order": True,
            "Parallel First Order": True,
        }

        self.guess_dict = {
            "Power Decay": [0.6, 2.0],
            "First Order": [0.6],
            "Parallel First Order": [1, 0.6, 0.6],
        }
        self.bounds_dict = {
            "Power Decay": [[0, None], [0, 2]],
            "First Order": [[0, None]],
            "Parallel First Order": [[0, 1], [0, None], [0, None]],
        }

        self.var_dict = {"Power Decay": 2, "First Order": 1, "Parallel First Order": 3}
        self.labels_dict = {
            "Power Decay": ["Decay rate (k) (h^-1)", "Decay order (n) dimensionless"],
            "First Order": ["Decay rate (k) (h^-1)"],
            "Parallel First Order": [
                "Ratio of slow to fast decay (w)",
                "Decay rate 1 (k1) (h^-1)",
                "Decay rate 2 (k2) (h^-1)",
            ],
        }
        self.solver_dict = {
            "Power Decay": "Unbound",
            "First Order": "Unbound",
            "Parallel First Order": "Bound",
        }
        self.SSE_ensemble_params = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }
        self.train_MSE = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }
        self.test_MSE = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }
        self.train_r2 = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }
        self.test_r2 = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }

        self.targets = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }

        self.SSE_test_preds = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }

        self.model = "Power Decay"

        self.confidence = {
            "Observations": None,
            "Stability": None,
            "Histogram": None,
            "Uniformity": None,
            "Decay Rate": None,
            "Model Fit": None,
        }

        self.confidence_reason = {
            "Observations": None,
            "Stability": None,
            "Histogram": None,
            "Uniformity": None,
            "Decay Rate": None,
            "Model Fit": None,
        }

        self.opt_params = {
            "Power Decay": [np.nan, np.nan],
            "First Order": [np.nan],
            "Parallel First Order": [np.nan, np.nan, np.nan],
        }

        self.CI_params = {
            "Power Decay": None,
            "First Order": None,
            "Parallel First Order": None,
        }

        self.min_params = {
            "Power Decay": [np.nan, np.nan],
            "First Order": [np.nan],
            "Parallel First Order": [np.nan, np.nan, np.nan],
        }

        self.max_params = {
            "Power Decay": [np.nan, np.nan],
            "First Order": [np.nan],
            "Parallel First Order": [np.nan, np.nan, np.nan],
        }

    def import_data(self):
        if not os.path.exists(self.savepath):
            os.makedirs(self.savepath)
        df = pd.read_csv(self.inputpath)

        df.dropna(
            inplace=True, subset=["ts_datetime", "ts_frc", "hh_datetime", "hh_frc"]
        )
        df.reset_index(drop=True, inplace=True)
        start_date = df["ts_datetime"]
        end_date = df["hh_datetime"]

        durations = []

        for i in range(0, len(start_date)):
            try:
                # excel type
                start = float(start_date[i])
                end = float(end_date[i])
                start = xldate_as_datetime(start, datemode=0)

                end = xldate_as_datetime(end, datemode=0)
                durations.append((end - start).total_seconds())

            except ValueError:
                start = start_date[i][:16].replace("/", "-")
                end = end_date[i][:16].replace("/", "-")
                start = datetime.datetime.strptime(start, "%Y-%m-%dT%H:%M")
                try:
                    end = datetime.datetime.strptime(end, "%Y-%m-%dT%H:%M")
                    durations.append((end - start).total_seconds())
                except ValueError:
                    end = np.nan
                    start = np.nan
                    durations.append(np.nan)

        df["se4_lag"] = np.array(durations) / 3600
        df = self.data_clean(df)
        X = df.drop(["hh_frc"], axis=1)
        Y = pd.Series(df["hh_frc"].values)

        self.X_cal, X_test, self.Y_cal, Y_test = train_test_split(
            X, Y, test_size=0.1, shuffle=True
        )

        self.f0_test = pd.Series(X_test["ts_frc"].values).to_numpy()
        self.t_test = pd.Series(X_test["se4_lag"].values).to_numpy()
        self.f_test = Y_test.to_numpy()
        return

    def data_clean(self, dataset):
        dataset["delta_frc"] = dataset["ts_frc"] - dataset["hh_frc"]
        dataset = dataset[dataset.delta_frc >= 0.06]
        dataset = dataset[dataset.se4_lag >= 0]
        dataset = dataset[dataset.se4_lag <= 48]
        dataset = dataset[dataset.ts_frc >= 0]
        dataset = dataset[dataset.hh_frc >= 0]
        return dataset

    def cost_check_SSE(self, x, C0, t, y_true):
        y_pred = self.predictor(x, C0, t)
        cost_fun = EO_functions.SSE(y_true, y_pred)
        return cost_fun

    def power_params_drop(self):
        delete_indices = [
            np.argwhere(self.SSE_ensemble_params["Power Decay"][:, 1] <= 0.0000001)
        ]
        self.SSE_ensemble_params["Power Decay"] = np.delete(
            self.SSE_ensemble_params["Power Decay"], delete_indices, axis=0
        )
        self.test_MSE["Power Decay"] = np.delete(
            self.test_MSE["Power Decay"], delete_indices, axis=0
        )
        self.test_r2["Power Decay"] = np.delete(
            self.test_r2["Power Decay"], delete_indices, axis=0
        )
        self.train_MSE["Power Decay"] = np.delete(
            self.train_MSE["Power Decay"], delete_indices, axis=0
        )
        self.train_r2["Power Decay"] = np.delete(
            self.train_r2["Power Decay"], delete_indices, axis=0
        )
        return

    def pfo_params_drop(self):
        delete_indices = [
            np.argwhere(
                self.SSE_ensemble_params["Parallel First Order"][:, 1] <= 0.0000001
            )
        ]
        self.SSE_ensemble_params["Parallel First Order"] = np.delete(
            self.SSE_ensemble_params["Parallel First Order"], delete_indices, axis=0
        )
        self.test_MSE["Parallel First Order"] = np.delete(
            self.test_MSE["Parallel First Order"], delete_indices, axis=0
        )
        self.test_r2["Parallel First Order"] = np.delete(
            self.test_r2["Parallel First Order"], delete_indices, axis=0
        )
        self.train_MSE["Parallel First Order"] = np.delete(
            self.train_MSE["Parallel First Order"], delete_indices, axis=0
        )
        self.train_r2["Parallel First Order"] = np.delete(
            self.train_r2["Parallel First Order"], delete_indices, axis=0
        )
        return

    def params_check(self, model):
        var = []
        for i in range(0, len(self.labels_dict[model])):
            var.append(np.var(self.SSE_ensemble_params[model][:, i]))
        if np.max(np.array(var)) > 0.1:
            self.convergence_checker[model] = False
        return [var]

    def overfitting_check(self, model):
        """
        % increase MSE training to testing
        % decrease r2 training to testing

        Look at best and 95th percentile since this is the optimum model
        """

        # check 1 r2:
        check_r2_best = (np.max(self.test_r2[model]) - np.max(self.train_r2[model])) / (
                1 - np.max(self.train_r2[model])
        )
        check_r2_worst = (
                                 np.percentile(self.test_r2[model], 75)
                                 - np.percentile(self.train_r2[model], 75)
                         ) / (1 - np.percentile(self.train_r2[model], 75))

        # check 2:
        check_mse_best = (
                                 np.min(self.test_MSE[model]) - np.min(self.train_MSE[model])
                         ) / np.min(self.train_MSE[model])
        check_mse_worst = (
                                  np.percentile(self.test_MSE[model], 25)
                                  - np.percentile(self.train_MSE[model], 25)
                          ) / np.percentile(self.train_MSE[model], 25)

        if (
                check_mse_best > 0.5
                or check_mse_worst > 0.5
                or check_r2_worst < -0.5
                or check_r2_best < -0.5
        ):
            self.convergence_checker[model] = False

        return [check_r2_best, check_r2_worst, check_mse_best, check_mse_worst]

    def fo_targets(self):
        times = np.arange(3, 25, 3)
        if self.inputtime not in times:
            times=np.append(times, self.inputtime)
        CI_idx = np.argwhere(
            self.test_MSE["First Order"]
            <= np.percentile(self.test_MSE["First Order"], 25)
        ).flatten()
        opt_idx = np.argmin(self.test_MSE["First Order"])
        params = self.SSE_ensemble_params["First Order"][CI_idx]
        opt_params = self.SSE_ensemble_params["First Order"][opt_idx]
        max_params = np.max(params)
        min_params = np.min(params)
        opt_targets = []
        max_decay_targets = []
        min_decay_targets = []
        for t in times:
            opt_targets.append(0.3 / (np.exp(-1 * opt_params * t)))
            max_decay_targets.append(0.3 / (np.exp(-1 * max_params * t)))
            min_decay_targets.append(0.3 / (np.exp(-1 * min_params * t)))

        self.targets["First Order"] = pd.DataFrame(
            {
                "Optimum Decay": np.array(opt_targets).flatten(),
                "Maximum Decay": np.array(max_decay_targets).flatten(),
                "Minimum Decay": np.array(min_decay_targets).flatten(),
            },
            index=times,
        )
        self.opt_params["First Order"] = opt_params
        self.CI_params["First Order"] = params
        self.min_params["First Order"] = [
            np.min(params),
            self.test_MSE["First Order"][
                np.argwhere(self.SSE_ensemble_params["First Order"] == np.min(params))
            ],
        ]
        self.max_params["First Order"] = [
            np.max(params),
            self.test_MSE["First Order"][
                np.argwhere(self.SSE_ensemble_params["First Order"] == np.max(params))
            ],
        ]
        return

    def power_targets(self):
        times = np.arange(3, 25, 3)
        if self.inputtime not in times:
            times=np.append(times, self.inputtime)
        CI_idx = np.argwhere(
            self.test_MSE["Power Decay"]
            <= np.percentile(self.test_MSE["Power Decay"], 25)
        ).flatten()
        opt_idx = np.argmin(self.test_MSE["Power Decay"])
        params = self.SSE_ensemble_params["Power Decay"][CI_idx]
        opt_params = self.SSE_ensemble_params["Power Decay"][opt_idx]
        n = params[:, 1]
        k = params[:, 0]
        n_opt = opt_params[1]
        k_opt = opt_params[0]
        opt_targets = []
        max_decay_targets = []
        min_decay_targets = []
        for t in times:
            opt_targets.append(
                (0.3 ** (1 - n_opt) - (n_opt - 1) * k_opt * t) ** (1 / (1 - n_opt))
            )
            targets = (0.3 ** (1 - n) - (n - 1) * k * t) ** (1 / (1 - n))
            max_decay_targets.append(np.max(targets))
            min_decay_targets.append(np.min(targets))

        self.targets["Power Decay"] = pd.DataFrame(
            {
                "Optimum Decay": np.array(opt_targets),
                "Maximum Decay": np.array(max_decay_targets),
                "Minimum Decay": np.array(min_decay_targets),
            },
            index=times,
        )
        if np.max(np.array(max_decay_targets) - np.array(min_decay_targets)) > 1:
            self.convergence_checker["Power Decay"] = False
            return
        if self.targets["Power Decay"][self.scenario].loc[self.inputtime] < 0.3:
            self.convergence_checker["Power Decay"] = False
            return

        if self.targets["Power Decay"][self.scenario].loc[self.inputtime] > 100:
            self.convergence_checker["Power Decay"] = False
            return

        if (
                np.isnan(self.targets["Power Decay"][self.scenario].loc[self.inputtime])
                == True
        ):
            self.convergence_checker["Power Decay"] = False
            return

        inp_target = (0.3 ** (1 - n) - (n - 1) * k * self.inputtime) ** (1 / (1 - n))
        n_min, k_min, n_max, k_max = (
            n[np.argmin(inp_target)],
            k[np.argmin(inp_target)],
            n[np.argmax(inp_target)],
            k[np.argmax(inp_target)],
        )

        self.opt_params["Power Decay"] = opt_params
        self.CI_params["Power Decay"] = params
        self.min_params["Power Decay"] = [k_min,n_min]
        self.max_params["Power Decay"] = [k_max,n_max ]
        return

    def pfo_targets(self):
        times = np.arange(3, 25, 3)
        if self.inputtime not in times:
            times=np.append(times, self.inputtime)
        CI_idx = np.argwhere(
            self.test_MSE["Parallel First Order"]
            <= np.percentile(self.test_MSE["Parallel First Order"], 25)
        ).flatten()
        opt_idx = np.argmin(self.test_MSE["Parallel First Order"])
        params = self.SSE_ensemble_params["Parallel First Order"][CI_idx]
        opt_params = self.SSE_ensemble_params["Parallel First Order"][opt_idx]
        w = params[:, 0]
        k1 = params[:, 1]
        k2 = params[:, 2]
        w_opt = opt_params[0]
        k1_opt = opt_params[1]
        k2_opt = opt_params[2]
        opt_targets = []
        max_decay_targets = []
        min_decay_targets = []
        for t in times:
            opt_targets.append(
                fsolve(
                    lambda C_0: w_opt * C_0 * np.exp(-1 * k1_opt * t)
                                + (1 - w_opt) * C_0 * np.exp(-1 * k2_opt * t)
                                - 0.3,
                    0.3,
                )
            )

            targets = fsolve(
                lambda C_0: w * C_0 * np.exp(-1 * k1 * t)
                            + (1 - w) * C_0 * np.exp(-1 * k2 * t)
                            - 0.3,
                np.array([0.3 for i in range(len(w))]),
            )
            max_decay_targets.append(np.max(targets))
            min_decay_targets.append(np.min(targets))

        self.targets["Parallel First Order"] = pd.DataFrame(
            {
                "Optimum Decay": np.array(opt_targets).flatten(),
                "Maximum Decay": np.array(max_decay_targets),
                "Minimum Decay": np.array(min_decay_targets),
            },
            index=times,
        )

        if np.max(np.array(max_decay_targets) - np.array(min_decay_targets)) > 1:
            self.convergence_checker["Parallel First Order"] = False
            return

        if (
                self.targets["Parallel First Order"][self.scenario].loc[self.inputtime]
                < 0.3
        ):
            self.convergence_checker["Parallel First Order"] = False
            return

        if (
                self.targets["Parallel First Order"][self.scenario].loc[self.inputtime]
                > 100
        ):
            self.convergence_checker["Parallel First Order"] = False
            return

        if (
                np.isnan(
                    self.targets["Parallel First Order"][self.scenario].loc[self.inputtime]
                )
                == True
        ):
            self.convergence_checker["Parallel First Order"] = False
            return

        inp_target = fsolve(
            lambda C_0: w * C_0 * np.exp(-1 * k1 * self.inputtime)
                        + (1 - w) * C_0 * np.exp(-1 * k2 * self.inputtime)
                        - 0.3,
            np.array([0.3 for i in range(len(w))]),
        )
        w_min, k1_min, k2_min, w_max, k1_max, k2_max = (
            w[np.argmin(inp_target)],
            k1[np.argmin(inp_target)],
            k2[np.argmin(inp_target)],
            w[np.argmax(inp_target)],
            k1[np.argmax(inp_target)],
            k2[np.argmax(inp_target)],
        )

        self.opt_params["Parallel First Order"] = opt_params
        self.CI_params["Parallel First Order"] = params
        self.min_params["Parallel First Order"] = [w_min, k1_min, k2_min]
        self.max_params["Parallel First Order"] = [w_max, k1_max, k2_max]
        return

    def First_Order_Model(self):
        eqn = "First Order"
        self.predictor = self.f_dict[eqn]
        self.nvar = self.var_dict[eqn]
        self.bounds = self.bounds_dict[eqn]
        self.guess = self.guess_dict[eqn]
        (
            params,
            train_preds,
            train_mse,
            train_r2,
            test_preds,
            test_mse,
            test_r2,
        ) = self.EO_Ensemble_SSE_unbound(
            self.X_cal, self.Y_cal, self.f0_test, self.f_test, self.t_test
        )

        self.SSE_ensemble_params[eqn] = params
        self.test_MSE[eqn] = test_mse
        self.train_MSE[eqn] = train_mse
        self.train_r2[eqn] = train_r2
        self.test_r2[eqn] = test_r2

        targets_check = self.fo_targets()
        params_check = self.params_check(eqn)
        overfit_check = self.overfitting_check(eqn)
        return

    def Power_Decay_Model(self):
        eqn = "Power Decay"
        self.predictor = self.f_dict[eqn]
        self.nvar = self.var_dict[eqn]
        self.bounds = self.bounds_dict[eqn]
        self.guess = self.guess_dict[eqn]
        (
            params,
            train_preds,
            train_mse,
            train_r2,
            test_preds,
            test_mse,
            test_r2,
        ) = self.EO_Ensemble_SSE_bound(
            self.X_cal, self.Y_cal, self.f0_test, self.f_test, self.t_test
        )
        self.SSE_ensemble_params[eqn] = params
        self.test_MSE[eqn] = test_mse
        self.train_MSE[eqn] = train_mse
        self.train_r2[eqn] = train_r2
        self.test_r2[eqn] = test_r2

        self.power_params_drop()
        if len(self.SSE_ensemble_params[eqn]) == 0:
            self.convergence_checker[eqn] = False
            return
        var = self.params_check(eqn)
        if self.convergence_checker[eqn] == False:
            return
        overfit_checks = self.overfitting_check(eqn)
        if self.convergence_checker[eqn] == False:
            return
        targets_check = self.power_targets()

        if self.convergence_checker[eqn] == False:
            return
        return

    def Parallel_First_Order_Model(self):
        eqn = "Parallel First Order"
        self.predictor = self.f_dict[eqn]
        self.nvar = self.var_dict[eqn]
        self.bounds = self.bounds_dict[eqn]
        self.guess = self.guess_dict[eqn]
        (
            params,
            train_preds,
            train_mse,
            train_r2,
            test_preds,
            test_mse,
            test_r2,
        ) = self.EO_Ensemble_SSE_bound(
            self.X_cal, self.Y_cal, self.f0_test, self.f_test, self.t_test
        )

        for i in range(len(params[:, 0])):
            if params[i][1] >= params[i][2]:
                params[i][0] = 1 - params[i][0]
                k1 = params[i][2]
                k2 = params[i][1]
                params[i][1] = k1
                params[i][2] = k2

        self.SSE_ensemble_params[eqn] = params
        self.test_MSE[eqn] = test_mse
        self.train_MSE[eqn] = train_mse
        self.train_r2[eqn] = train_r2
        self.test_r2[eqn] = test_r2

        self.pfo_params_drop()

        if len(self.SSE_ensemble_params[eqn]) == 0:
            self.convergence_checker[eqn] = False
            return
        var = self.params_check(eqn)
        if self.convergence_checker[eqn] == False:
            return
        overfit_checks = self.overfitting_check(eqn)
        if self.convergence_checker[eqn] == False:
            return
        targets_check = self.pfo_targets()
        if self.convergence_checker[eqn] == False:
            return
        return

    def EO_Ensemble_SSE_unbound(self, X_cal, Y_cal, f0_test, f_test, t_test):
        ens_params = np.empty((0, self.nvar), float)
        ens_MSE_train = []
        ens_MSE_test = []
        ens_r2_train = []
        ens_r2_test = []
        f_pred = []
        f_pred_test = []
        for i in range(0, self.ensemble_size):
            guess = np.random.sample(self.nvar) * self.guess
            X_train, X_val, Y_train, Y_val = train_test_split(
                self.X_cal, Y_cal, test_size=0.25, shuffle=True,
                random_state=i ** 2
            )
            f0 = pd.Series(X_train["ts_frc"].values).to_numpy()
            t = pd.Series(X_train["se4_lag"].values).to_numpy()
            f = Y_train.to_numpy()

            sol = minimize(
                self.cost_check_SSE,
                guess,
                method="Powell",
                args=(f0, t, f),
                options={"maxiter": 600},
            )
            xopt = sol.x

            f0 = pd.Series(X_cal["ts_frc"].values).to_numpy()
            t = pd.Series(X_cal["se4_lag"].values).to_numpy()
            f = Y_cal.to_numpy()

            ens_params = np.vstack((ens_params, xopt))
            f_pred.append(self.predictor(xopt, f0, t))
            f_pred_test.append(self.predictor(xopt, f0_test, t_test))

            ens_MSE_train.append(EO_functions.SSE(f, f_pred[i]) / len(f))
            ens_MSE_test.append(EO_functions.SSE(f_test, f_pred_test[i]) / len(f_test))
            ens_r2_train.append(r2_score(f, f_pred[i]))
            ens_r2_test.append(r2_score(f_test, f_pred_test[i]))

        return (
            ens_params,
            np.array(f_pred),
            np.array(ens_MSE_train),
            np.array(ens_r2_train),
            np.array(f_pred_test),
            np.array(ens_MSE_test),
            np.array(ens_r2_test),
        )

    def EO_Ensemble_SSE_bound(self, X_cal, Y_cal, f0_test, f_test, t_test):
        ens_params = np.empty((0, self.nvar), float)
        ens_MSE_train = []
        ens_MSE_test = []
        ens_r2_train = []
        ens_r2_test = []
        f_pred_test = []
        f_pred = []
        for i in range(0, self.ensemble_size):
            guess = np.random.sample(self.nvar) * self.guess
            X_train, X_val, Y_train, Y_val = train_test_split(
                X_cal, Y_cal, test_size=0.25, shuffle=True,
                random_state=i ** 2
            )
            f0 = pd.Series(X_train["ts_frc"].values).to_numpy()
            t = pd.Series(X_train["se4_lag"].values).to_numpy()
            f = Y_train.to_numpy()

            sol = minimize(
                self.cost_check_SSE,
                guess,
                method="TNC",
                args=(f0, t, f),
                bounds=self.bounds,
            )
            xopt = sol.x

            ens_params = np.vstack((ens_params, xopt))

            f_pred.append(self.predictor(xopt, f0, t))
            f_pred_test.append(self.predictor(xopt, f0_test, t_test))

            ens_MSE_train.append(EO_functions.SSE(f, f_pred[i]) / len(f))
            ens_MSE_test.append(EO_functions.SSE(f_test, f_pred_test[i]) / len(f_test))
            ens_r2_train.append(r2_score(f, f_pred[i]))
            ens_r2_test.append(r2_score(f_test, f_pred_test[i]))

        return (
            ens_params,
            np.array(f_pred),
            np.array(ens_MSE_train),
            np.array(ens_r2_train),
            np.array(f_pred_test),
            np.array(ens_MSE_test),
            np.array(ens_r2_test),
        )

    def k_n_fig(self, solver):
        if len(self.labels_dict[solver]) == 1:
            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_series1_scatter.csv"),
                       np.transpose([self.SSE_ensemble_params[solver].flatten(),self.test_MSE[solver]]),
                       delimiter=',',header="Decay Rate, MSE",comments='')
            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_series2_scatter.csv"),
                       np.transpose([np.array(self.CI_params[solver]).flatten(),
                        np.array(self.test_MSE[solver][
                            np.argwhere(self.test_MSE[solver] <= np.percentile(self.test_MSE[solver], 25))]).flatten()]),
                       delimiter=',',header="Upper 25 Decay Rate, MSE",comments='')
            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_optseries_scatter.csv"),
                       np.append(self.opt_params[solver], np.min(self.test_MSE[solver])).reshape(1,2),
                       delimiter=',',header="Optimum Solution Decay Rate, MSE",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_minseries_scatter.csv"),
                       np.append(self.min_params[solver][0],np.min(self.min_params[solver][1])).reshape(1,2),
                       delimiter=',',header="Min Decay Solution Decay Rate, MSE",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_maxseries_scatter.csv"),
                       np.append(self.max_params[solver][0],np.min(self.max_params[solver][1])).reshape(1,2),
                       delimiter=',',header="Max Decay Solution Decay Rate, MSE",comments='')

            param_df=pd.DataFrame({"Decay Rate (k)":[self.opt_params[solver][0],self.max_params[solver][0],self.min_params[solver][0]]},index=['Optimum Decay','Maximum Decay','Minimum Decay'])
            param_df.index.name='Scenario'
            str_io=io.StringIO()
            param_df.to_html(buf=str_io, table_id="paramTable")
            self.params_table_html=str_io.getvalue()

            '''kn_fig, ax = plt.subplots(1, 1)
            ax.scatter(
                self.SSE_ensemble_params[solver],
                self.test_MSE[solver],
                marker="o",
                edgecolors="k",
                facecolors="none",
                linewidths=0.5,
            )
            ax.scatter(
                self.opt_params[solver],
                np.min(self.test_MSE[solver]),
                marker="*",
                s=100,
                c="#0000ff",
                label="Optimum Decay Parameters",
            )
            ax.scatter(
                self.CI_params[solver],
                self.test_MSE[solver][
                    np.argwhere(
                        self.test_MSE[solver]
                        <= np.percentile(self.test_MSE[solver], 25)
                    )
                ],
                edgecolors="#0000ff",
                facecolors="none",
                linewidths=0.5,
                label="75% Confidence Region",
            )
            ax.scatter(
                self.min_params[solver][0],
                np.min(self.min_params[solver][1]),
                marker="*",
                s=100,
                c="#00ff00",
                label="Minimum Decay Parameters",
            )
            ax.scatter(
                self.max_params[solver][0],
                np.min(self.max_params[solver][1]),
                marker="*",
                s=100,
                c="#ff0000",
                label="Maximum Decay Parameters",
            )
            ax.set_title(solver)
            ax.set_xlabel(self.labels_dict[solver][0])
            ax.set_ylabel("Test MSE")
            ax.legend()
            kn_fig.savefig(os.path.join(self.savepath, "params.png"))

            StringIOBytes_kn = io.BytesIO()
            kn_fig.savefig(StringIOBytes_kn, format="png", bbox_inches="tight")
            StringIOBytes_kn.seek(0)
            self.kn_base_64_pngData = base64.b64encode(StringIOBytes_kn.read())
            plt.close(kn_fig)'''
            return
        elif len(self.labels_dict[solver]) == 2:
            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_series1_scatter.csv"),
                       np.transpose([self.SSE_ensemble_params[solver][:, 0],
                self.SSE_ensemble_params[solver][:, 1]]),
                       delimiter=',',header="Decay Rate,Decay Order",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_2_plot_series2_scatter.csv"),
                       np.transpose([self.CI_params[solver][0],
                    self.CI_params[solver][1]]),
                       delimiter=',',header="Upper 25 Decay Rate,Upper 25 Decay Order",comments='')
            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_optseries_scatter.csv"),
                       np.append(self.opt_params[solver][0], self.opt_params[solver][1]).reshape(1,2),
                       delimiter=',',header="Optimum Decay Rate,Optimum Decay Order",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_minseries_scatter.csv"),
                       np.append(self.min_params[solver][0], self.min_params[solver][1]).reshape(1,2),
                       delimiter=',',header="Minimum Decay Rate,Minimum Decay Order",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_maxseries_scatter.csv"),
                       np.append(self.max_params[solver][0], self.max_params[solver][1]).reshape(1,2),
                       delimiter=',',header="Maximum Decay rate,Maximum Decay Order",comments='')

            param_df = pd.DataFrame({"Decay Rate (k)": [self.opt_params[solver][0], self.max_params[solver][0],
                                                        self.min_params[solver][0]],"Decay Order (n)":[self.opt_params[solver][1], self.max_params[solver][1],
                                                        self.min_params[solver][1]]},
                                    index=['Optimum Decay', 'Maximum Decay', 'Minimum Decay'])
            param_df.index.name = 'Scenario'
            str_io = io.StringIO()
            param_df.to_html(buf=str_io, table_id="paramTable")
            self.params_table_html = str_io.getvalue()

            '''kn_fig, ax = plt.subplots(1, 1)
            ax.scatter(
                self.SSE_ensemble_params[solver][:, 0],
                self.SSE_ensemble_params[solver][:, 1],
                marker="o",
                edgecolors="k",
                facecolors="none",
                linewidths=0.5,
            )
            ax.scatter(
                self.opt_params[solver][0],
                self.opt_params[solver][1],
                marker="*",
                s=100,
                c="#0000ff",
                label="Optimum Decay Parameters",
            )
            if len(self.SSE_ensemble_params[solver]) > 1:
                ax.scatter(
                    self.CI_params[solver][0],
                    self.CI_params[solver][1],
                    edgecolors="#0000ff",
                    facecolors="none",
                    linewidths=0.5,
                    label="75% Confidence Region",
                )
                ax.scatter(
                    self.min_params[solver][0],
                    self.min_params[solver][1],
                    marker="*",
                    s=100,
                    c="#00ff00",
                    label="Minimum Decay Parameters",
                )
                ax.scatter(
                    self.max_params[solver][0],
                    self.max_params[solver][1],
                    marker="*",
                    s=100,
                    c="#ff0000",
                    label="Maximum Decay Parameters",
                )
            ax.set_title(solver)
            ax.set_xlabel(self.labels_dict[solver][0])
            ax.set_ylabel(self.labels_dict[solver][1])
            ax.legend()
            kn_fig.savefig(os.path.join(self.savepath, "params.png"))
            StringIOBytes_kn = io.BytesIO()
            kn_fig.savefig(StringIOBytes_kn, format="png", bbox_inches="tight")
            StringIOBytes_kn.seek(0)
            self.kn_base_64_pngData = base64.b64encode(StringIOBytes_kn.read())
            plt.close(kn_fig)'''
            return
        elif len(self.labels_dict[solver]) == 3:

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_series1_scatter.csv"),
                       np.transpose([self.SSE_ensemble_params[solver][:, 0],
                                     self.SSE_ensemble_params[solver][:, 1],
                                    self.SSE_ensemble_params[solver][:, 2]]),
                       delimiter=',',header="Slow Fast Ratio,Slow Decay Rate,Fast Decay Rate",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_2_plot_series2_scatter.csv"),
                       np.transpose([self.CI_params[solver][0],
                                     self.CI_params[solver][1],
                                     self.CI_params[solver][2]]),
                       delimiter=',',header="Upper 25 Slow Fast Ratio,Upper 25 Slow Decay Rate,Upper 25 Fast Decay Rate",comments='')
            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_optseries_scatter.csv"),
                       np.array([self.opt_params[solver][0], self.opt_params[solver][1],self.opt_params[solver][2]]).reshape(1,3),
                       delimiter=',',header="Optimum Slow Fast Ratio,Optimum Slow Decay Rate,Optimum Fast Decay Rate",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_minseries_scatter.csv"),
                       np.array([self.min_params[solver][0], self.min_params[solver][1],self.min_params[solver][2]]).reshape(1,3),
                       delimiter=',',header="Minimum Slow Fast Ratio,Minimum Slow Decay Rate,Minimum Fast Decay Rate",comments='')

            np.savetxt(os.path.join(self.savepath, "params_fig_1_plot_maxseries_scatter.csv"),
                       np.array([self.max_params[solver][0], self.max_params[solver][1],self.max_params[solver][2]]).reshape(1,3),
                       delimiter=',',header="Maximum Slow Fast Ratio,Maximum Slow Decay Rate,Maximum Fast Decay Rate",comments='')

            param_df = pd.DataFrame({"Ratio of Slow to Fast Decay (w)": [self.opt_params[solver][0], self.max_params[solver][0],
                                                        self.min_params[solver][0]],
                                     "Slow Decay Rate (k1)": [self.opt_params[solver][1], self.max_params[solver][1],
                                                         self.min_params[solver][1]],"Fast Decay Rate (k1)":[self.opt_params[solver][2], self.max_params[solver][2],
                                                         self.min_params[solver][2]]},
                                    index=['Optimum Decay', 'Maximum Decay', 'Minimum Decay'])
            param_df.index.name = 'Scenario'
            str_io = io.StringIO()
            param_df.to_html(buf=str_io, table_id="paramTable")
            self.params_table_html = str_io.getvalue()


            '''kn_fig, ax = plt.subplots(1, 1)
            plt.suptitle(solver)
            ax = kn_fig.add_subplot(221)
            ax.scatter(
                self.SSE_ensemble_params[solver][:, 1],
                self.SSE_ensemble_params[solver][:, 2],
                marker="o",
                edgecolors="k",
                facecolors="none",
                linewidths=0.5,
            )
            ax.scatter(
                self.opt_params[solver][1],
                self.opt_params[solver][2],
                marker="*",
                s=100,
                c="#0000ff",
                label="Optimum Decay Parameters",
            )
            if len(self.SSE_ensemble_params[solver]) > 1:
                ax.scatter(
                    self.CI_params[solver][0][1],
                    self.CI_params[solver][0][2],
                    edgecolors="#0000ff",
                    facecolors="none",
                    linewidths=0.5,
                    label="75% Confidence Region",
                )
                ax.scatter(
                    self.min_params[solver][1],
                    self.min_params[solver][2],
                    marker="*",
                    s=100,
                    c="#00ff00",
                    label="Minimum Decay Parameters",
                )
                ax.scatter(
                    self.max_params[solver][1],
                    self.max_params[solver][2],
                    marker="*",
                    s=100,
                    c="#ff0000",
                    label="Maximum Decay Parameters",
                )
            ax.set_xlabel(self.labels_dict[solver][1])
            ax.set_ylabel(self.labels_dict[solver][2])

            ax = kn_fig.add_subplot(223)
            ax.scatter(
                self.SSE_ensemble_params[solver][:, 0],
                self.SSE_ensemble_params[solver][:, 1],
                marker="o",
                edgecolors="k",
                facecolors="none",
                linewidths=0.5,
            )
            ax.scatter(
                self.opt_params[solver][0],
                self.opt_params[solver][1],
                marker="*",
                s=100,
                c="#0000ff",
            )
            if len(self.SSE_ensemble_params[solver]) > 1:
                ax.scatter(
                    self.CI_params[solver][0][0],
                    self.CI_params[solver][0][1],
                    edgecolors="#0000ff",
                    facecolors="none",
                )
                ax.scatter(
                    self.min_params[solver][0],
                    self.min_params[solver][1],
                    marker="*",
                    s=100,
                    c="#00ff00",
                )
                ax.scatter(
                    self.max_params[solver][0],
                    self.max_params[solver][1],
                    marker="*",
                    s=100,
                    c="#ff0000",
                )
            ax.set_xlabel(self.labels_dict[solver][0])
            ax.set_ylabel(self.labels_dict[solver][1])

            ax = kn_fig.add_subplot(222)
            ax.scatter(
                self.SSE_ensemble_params[solver][:, 0],
                self.SSE_ensemble_params[solver][:, 2],
                marker="o",
                edgecolors="k",
                facecolors="none",
                linewidths=0.5,
            )
            ax.scatter(
                self.opt_params[solver][0],
                self.opt_params[solver][2],
                marker="*",
                s=100,
                c="#0000ff",
            )
            if len(self.SSE_ensemble_params[solver]) > 1:
                ax.scatter(
                    self.CI_params[solver][0][0],
                    self.CI_params[solver][0][2],
                    edgecolors="#0000ff",
                    facecolors="none",
                    linewidths=0.5,
                )
                ax.scatter(
                    self.min_params[solver][0],
                    self.min_params[solver][2],
                    marker="*",
                    s=100,
                    c="#00ff00",
                )
                ax.scatter(
                    self.max_params[solver][0],
                    self.max_params[solver][2],
                    marker="*",
                    s=100,
                    c="#ff0000",
                )
            ax.set_xlabel(self.labels_dict[solver][0])
            ax.set_ylabel(self.labels_dict[solver][2])
            kn_fig.legend(bbox_to_anchor=(0.55, 0.45), loc="upper left")

            kn_fig.savefig(os.path.join(self.savepath, "params.png"))
            StringIOBytes_kn = io.BytesIO()
            kn_fig.savefig(StringIOBytes_kn, format="png", bbox_inches="tight")
            StringIOBytes_kn.seek(0)
            self.kn_base_64_pngData = base64.b64encode(StringIOBytes_kn.read())
            plt.close(kn_fig)'''
            return

    def confidence_assess(self):
        """Four checks:
        1. Observations
            High: >150
            Moderate: >100
            Low: <100
        2. Histogram
        3. Decay
        4. NSE

        self.confidence = {
            'Histogram': None,
            'Observations': None,
            'Decay Rate': None,
            'NSE': None
        }

        self.confidence_reason = {
            'Histogram': None,
            'Observations': None,
            'Decay Rate': None,
            'NSE': None
        }
        """
        # 1 - Check number of observations
        n_obs = len(self.Y_cal) + len(self.f_test)
        if n_obs >= 150:
            self.confidence["Observations"] = "High"
            self.confidence_reason[
                "Observations"
            ] = "At least 150 observations sent for analysis"
        elif n_obs >= 100:
            self.confidence["Observations"] = "Moderate"
            self.confidence_reason[
                "Observations"
            ] = "Between 100 and 150 observations sent for analysis"
        else:
            self.confidence["Observations"] = "Low"
            self.confidence_reason[
                "Observations"
            ] = "Fewer than 100 observations sent for analysis"

        # 2 - Check Variance Stability
        hh_frc_combined = np.append(self.Y_cal.values, self.f_test)
        ts_frc_combined = np.append(self.X_cal["ts_frc"].values, self.f0_test)
        ts_p_ks2 = []
        ts_p_levene = []
        hh_p_ks2 = []
        hh_p_levene = []
        for i in range(100):
            hh_check, drop, ts_check, drop2 = train_test_split(
                hh_frc_combined, ts_frc_combined, test_size=0.1

            )
            d, p = ks_2samp(ts_check, ts_frc_combined)
            ts_p_ks2.append(p)
            stat, p = levene(ts_check, ts_frc_combined)
            ts_p_levene.append(p)
            d, p = ks_2samp(hh_check, hh_frc_combined)
            hh_p_ks2.append(p)
            stat, p = levene(hh_check, hh_frc_combined)
            hh_p_levene.append(p)
        mins = [
            np.min(ts_p_ks2),
            np.min(ts_p_levene),
            np.min(hh_p_ks2),
            np.min(hh_p_levene),
        ]
        if np.min(mins) > 0.10:
            self.confidence["Stability"] = "High"
            self.confidence_reason[
                "Stability"
            ] = "Variance and distribution of household and tapstand FRC are not significantly different at a p-value level of 0.10"
        elif np.min(mins) > 0.05:
            self.confidence["Stability"] = "Moderate"
            self.confidence_reason[
                "Stability"
            ] = "Variance and distribution of household and tapstand FRC are not significantly different at a p-value 0.05 but are significantly different at a p-value of 0.10"
        else:
            self.confidence["Stability"] = "Low"
            self.confidence_reason[
                "Stability"
            ] = "Variance or distribution of household and tapstand FRC are significantly different stable at a p-value level of 0.05"

        # 3 - Check histogram densities
        if self.inputtime<=24:
            bin_centre = np.arange(3, 30, 3)
        else:
            bin_centre = np.arange(3, self.inputtime+12, 3)
        bins = bin_centre - 1.5
        hist = np.histogram(
            np.append(self.X_cal["se4_lag"].values, self.t_test).flatten(),
            bins=bins,
            density=True,
        )[0]
        density_hist = hist[np.argwhere(bin_centre == self.inputtime)]
        max_density = np.max(hist)
        if density_hist >= np.percentile(hist, 75):
            self.confidence["Histogram"] = "High"
            self.confidence_reason[
                "Histogram"
            ] = "Target storage duration and sampling durations are very similar (density of observations around storage target is higher than the upper quartile density)"
        elif density_hist >= np.percentile(hist, 50):
            self.confidence["Histogram"] = "Moderate"
            self.confidence_reason[
                "Histogram"
            ] = "Target storage duration are mostly similar (density of observations around storage target is higher than the median density)"
        elif density_hist > 0:
            self.confidence["Histogram"] = "Low"
            self.confidence_reason[
                "Histogram"
            ] = "Fewer than 25% of observations have sampling durations similar to the target storage duration"
        else:
            self.confidence["Histogram"] = "Low"
            self.confidence_reason[
                "Histogram"
            ] = "No observations have sampling durations similar to the target storage duration"

        # 4 - Check histogram uniformity

        uniform = [
            1 / len(np.arange(3, self.inputtime + 1, 3))
            for i in np.arange(3, self.inputtime + 1, 3)
        ]
        hist_test = (hist[np.argwhere(bin_centre <= self.inputtime)]).flatten()
        ks, p = ks_2samp(hist_test, uniform)
        if p >= 0.5:
            self.confidence["Uniformity"] = "High"
            self.confidence_reason[
                "Uniformity"
            ] = "Uniform distribution of samples up to target storage duration"
        elif p >= 0.05:
            self.confidence["Uniformity"] = "Moderate"
            self.confidence_reason[
                "Uniformity"
            ] = "Mostly uniform distribution of samples up to target storage duration"
        else:
            self.confidence["Uniformity"] = "Low"
            self.confidence_reason[
                "Uniformity"
            ] = "Minimal uniformity of distribution of samples up to target storage duration"

        # 5 - Check Target at 24 hours
        target_24 = self.targets[self.model][self.scenario].loc[24]
        if target_24 < 2:
            self.confidence["Decay Rate"] = "High"
            self.confidence_reason["Decay Rate"] = "FRC decay parameters are reasonable"
        else:
            self.confidence["Decay Rate"] = "Low"
            self.confidence_reason[
                "Decay Rate"
            ] = "FRC decay parameters are higher than normal. This may lead to higher FRC than acceptable FRC targets at long storage durations. This can occur normally in some sites or it may be due to a large number of short-duration samples. If the average storage duration is shorter than 6 hours, please collect additional samples at longer storage durations to see if this warning message changes"

        # 6 - Check R2/NSE
        if np.max(self.test_r2[self.model]) >= 0.8:
            self.confidence["Model Fit"] = "High"
            self.confidence_reason["Model Fit"] = "Model performance is good"
        elif np.max(self.test_r2[self.model]) > 0:
            self.confidence["Model Fit"] = "Moderate"
            self.confidence_reason["Model Fit"] = "Model performance is acceptable"
        else:
            self.confidence["Model Fit"] = "Low"
            self.confidence_reason["Model Fit"] = "Model performance is poor"

        confidence_df = pd.DataFrame(
            {
                "Confidence Level": self.confidence.values(),
                "Confidence Reason": self.confidence_reason.values(),
            },
            index=self.confidence.keys(),
        )
        confidence_df.to_csv(os.path.join(self.savepath, "confidence.csv"))
        str_io = io.StringIO()

        confidence_df.to_html(buf=str_io, table_id="confTable")
        confidence_html_str = str_io.getvalue()

        return confidence_html_str

    def get_targets(self):
        """Add in FRC targets over time fig for all three scenarios, call backcheck fig"""
        FRC_target = self.targets[self.model][self.scenario].loc[self.inputtime]
        self.target_fig(FRC_target)
        self.back_check_fig(FRC_target)
        return FRC_target

    def target_fig(self, target):
        csv_header = "Tapstand FRC,Household FRC"
        np.savetxt(os.path.join(self.savepath, "targets_fig_series1_line.csv"),
                   np.transpose([self.targets[self.model].index,self.targets[self.model][self.scenario]]),
                   delimiter=',',header=csv_header,comments='')
        hist_bars=np.histogram(np.append(self.X_cal["se4_lag"].values, self.t_test).flatten())
        widths=hist_bars[1][1]-hist_bars[1][0]
        xs=hist_bars[1][:-1]-widths/2
        np.savetxt(os.path.join(self.savepath, "targets_fig_series2_bar_width_"+str(widths)+".csv"),
                   np.transpose([xs,hist_bars[0]]),
                   delimiter=',',header="Tapstand FRC, Frequency Count",comments='')
        np.savetxt(os.path.join(self.savepath, "targets_fig_series3_vertline.csv"),
                   np.transpose([[self.inputtime,self.inputtime],[0,np.max(self.targets[self.model][self.scenario])]]),
                   delimiter=',',header=csv_header,comments='')

        str_io = io.StringIO()
        self.targets[self.model].index.name = 'Storage Duration'
        self.targets[self.model].to_html(buf=str_io, table_id="targetTable")
        self.targets_table_html=str_io.getvalue()


        '''target_fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_title("Required FRC Over Time", fontsize=10)
        ax.plot(
            self.targets[self.model].index,
            self.targets[self.model][self.scenario],
            c="#0000ff",
            label=self.scenario + " FRC Target",
            zorder=10,
        )
        secax = ax.twinx()
        secax.hist(
            np.append(self.X_cal["se4_lag"].values, self.t_test).flatten(),
            zorder=100,
            alpha=0.25,
            facecolor="purple",
        )
        secax.set_ylabel("Storage Duration Frequency")
        ax.axvline(self.inputtime, label="Time Target", c="k", ls="--")
        ax.legend(bbox_to_anchor=(0.999, 0.999), loc="upper right")
        ax.set_xlabel("Storage Duration (hours)")
        ax.set_ylabel("Required Tapstand FRC (mg/L)")
        target_fig.savefig(os.path.join(self.savepath, "targets.png"))
        StringIOBytes_target = io.BytesIO()
        target_fig.savefig(StringIOBytes_target, format="png", bbox_inches="tight")
        StringIOBytes_target.seek(0)
        self.target_base_64_pngData = base64.b64encode(StringIOBytes_target.read())
        plt.close(target_fig)'''

        if self.inputtime<=36:
            times = np.arange(self.inputtime, 37, 3)
        else:
            times = np.arange(self.inputtime, self.inputtime+12, 3)
        params = self.opt_params[self.model]
        pred = self.f_dict[self.model](
            params, np.array([target for i in range(len(times))]), times
        )

        np.savetxt(os.path.join(self.savepath, "decay_fig_series1_line.csv"),
                   np.transpose([times, pred]),
                   delimiter=',',header=csv_header,comments='')
        np.savetxt(os.path.join(self.savepath, "decay_fig_series2_horizline.csv"),
                   np.transpose([[np.max(times),np.min(times)],[0.2,0.2]]),
                   delimiter=',',header=csv_header,comments='')

        str_io2 = io.StringIO()
        df_decay=pd.DataFrame({"Predicted Household FRC":pred},index=times)
        df_decay.index.name='Storage Duration'
        df_decay.to_html(buf=str_io2, table_id="targetTable")
        self.decay_table_html = str_io2.getvalue()

        '''decay_fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_title("Household FRC After Target Storage Duration", fontsize=10)
        ax.plot(times, pred, c="b")

        ax.axhline(0.2, label="Time Target", c="k", ls="--")
        ax.set_xlabel("Storage Duration (hours)")
        ax.set_xticks(times)
        ax.set_xlim([self.inputtime, 36])

        ax.set_ylim([0, 0.3])
        ax.set_ylabel("Household FRC (mg/L)")
        decay_fig.savefig(os.path.join(self.savepath, "target_decay.png"))
        StringIOBytes_targetdecay = io.BytesIO()
        decay_fig.savefig(StringIOBytes_targetdecay, format="png", bbox_inches="tight")
        StringIOBytes_targetdecay.seek(0)
        self.targetdecay_base_64_pngData = base64.b64encode(
            StringIOBytes_targetdecay.read()
        )
        plt.close(decay_fig)'''
        return

    def back_check_fig(self, target):
        csv_header = "Tapstand FRC,Household FRC"
        time_lb = self.inputtime - 3
        time_ub = self.inputtime + 3
        test_df = pd.DataFrame(
            {"se4_lag": self.t_test, "ts_frc": self.f0_test, "hh_frc": self.f0_test}
        )
        test_df = test_df[test_df.se4_lag <= time_ub]
        test_df = test_df[test_df.se4_lag >= time_lb]
        test_idx = test_df.index

        cal_df = self.X_cal
        cal_df["hh_frc"] = self.Y_cal.to_numpy()

        cal_df = cal_df[cal_df.se4_lag <= time_ub]
        cal_df = cal_df[cal_df.se4_lag >= time_lb]

        n = len(cal_df.index) + len(test_df.index)
        s = (cal_df["se4_lag"].sum() + test_df["se4_lag"].sum()) / n

        sphere_df_test = test_df[test_df.ts_frc <= 0.5]
        sphere_df_test = sphere_df_test[sphere_df_test.ts_frc >= 0.2]
        sphere_total_test = len(sphere_df_test.index)
        sphere_safe_test = len(sphere_df_test[sphere_df_test.hh_frc >= 0.2].index)

        sphere_df_cal = cal_df[cal_df.ts_frc <= 0.5]
        sphere_df_cal = sphere_df_cal[sphere_df_cal.ts_frc >= 0.2]
        sphere_total_cal = len(sphere_df_cal.index)
        sphere_safe_cal = len(sphere_df_cal[sphere_df_cal.hh_frc >= 0.2].index)

        if (sphere_total_cal + sphere_total_test) > 0:
            sphere_safe_percent = np.round(
                (sphere_safe_cal + sphere_safe_test)
                / (sphere_total_cal + sphere_total_test)
                * 100,
                decimals=1,
            )
            self.sphere_text = (
                    "Existing Guidelines, 0.2-0.5 mg/L, "
                    + str(sphere_safe_cal + sphere_safe_test)
                    + " of "
                    + str(sphere_total_cal + sphere_total_test)
                    + ", "
                    + str(np.round(sphere_safe_percent, decimals=1))
                    + "% household water safety success rate"
            )
        else:
            self.sphere_text = "Existing Guidelines, 0.2-0.5 mg/L, Water safety success rate unavailable (no samples with similar storage duration and target)"

        target_df_test = test_df[test_df.ts_frc >= target]
        target_df_test = target_df_test[target_df_test.ts_frc <= target + 0.2]
        target_total_test = len(target_df_test.index)
        target_test_safe = len(target_df_test[target_df_test.hh_frc >= 0.2].index)

        target_cal_df = cal_df[cal_df.ts_frc >= target]
        target_cal_df = target_cal_df[target_cal_df.ts_frc <= target + 0.2]
        target_cal_total = len(target_cal_df.index)
        target_cal_safe = len(target_cal_df[target_cal_df.hh_frc >= 0.2].index)

        if (target_cal_total + target_total_test) > 0:
            target_safe_percent = (
                    (target_cal_safe + target_test_safe)
                    / (target_cal_total + target_total_test)
                    * 100
            )
            self.target_text = (
                    "Proposed Guidelines, "
                    + str(np.round(target, decimals=2))
                    + "-"
                    + str(np.round(target + 0.2, decimals=2))
                    + " mg/L, "
                    + str(target_cal_safe + target_test_safe)
                    + " of "
                    + str(target_cal_total + target_total_test)
                    + ", "
                    + str(np.round(target_safe_percent, decimals=1))
                    + "% household water safety success rate"
            )
        else:
            self.target_text = (
                    "Proposed Guidelines, "
                    + str(np.round(target, decimals=2))
                    + "-"
                    + str(np.round(target + 0.2, decimals=2))
                    + " mg/L, Water safety success rate unavailable (no samples with similar storage duration and target)"
            )

        box_props = dict(boxstyle="square", facecolor="white", alpha=0.5)
        frc_combined=np.append(test_df['hh_frc'].values,cal_df['hh_frc'].values)
        if len(frc_combined)>0:
            max_frc=np.max(frc_combined)
        else:
            max_frc=2
        np.savetxt(os.path.join(self.savepath, "backcheck_fig_series1_scatter.csv"),
                   np.transpose([test_df["ts_frc"].values,test_df["hh_frc"].values]),
                   delimiter=',',header=csv_header,comments='')
        np.savetxt(os.path.join(self.savepath, "backcheck_fig_series2_scatter.csv"),
                   np.transpose([cal_df["ts_frc"].values, cal_df["hh_frc"].values]),
                   delimiter=',',header=csv_header,comments='')

        np.savetxt(os.path.join(self.savepath, "backcheck_fig_series3_vertline.csv"),
                   np.transpose([[0.2,0.2],[0,max_frc]]),
                   delimiter=',',header=csv_header,comments='')
        np.savetxt(os.path.join(self.savepath, "backcheck_fig_series4_vertline.csv"),
                   np.transpose([[0.5, 0.5], [0, max_frc]]),
                   delimiter=',',header=csv_header,comments='')
        np.savetxt(os.path.join(self.savepath, "backcheck_fig_series5_vertline.csv"),
                   np.transpose([[target, target], [0, max_frc]]),
                   delimiter=',',header=csv_header,comments='')
        np.savetxt(os.path.join(self.savepath, "backcheck_fig_series6_vertline.csv"),
                   np.transpose([[target+0.2, target+0.2], [0, max_frc]]),
                   delimiter=',',header=csv_header,comments='')

        '''backcheck_fig, ax = plt.subplots(figsize=(8, 5.5))
        ax.set_title(
            "SWOT Engineering Optimization Model - Empirical Back-Check at "
            + str(time_lb)
            + "-"
            + str(time_ub)
            + "h follow-up (average "
            + str(np.round(s, decimals=1))
            + ", n="
            + str(n)
            + ")\nCode Version: "
            + self.version,
            fontsize=10,
        )
        # ax.text(0.01,0.95,sphere_text+'\n'+target_text,transform=ax.transAxes,bbox=box_props,fontsize=8,wrap=True)
        ax.scatter(
            test_df["ts_frc"],
            test_df["hh_frc"],
            marker="o",
            edgecolors="#0000ff",
            facecolors="none",
            linewidths=0.5,
        )
        ax.scatter(
            cal_df["ts_frc"],
            cal_df["hh_frc"],
            marker="o",
            edgecolors="#0000ff",
            facecolors="none",
            linewidths=0.5,
        )
        ax.axvline(0.2, color="#ff0000", ls="--")
        ax.axvline(0.5, color="#ff0000", ls="--")
        ax.axvline(target, color="#00ff00", ls="--")
        ax.axvline(target + 0.2, color="#00ff00", ls="--")
        ax.axhline(0.2, color="k", ls="--")
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
        ]
        ax.plot(lims, lims, "k-", linewidth=0.75)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        # plt.grid()
        ax.set_xlabel("Tapstand FRC (mg/L)")
        ax.set_ylabel("Household FRC (mg/L)")
        backcheck_fig.savefig(os.path.join(self.savepath, "backcheck.png"))
        StringIOBytes_backcheck = io.BytesIO()
        backcheck_fig.savefig(StringIOBytes_backcheck, format="png")
        StringIOBytes_backcheck.seek(0)
        self.backcheck_base_64_pngData = base64.b64encode(
            StringIOBytes_backcheck.read()
        )
        plt.close(backcheck_fig)'''
        return

    def select_model(self):
        """We could update this to use r2 instead/mse"""
        mse_test_sort = sorted(self.test_MSE, reverse=True)
        if self.convergence_checker[mse_test_sort[0]] == True:
            self.model = mse_test_sort[0]
        elif self.convergence_checker[mse_test_sort[1]] == True:
            self.model = mse_test_sort[1]
        else:
            self.model = "First Order"
        self.k_n_fig(self.model)
        return

    def generate_html_report(self, confidence):
        #targets = self.target_base_64_pngData.decode("UTF-8")
        #target_decay = self.targetdecay_base_64_pngData.decode("UTF-8")
        #backcheck = self.backcheck_base_64_pngData.decode("UTF-8")
        #k_n = self.kn_base_64_pngData.decode("UTF-8")

        st_dur = np.mean(np.append(self.X_cal["se4_lag"].values, self.t_test))

        doc, tag, text, line = Doc().ttl()
        with tag("h1", klass="title"):
            text("SWOT ENGINEERING OPTIMIZATION TOOL REPORT")
        with tag("p", klass="swot_version"):
            text("SWOT EO Version: " + self.version)
        with tag("h2", klass="Header"):
            text("Recommended Tapstand FRC Target: " + str(self.target) + "mg/L")
        with tag("p", klass="storage_target"):
            text("Target Storage Duration: " + str(self.inputtime) + " hours")
        with tag("p",klass="scenario"):
            text("Scenario: "+self.scenario)
        with tag("h2",klass="Header"):
            text("Required FRC over Time")
        with tag("table",id="targets_table"):
            doc.asis(self.targets_table_html)


        with tag("p", klass="model_selected"):
            text("Decay model used: " + self.model)

        with tag("h2",klass="model_params"):
            text("Model parameters")
        with tag("table",id="params_table"):
            doc.asis(self.params_table_html)
        #elif len(self.labels_dict[self.model]) == 2:
        with tag("p", klass="storage_target"):
            text("Target Storage Duration: " + str(self.inputtime) + " hours")
        with tag("p", klass="storage_duration"):
            text(
                "Average Storage Duration: "
                + str(int(np.floor(st_dur)))
                + " hours and "
                + str(int((st_dur - np.floor(st_dur)) * 60))
                + " minutes"
            )


        with tag("h2", klass="Header"):
            text(
                "Anticipated household FRC after storage target for tapstand FRC of "
                + str(self.target)
                + " mg/L"
            )
        with tag("table", id="target decay table"):
            doc.asis(self.decay_table_html)

        with tag("h2", klass="Header"):
            text("Empirical Water Safety Backcheck")
        with tag("p", klass="back_check_text_Sphere"):
            text(self.sphere_text)
        with tag("p", klass="back_check_text_Target"):
            text(self.target_text)

        #with tag("h2", klass="Header"):
        #    text("Selected Model Parameters")
        #with tag("div", id="params_fig"):
        #    doc.stag("img", src=(os.path.join(self.savepath, "params.png")))

        with tag("h2", klass="Header"):
            text("Model Confidence Assessment")
        with tag("table", id="confidence_assess"):
            doc.asis(confidence)

        file = open(os.path.join(self.savepath, "report.html"), "w+")
        file.write(doc.getvalue())
        file.close

        return

    def run_EO(self):
        self.import_data()
        np.array(self.First_Order_Model())
        np.array(self.Power_Decay_Model())
        np.array(self.Parallel_First_Order_Model())
        params = [self.opt_params[key] for key in self.opt_params.keys()]
        best_model_params = pd.Series(
            data=np.hstack((params)),
            index=[
                "Power Decay k",
                "Power Decay n",
                "First Order k",
                "Parallel First Order w",
                "Parallel First Order k1",
                "Parallel First Order k2",
            ],
        )
        self.select_model()
        self.target = np.round(self.get_targets(), decimals=1)
        conf_assess = self.confidence_assess()
        self.generate_html_report(conf_assess)

        results = {
            'best_model_params': best_model_params,
            'frc': self.target
        }

        return results
