import EO_ens_SWOT
import os

from EO_ens_SWOT import EO_Ensemble

data_path = "C:\\Users\\micha\\Research\\SWOT\\Data\\Operational_Data\\Test_Sets\\All_test"
file = 'Test5.csv'
inputtime = 9
scenario='Minimum Decay' #Has to be 'Optimum Decay', 'Maximum Decay', or 'Minimum Decay'

eo_ens = EO_Ensemble(inputtime, "C:\\Users\\micha\\Research\\SWOT\\Data\\Operational_Data\\Test_Sets\\All_test",
                     file, scenario)
best_params=eo_ens.run_EO()
