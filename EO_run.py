import sys

from swoteo.EO_ens_SWOT import EO_Ensemble

data_path = "C:\\Users\\micha\\Research\\SWOT\\Operations\\Data\\Somaliland\\"
#files = [file.name for file in os.scandir(data_path) if file.is_file()]
files=['SOM_Piped_baseline_SWOT_150.csv']
for file in files:

    inputtime = 18
    scenario='maxDecay' #Has to be 'Optimum Decay', 'Maximum Decay', or 'Minimum Decay'

    eo_ens = EO_Ensemble(inputtime,data_path,
                         data_path+"\\"+file, scenario)
    best_params=eo_ens.run_EO()
