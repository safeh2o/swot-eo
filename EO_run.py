import EO_ens_SWOT
import os
from EO_ens_SWOT import EO_Ensemble

data_path = sys_arg1
file = sys_arg2
inputtime = sys_arg3
scenario=sys_arg4  #Has to be 'Optimum Decay', 'Maximum Decay', or 'Minimum Decay'

eo_ens = EO_Ensemble(inputtime,data_path,
                     file, scenario)
