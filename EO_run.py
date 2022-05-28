import sys

from swoteo.EO_ens_SWOT import EO_Ensemble

savepath = sys.argv[1]
inputpath = sys.argv[2]
inputtime = sys.argv[3]
scenario = sys.argv[4]  # Has to be 'optimumDecay', or 'minDecay', or 'maxDecay'

# savepath = "res"
# inputpath = "tests/test1.csv"
# inputtime = 3
# scenario = "optimumDecay"

eo_ens = EO_Ensemble(inputtime, savepath, inputpath, scenario)
eo_ens.run_EO()
