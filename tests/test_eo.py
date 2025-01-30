import os, shutil

import pytest
import glob

# Import the ensemble and instantiate
from swoteo.EO_ens_SWOT import EO_Ensemble

testspath = os.path.dirname(__file__)
test_files = glob.glob(os.path.join(testspath, "test*.csv"))
output_prefix = "out"
STORAGE_TARGET = 3


def test_run_harness():
    for file in test_files:
        eo = EO_Ensemble(STORAGE_TARGET, output_prefix, file, 'optimumDecay')

        eo.run_EO()
        shutil.rmtree(output_prefix)


pytest.main()
