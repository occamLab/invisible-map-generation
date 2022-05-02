
import pytest
from typing import List
import os
from pathlib import Path
import numpy as np

from map_processing.data_models import UGDataSet, Weights, OG2oOptimizer, OConfig, OComputeInfParams

CURR_FILE_DIR = Path(os.path.abspath(__file__)).absolute().parent
TEST_FILES_DIR = os.path.join(CURR_FILE_DIR, "test_files")

test_ug_data_model_targets: List[str] = [
    os.path.join(TEST_FILES_DIR, "target_1.json"),
]


@pytest.mark.parametrize("targets", test_ug_data_model_targets)
def test_ug_data_model(targets: List[str]):
    if isinstance(targets, str):
        targets = [targets, ]
    for target in targets:
        with open(target, "r") as f:
            json_str = f.read()
            assert UGDataSet.parse_raw(json_str)


def test_weights_model():
    w = Weights()
    json_str = w.json(indent=2)
    Weights.parse_raw(json_str)


def test_og2o_optimizer():
    o = OG2oOptimizer(
        locations=np.random.randn(3, 9),
        tags=np.random.randn(6, 8),
        tagpoints=np.random.randn(4, 3),
        waypoints_arr=np.random.randn(3, 8),
        waypoints_metadata=[{}, {}, {}]
    )
    json_str = o.json(indent=2)
    OG2oOptimizer.parse_raw(json_str)


def test_oconfig():
    oconfig = OConfig(is_sba=False)
    json_str = oconfig.json(indent=2)
    OConfig.parse_raw(json_str)


def test_ocompute_inf_params():
    o = OComputeInfParams()
    json_str = o.json()
    OComputeInfParams.parse_raw(json_str)
