
import pytest
from typing import List
import os
from pathlib import Path

from map_processing.data_set_models import UGDataSet


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
