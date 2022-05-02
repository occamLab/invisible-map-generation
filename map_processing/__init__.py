"""Package containing map processing capabilities.
"""

from enum import Enum
from typing import Dict, List
import numpy as np
from g2o import SE3Quat
from map_processing.transform_utils import transform_vector_to_matrix


ASSUMED_FOCAL_LENGTH = 1464

# TODO: send tag size with the tag detection
ASSUMED_TAG_SIZE = 0.172  # Tag's x and y dimension length in meters


class PrescalingOptEnum(Enum):
    """Enum used in the as_graph method to select which approach is taken

    Class attributes:
        USE_SBA: Do not use sparse bundle adjustment
        FULL_COV: When creating the tag edge prescaling matrix, compute it from the covariance matrix calculated
         to account for the reliability of the tag pose estimate.
        DIAG_COV: Same as `FULL_COV`, except only the matrix diagonal is used.
        ONES: Prescaling matrix is set to a matrix of 1s.
    """
    USE_SBA = 0
    FULL_COV = 1
    DIAG_COV = 2
    ONES = 3


class VertexType(Enum):
    """An enumeration containing the vertex types
    """
    ODOMETRY = 0
    TAG = 1
    TAGPOINT = 2
    WAYPOINT = 3


# noinspection GrazieInspection
SQRT_2_OVER_2 = np.sqrt(2) / 2
GT_TAG_DATASETS: Dict[str, Dict[int, np.ndarray]] = {
    "3line": {
        0: np.array([
            [1, 0, 0, -3],
            [0, 1, 0, 0],
            [0, 0, 1, -4],
            [0, 0, 0, 1]
        ]),
        1: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, -5],
            [0, 0, 0, 1]
        ]),
        2: np.array([
            [1, 0, 0, 3],
            [0, 1, 0, 0],
            [0, 0, 1, -4],
            [0, 0, 0, 1]
        ])
    },
    "occam": {
        # The ground truth tags for the 6-17-21 OCCAM Room. Keyed by tag ID. Measurements in meters (measurements
        # were taken in inches and converted to meters by multiplying by 0.0254). Measurements are in a right-handed
        # coordinate system with its origin at the floor beneath tag id=0 (+Z pointing out of the wall and +X
        # pointing to the right).
        0: transform_vector_to_matrix(SE3Quat([0, 63.25 * 0.0254, 0, 0, 0, 0, 1]).to_vector()),
        1: transform_vector_to_matrix(
            SE3Quat([269 * 0.0254, 48.5 * 0.0254, -31.25 * 0.0254, 0, 0, 0, 1]).to_vector()),
        2: transform_vector_to_matrix(
            SE3Quat([350 * 0.0254, 58.25 * 0.0254, 86.25 * 0.0254, 0, SQRT_2_OVER_2, 0,
                     -SQRT_2_OVER_2]).to_vector()),
        3: transform_vector_to_matrix(
            SE3Quat([345.5 * 0.0254, 58 * 0.0254, 357.75 * 0.0254, 0, 1, 0, 0]).to_vector()),
        4: transform_vector_to_matrix(
            SE3Quat([240 * 0.0254, 86 * 0.0254, 393 * 0.0254, 0, 1, 0, 0]).to_vector()),
        5: transform_vector_to_matrix(
            SE3Quat([104 * 0.0254, 31.75 * 0.0254, 393 * 0.0254, 0, 1, 0, 0]).to_vector()),
        6: transform_vector_to_matrix(
            SE3Quat([-76.75 * 0.0254, 56.5 * 0.0254, 316.75 * 0.0254, 0, SQRT_2_OVER_2, 0,
                     SQRT_2_OVER_2]).to_vector()),
        7: transform_vector_to_matrix(SE3Quat([-76.75 * 0.0254, 54 * 0.0254, 75 * 0.0254, 0, SQRT_2_OVER_2, 0,
                                               SQRT_2_OVER_2]).to_vector()),
    }
}

GROUND_TRUTH_MAPPING_STARTING_PT: Dict[str, List[str]] = {
    "occam": ["duncan-occam-room-10-1-21-2-38 267139330396791",
              "duncan-occam-room-10-1-21-2-48 26773176629225"]
}
"""
This dictionary is used as the default ground truth dataset-to-map-name mapping when one does not already exist
in the ground_truth/ sub-directory of the cache.
"""
