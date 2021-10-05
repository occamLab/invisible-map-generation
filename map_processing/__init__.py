"""Package containing map processing capabilities.
"""

from enum import Enum

import numpy as np
from g2o import SE3Quat

ASSUMED_FOCAL_LENGTH = 1464

# TODO: send tag size with the tag detection
ASSUMED_TAG_SIZE = 0.172  # Tag's x and y dimension length in meters


# The ground truth tags for the 6-17-21 OCCAM Room. Keyed by tag ID. Measurements in meters (measurements taken in
# inches and converted to meters by multiplying by 0.0254. Measurements are in a right-handed coordinate system with its
# origin at the floor beneath tag id=0 (+Z pointing out of the wall and +X pointing to the right).
SQRT_2_OVER_2 = np.sqrt(2) / 2
OCCAM_ROOM_TAGS_DICT: np.ndarray = np.asarray([
    SE3Quat([0, 63.25 * 0.0254, 0, 0, 0, 0, 1]),
    SE3Quat([269 * 0.0254, 48.5 * 0.0254, -31.25 * 0.0254, 0, 0, 0, 1]),
    SE3Quat([350 * 0.0254, 58.25 * 0.0254, 86.25 * 0.0254, 0, SQRT_2_OVER_2, 0, -SQRT_2_OVER_2]),
    SE3Quat([345.5 * 0.0254, 58 * 0.0254, 357.75 * 0.0254, 0, 1, 0, 0]),
    SE3Quat([240 * 0.0254, 86 * 0.0254, 393 * 0.0254, 0, 1, 0, 0]),
    SE3Quat([104 * 0.0254, 31.75 * 0.0254, 393 * 0.0254, 0, 1, 0, 0]),
    SE3Quat([-76.75 * 0.0254, 56.5 * 0.0254, 316.75 * 0.0254, 0, SQRT_2_OVER_2, 0, SQRT_2_OVER_2]),
    SE3Quat([-76.75 * 0.0254, 54 * 0.0254, 75 * 0.0254, 0, SQRT_2_OVER_2, 0, SQRT_2_OVER_2])
])


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
