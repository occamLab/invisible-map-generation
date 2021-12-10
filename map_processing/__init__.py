"""Package containing map processing capabilities.
"""

from enum import Enum

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
