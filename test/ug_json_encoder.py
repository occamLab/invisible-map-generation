"""
Contains the UGJsonEncoder class and the necessary dependencies.
"""

import json
from typing import *
import numpy as np

from type_checking_json_encoder import TypeCheckingJSONEncoder

Flattened4x4MatrixTuple = Tuple[
    float, float, float, float,
    float, float, float, float,
    float, float, float, float,
    float, float, float, float
]
Flattened7x7MatrixTuple = Tuple[
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
    float, float, float, float, float, float, float,
]
Flattened4x2MatrixTuple = Tuple[
    float, float,
    float, float,
    float, float,
    float, float
]
Length4VectorTuple = Tuple[float, float, float, float]
Length3VectorTuple = Tuple[float, float, float]


class UGPoseDatum(TypeCheckingJSONEncoder):
    """Represents a single pose datum.
    """

    type_pose: Type = Flattened4x4MatrixTuple
    type_timestamp: Type = float
    type_planes: Type = List
    type_id: Type = int

    def __init__(self, pose: Optional[Flattened4x4MatrixTuple] = None, timestamp: Optional[float] = None,
                 planes: Optional[List] = None, pose_id: Optional[int] = None):
        self.pose: Flattened4x4MatrixTuple = pose if pose is not None else (0.0,) * 16
        self.timestamp: float = timestamp if timestamp is not None else 0.0
        self.planes: List = planes if planes is not None else []
        self.id: int = pose_id if pose_id is not None else -1
        super().__init__()

    @property
    def pose_as_matrix(self) -> np.ndarray:
        return np.reshape(np.array(self.pose), (4, 4), order="F")

    @property
    def position(self) -> np.ndarray:
        return self.pose_as_matrix[:3, 3]

    def __repr__(self):
        return f"<id={self.id}> position(x,y,z)={tuple(self.position)}"


class UGTagDatum(TypeCheckingJSONEncoder):
    """Represents a single tag observation datum.
    """

    type_tag_corners_pixel_coordinates: Type = Flattened4x2MatrixTuple
    """
    Values alternate between x and y coordinates in the camera frame. Tag corner order convention: Bottom right, bottom 
    left, top left, top right.
    """

    type_tag_id: Type = int
    type_pose_id: Type = int

    type_camera_intrinsics: Type = Length4VectorTuple
    """
    Camera intrinsics in the order of: fx, fy, cx, cy
    """

    type_tag_position_variance: Type = Length3VectorTuple
    type_tag_orientation_variance: Type = Length4VectorTuple
    type_timestamp: Type = float
    type_tag_pose: Type = Flattened4x4MatrixTuple
    type_joint_covar: Type = Flattened7x7MatrixTuple

    def __init__(self, tag_corners_pixel_coordinates: Optional[Flattened4x2MatrixTuple] = None,
                 tag_id: Optional[int] = None,
                 pose_id: Optional[int] = None,
                 camera_intrinsics: Optional[Length4VectorTuple] = None,
                 tag_position_variance: Optional[Length3VectorTuple] = None,
                 tag_orientation_variance: Optional[Length4VectorTuple] = None,
                 timestamp: Optional[float] = None,
                 tag_pose: Optional[Flattened4x4MatrixTuple] = None,
                 joint_covar: Optional[Flattened7x7MatrixTuple] = None):
        self.tag_corners_pixel_coordinates: Flattened4x2MatrixTuple = tag_corners_pixel_coordinates if \
            tag_corners_pixel_coordinates is not None else (0.0,) * 8
        self.tag_id: int = tag_id if tag_id is not None else -1
        self.pose_id: int = pose_id if pose_id is not None else -1
        self.camera_intrinsics: Length4VectorTuple = camera_intrinsics if camera_intrinsics is not None else (0.0,) * 4
        self.tag_position_variance: Length3VectorTuple = tag_position_variance if tag_position_variance is not None \
            else (0.0,) * 3
        self.tag_orientation_variance: Length4VectorTuple = tag_orientation_variance if tag_orientation_variance is \
            not None else (0.0,) * 4
        self.timestamp: float = timestamp if timestamp is not None else 0.0
        self.tag_pose: Flattened4x4MatrixTuple = tag_pose if tag_pose is not None else (0.0,) * 16
        self.joint_covar: Flattened7x7MatrixTuple = joint_covar if joint_covar is not None else (0.0,) * 49
        super().__init__()

    @property
    def tag_pose_as_matrix(self) -> np.ndarray:
        return np.reshape(np.array(self.tag_pose), (4, 4), order="F")

    @property
    def obs_dist(self) -> float:
        return np.linalg.norm(self.tag_pose_as_matrix[:3, 3])

    def __repr__(self):
        return f"<tag_id={self.tag_id} pose_id={self.pose_id} obs_dist={self.obs_dist}>"


class UGJsonEncoder(TypeCheckingJSONEncoder):
    """Represents an unprocessed graph dataset.

    Notes:
        Per the interface implemented by the TypeCheckingJSONEncoder superclass, this class's instances can be
        serialized into JSON though the following process:

        >>> json_dict = ug_json.default("example.json", "w")
        >>> json_str = str(json_dict)

        For more information about what attributes are included in the dictionary and the different checks that occur
        along the way, read the documentation of TypeCheckingJSONEncoder.
    """

    type_location_data: Type = List
    type_map_id: Type = str
    type_plane_data: Type = List
    type_pose_data: Type = List[UGPoseDatum]
    type_tag_data: Type = List[List[UGTagDatum]]

    def __init__(self, location_data: Optional[List] = None, map_id: Optional[str] = None,
                 plane_data: Optional[List] = None, pose_data: Optional[List[UGPoseDatum]] = None,
                 tag_data: Optional[List[List[UGTagDatum]]] = None):
        self.location_data: List = location_data if location_data is not None else []
        self.map_id: str = map_id if map_id is not None else -1
        self.plane_data: List = plane_data if plane_data is not None else []
        self.pose_data: List[UGPoseDatum] = pose_data if pose_data is not None else [UGPoseDatum()]
        self.tag_data: List[List[UGTagDatum]] = tag_data if tag_data is not None else [[UGTagDatum()]]
        super().__init__()

    @property
    def pose_data_len(self) -> int:
        return len(self.pose_data)

    @property
    def tag_data_len(self) -> int:
        return len(self.tag_data)

    def __repr__(self):
        return f"<map_id={self.map_id} pose_data_len={self.pose_data_len} tag_data_len={self.tag_data_len}>"


if __name__ == "__main__":
    ug_json = UGJsonEncoder()
    with open("example.json", "w") as e:
        d = ug_json.default(ug_json)
        print(d)
        json.dump(d, e, indent=2)
