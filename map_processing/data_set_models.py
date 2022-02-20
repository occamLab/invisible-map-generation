"""
Models various serialized data sets with help from pydantic.

For more info on pydantic, visit: https://pydantic-docs.helpmanual.io/

Notes:
    Interpreting prefixes: GT --> ground truth. UG --> un-processed graph.
"""

from typing import List, Dict, Union
from pydantic import BaseModel, ValidationError, conlist
import numpy as np
import itertools
from .transform_utils import FLIP_Y_AND_Z_AXES


class UGPoseDatum(BaseModel):
    """Represents a single pose datum.
    """
    pose: conlist(Union[float, int], min_items=16, max_items=16)
    """
    Pose as a tuple of floats where reshaping into a 4x4 array using Fortran-like index order results in the transform 
    matrix. For more information on Fortran-like indexing from the numpy documentation: "...means to read / write the 
    elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest. 
    """
    timestamp: float
    planes: List = []
    id: int

    @property
    def pose_as_matrix(self) -> np.ndarray:
        return np.reshape(np.array(self.pose), (4, 4), order="F")

    @property
    def position(self) -> np.ndarray:
        return self.pose_as_matrix[:3, 3]

    def __repr__(self):
        return f"<{UGPoseDatum.__name__} id={self.id}> position(x,y,z)={tuple(self.position)}"


class UGTagDatum(BaseModel):
    """Represents a single tag observation datum.
    """

    tag_corners_pixel_coordinates: conlist(Union[float, int], min_items=8, max_items=8)
    """
    Values alternate between x and y coordinates in the camera frame. Tag corner order convention: Bottom right, bottom 
    left, top left, top right.
    """
    tag_id: int
    pose_id: int
    camera_intrinsics: conlist(Union[float, int], min_items=4, max_items=4)
    """
    Camera intrinsics in the order of: fx, fy, cx, cy
    """
    timestamp: float
    tag_pose: conlist(Union[float, int], min_items=16, max_items=16)
    tag_position_variance: conlist(Union[float, int], min_items=3, max_items=3) = [1, ] * 3
    tag_orientation_variance: conlist(Union[float, int], min_items=4, max_items=4) = [1, ] * 4
    joint_covar: conlist(Union[float, int], min_items=49, max_items=49) = [1, ] * 49

    @property
    def tag_pose_as_matrix(self) -> np.ndarray:
        return np.reshape(np.array(self.tag_pose), (4, 4), order="F")

    @property
    def obs_dist(self) -> float:
        return np.linalg.norm(self.tag_pose_as_matrix[:3, 3])

    def __repr__(self):
        return f"<{UGTagDatum.__name__} tag_id={self.tag_id} pose_id={self.pose_id} obs_dist={self.obs_dist}>"


class UGLocationDatum(BaseModel):
    transform: conlist(Union[float, int], min_items=16, max_items=16)
    """
    Pose as a tuple of floats where reshaping into a 4x4 array using C-like index order results in the transform 
    matrix. For more information on Fortran-like indexing from the numpy documentation: "means to read / write the 
    elements using C-like index order, with the last axis index changing fastest, back to the first axis index changing 
    slowest.
    """
    # TODO: validate assumption that this transform actually uses C-like indexing

    name: str
    timestamp: float
    pose_id: int


class UGDataSet(BaseModel):
    """Represents an unprocessed graph dataset.
    """
    location_data: List[UGLocationDatum] = []
    map_id: str
    plane_data: List = []
    pose_data: List[UGPoseDatum]
    tag_data: List[List[UGTagDatum]] = []

    @property
    def pose_data_len(self) -> int:
        return len(self.pose_data)

    @property
    def tag_data_len(self) -> int:
        return len(self.tag_data)

    def __repr__(self):
        return f"<{UGDataSet.__name__} map_id={self.map_id} pose_data_len={self.pose_data_len} " \
               f"tag_data_len={self.tag_data_len}>"

    @property
    def frame_ids_to_timestamps(self) -> Dict[int, float]:
        return {pose.id: pose.timestamp for pose in self.pose_data}

    @property
    def pose_matrices(self) -> np.ndarray:
        return np.array([pose_datum.pose for pose_datum in self.pose_data]).reshape((-1, 4, 4), order="F")

    @property
    def poses_by_pose_ids(self) -> Dict[int, np.ndarray]:
        ret: Dict[int, np.ndarray] = {}
        pose_matrices = self.pose_matrices
        for i, pose_datum in enumerate(self.pose_data):
            ret[pose_datum.id] = pose_matrices[i]
        return ret

    @property
    def tag_edge_measurements_matrix(self) -> np.ndarray:
        # The camera axis used to get tag measurements is flipped relative to the phone frame used for odom
        # measurements. Additionally, note that the matrix here is recorded in row-major format.
        return np.zeros((0, 4, 4)) if len(self.tag_data) == 0 else \
            np.matmul(FLIP_Y_AND_Z_AXES, np.vstack([[x.tag_pose for x in tags_from_frame] for tags_from_frame in
                                                    self.tag_data]).reshape([-1, 4, 4]))

    @property
    def timestamps(self) -> np.ndarray:
        return np.array([pose_datum.timestamp for pose_datum in self.pose_data])

    @property
    def approx_tag_in_global_by_id(self) -> Dict[int, np.ndarray]:
        """
        Returns:
            A dictionary mapping tag ids to their poses in the global reference frame according to the first observation
             of the tag.
        """
        ret = {}
        tag_edge_measurements_matrix = self.tag_edge_measurements_matrix
        pose_ids = self.pose_ids.flatten()
        sort_indices = np.argsort(pose_ids)
        tag_ids_sorted_by_pose_ids = self.tag_ids.flatten()[sort_indices]
        num_unique_tag_ids = len(np.unique(tag_ids_sorted_by_pose_ids))
        poses_by_pose_ids: Dict[int, np.ndarray] = self.poses_by_pose_ids

        for i, tag_id in enumerate(tag_ids_sorted_by_pose_ids):
            if len(ret) == num_unique_tag_ids:
                break  # Additional looping will not add any new entries to the return value
            if tag_id not in ret:
                corresponding_pose_id = pose_ids[i]
                ret[tag_id] = poses_by_pose_ids[corresponding_pose_id].dot(
                    tag_edge_measurements_matrix[sort_indices[i]])
        return ret

    @property
    def camera_intrinsics_for_tag(self) -> np.ndarray:
        return np.zeros((0, 4)) if len(self.tag_data) == 0 else \
            np.vstack([[x.camera_intrinsics for x in tags_from_frame] for tags_from_frame in self.tag_data])

    @property
    def tag_corners(self) -> np.ndarray:
        return np.zeros((0, 8)) if len(self.tag_data) == 0 else \
            np.vstack([[x.tag_corners_pixel_coordinates for x in tags_from_frame] for tags_from_frame in
                       self.tag_data])

    @property
    def tag_joint_covar(self) -> np.ndarray:
        # Note that the variance deviation of qw since we use a compact quaternion parameterization of orientation
        return np.zeros((0, 49), dtype=np.double) if len(self.tag_data) == 0 else \
            np.vstack([[x.joint_covar for x in tags_from_frame] for tags_from_frame in self.tag_data])

    @property
    def tag_joint_covar_matrices(self) -> np.ndarray:
        return self.tag_joint_covar.reshape((-1, 7, 7))

    @property
    def tag_position_variances(self) -> np.ndarray:
        return np.zeros((0, 3), dtype=np.double) if len(self.tag_data) == 0 else \
            np.vstack([[x.tag_position_variance for x in tags_from_frame] for tags_from_frame in self.tag_data])

    @property
    def tag_orientation_variances(self) -> np.ndarray:
        return np.zeros((0, 4), dtype=np.double) if len(self.tag_data) == 0 else \
            np.vstack([[x.tag_orientation_variance for x in tags_from_frame] for tags_from_frame in self.tag_data])

    @property
    def tag_ids(self) -> np.ndarray:
        return np.zeros((0, 1), dtype=np.int64) if len(self.tag_data) == 0 else \
            np.vstack(list(itertools.chain(*[[x.tag_id for x in tags_from_frame] for tags_from_frame in
                                             self.tag_data])))

    @property
    def pose_ids(self) -> np.ndarray:
        return np.zeros((0, 1), dtype=np.int64) if len(self.tag_data) == 0 else \
            np.vstack(list(itertools.chain(*[[x.pose_id for x in tags_from_frame] for tags_from_frame in
                                             self.tag_data])))

    @property
    def waypoint_names(self) -> List[str]:
        return [location_data.name for location_data in self.location_data]

    @property
    def waypoint_edge_measurements_matrix(self) -> np.ndarray:
        return np.zeros((0, 4, 4)) if len(self.location_data) == 0 else \
            np.concatenate(
                [np.asarray(location_data.transform).reshape((-1, 4, 4)) for location_data in self.location_data]
            )

    @property
    def waypoint_frame_ids(self) -> List[int]:
        return [location_data.pose_id for location_data in self.location_data]


class GTTagPose(BaseModel):
    tag_id: int
    pose: conlist(Union[float, int], min_items=7, max_items=7)


class GTDataSet(BaseModel):
    poses: List[GTTagPose]


if __name__ == "__main__":
    with open("example.json", "r") as e:
        example_json = e.read()
        try:
            ugj = UGDataSet.parse_raw(example_json)
        except ValidationError as ve:
            print(ve.json())
        print(ugj.json(indent=2))
