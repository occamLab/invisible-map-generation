"""
Models various serialized data sets with help from pydantic.

For more info on pydantic, visit: https://pydantic-docs.helpmanual.io/

Notes:
    Interpreting class name prefixes which describe how the models are used:
    - "GT" --> ground truth data
    - "UG" --> un-processed graph data
    - "PG" --> processed graph data
    - "O"  --> optimization-related data (either results or configuration)
"""

import itertools
from typing import List, Dict, Union, Optional

import numpy as np
from pydantic import BaseModel, conlist, Field, confloat, validator

from map_processing import ASSUMED_FOCAL_LENGTH, VertexType
from .transform_utils import FLIP_Y_AND_Z_AXES


def _is_vector_of_right_length(v: np.ndarray, length: int) -> np.ndarray:
    v_sqz = np.squeeze(v)
    if v_sqz.ndim != 1:
        raise ValueError(
            f"field that should have been a vector was found not to have the right dimensions (number of "
            f"dims found to be {v_sqz.ndim} after squeezing the array)")
    if v_sqz.size != length:
        raise ValueError(
            f"Expected vector to be of length {length} but instead found the length to be {v_sqz.size}")
    return v_sqz


def _validator_for_numpy_array_deserialization(v: Union[str, np.ndarray]):
    if isinstance(v, np.ndarray):
        return v
    elif isinstance(v, str):
        return np.fromstring(v.strip("[").strip("]"), sep=" ")
    else:
        raise ValueError(f"Attempted to parse value for an array-type field that is not handled: {type(v)}")


class Weights(BaseModel):
    gravity: np.ndarray = Field(default_factory=lambda: np.ones(3))
    odometry: np.ndarray = Field(default_factory=lambda: np.ones(6))
    tag: np.ndarray = Field(default_factory=lambda: np.ones(6))
    tag_sba: np.ndarray = Field(default_factory=lambda: np.ones(2))
    odom_tag_ratio: confloat(ge=0.00001) = 1.0

    class Config:
        arbitrary_types_allowed = True  # Needed to allow numpy arrays to be used as fields
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}

    @validator("odom_tag_ratio")
    def odom_tag_ratio_pre_validator(cls, v):
        if isinstance(v, np.ndarray):
            return np.squeeze(v)[0]
        return v

    # Vector validators
    _check_gravity_is_correct_length_vector = validator("gravity", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 3))
    _check_odometry_is_correct_length_vector = validator("odometry", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 6))
    _check_tag_is_correct_length_vector = validator("tag", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 6))
    _check_tag_sba_is_correct_length_vector = validator("tag_sba", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 2))
    _deserialize_gravity_vector_if_needed = validator("gravity", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)
    _deserialize_odometry_vector_if_needed = validator("odometry", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)
    _deserialize_tag_vector_if_needed = validator("tag", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)
    _deserialize_tag_sba_vector_if_needed = validator("tag_sba", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)

    @property
    def tag_odom_ratio(self):
        return 1 / self.odom_tag_ratio

    @classmethod
    def legacy_from_array(cls, array: Union[np.ndarray, List[float]]) -> "Weights":
        return Weights(**cls.legacy_weight_dict_from_array(array))

    @staticmethod
    def legacy_weight_dict_from_array(array: Union[np.ndarray, List[float]]) -> Dict[str, Union[float, np.ndarray]]:
        """Construct a normalized weight dictionary from a given array of values using the legacy approach.
        """
        weights = Weights().dict()
        length = array.size if isinstance(array, np.ndarray) else len(array)
        half_len = length // 2
        has_ratio = length % 2 == 1

        if length == 1:  # ratio
            weights['odom_tag_ratio'] = array[0]
        elif length == 2:  # tag/odom pose:rot/tag-sba x:y, ratio
            weights['odometry'] = np.array([array[0]] * 3 + [1] * 3)
            weights['tag'] = np.array([array[0]] * 3 + [1] * 3)
            weights['tag_sba'] = np.array([array[0], 1])
            weights['odom_tag_ratio'] = array[1]
        elif length == 3:  # odom pose:rot, tag pose:rot/tag-sba x:y, ratio
            weights['odometry'] = np.array([array[0]] * 3 + [1] * 3)
            weights['tag'] = np.array([array[1]] * 3 + [1] * 3)
            weights['tag_sba'] = np.array([array[1], 1])
            weights['odom_tag_ratio'] = array[2]
        elif half_len == 2:  # odom pose, odom rot, tag pose/tag-sba x, tag rot/tag-sba y, (ratio)
            weights['odometry'] = np.array([array[0]] * 3 + [array[1]] * 3)
            weights['tag'] = np.array([array[2]] * 3 + [array[3]] * 3)
            weights['tag_sba'] = np.array(array[2:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif half_len == 3:  # odom x y z qx qy, tag-sba x, (ratio)
            weights['odometry'] = np.array(array[:5])
            weights['tag_sba'] = np.array([array[5]])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 4:  # odom, tag-sba, (ratio)
            weights['odometry'] = np.array(array[:6])
            weights['tag_sba'] = np.array(array[6:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 5:  # odom x y z qx qy, tag x y z qx qy, (ratio)
            weights['odometry'] = np.array(array[:5])
            weights['tag'] = np.array(array[5:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 6:  # odom, tag, (ratio)
            weights['odometry'] = np.array(array[:6])
            weights['tag'] = np.array(array[6:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        else:
            raise Exception(f'Weight length of {length} is not supported')

        w = Weights(**weights)
        w.scale_tag_and_odom_weights(normalize=True)
        return w.dict()

    def scale_tag_and_odom_weights(self, normalize: bool = False):
        """Apply the odom-to-tag ratio as a scaling factor to the odometry and tag vectors; divide the tag_sba vector
         by the camera's focal length.

        Args:
            normalize: If true, add a multiplicative factor that is the reciprocal of each vector's magnitude.
        """
        if normalize:
            odom_mag = np.linalg.norm(self.odometry)
            if odom_mag == 0:  # Avoid divide by zero error
                odom_mag = 1

            sba_mag = np.linalg.norm(self.tag_sba)
            if sba_mag == 0:
                sba_mag = 1  # Avoid divide by zero error

            tag_mag = np.linalg.norm(self.tag)
            if tag_mag == 0:  # Avoid divide by zero error
                tag_mag = 1
        else:
            odom_mag = 1
            sba_mag = 1
            tag_mag = 1

        self.odometry *= self.odom_tag_ratio / odom_mag
        # TODO: The below implements what was previously in place for SBA weighting. Should it be changed? Why is
        #  such a low weighting so effective?
        self.tag_sba *= 1 / (sba_mag * ASSUMED_FOCAL_LENGTH)
        self.tag *= 1 / tag_mag

    def get_weights_from_end_vertex_mode(self, end_vertex_mode: Optional[VertexType]):
        """
        Args:
            end_vertex_mode: Mode of the end vertex of the edge

        Returns:
            A copy of the edge weight vector selected according to the mode of an edge's end vertex. An end vertex mode
             of type waypoint returns a vector of 1s.

        Raises:
            ValueError: If the end_vertex_mode is not recognized
        """
        if end_vertex_mode == VertexType.ODOMETRY:
            return np.array(self.odometry)
        elif end_vertex_mode == VertexType.TAG:
            return np.array(self.tag)
        elif end_vertex_mode == VertexType.TAGPOINT:
            return np.array(self.tag_sba)
        elif end_vertex_mode is None:
            return np.array(self.gravity)
        elif end_vertex_mode == VertexType.WAYPOINT:
            return np.ones(6)  # TODO: set to something other than identity?
        else:
            raise Exception(f"Edge of end type {end_vertex_mode} not recognized")


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

    # TODO: Add documentation for the following properties

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
        pose_ids: np.ndarray = self.pose_ids.flatten()
        sort_indices: np.ndarray = np.argsort(pose_ids)
        tag_ids_sorted_by_pose_ids = self.tag_ids.flatten()[sort_indices]
        num_unique_tag_ids = len(np.unique(tag_ids_sorted_by_pose_ids))
        poses_by_pose_ids: Dict[int, np.ndarray] = self.poses_by_pose_ids

        for i, tag_id in enumerate(tag_ids_sorted_by_pose_ids):
            if len(ret) == num_unique_tag_ids:
                break  # Additional looping will not add any new entries to the return value
            if tag_id not in ret:
                corresponding_pose_id = pose_ids[i]
                # For some reason, the linter thinks that sort_indices is an integer
                # noinspection PyUnresolvedReferences
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


class PGTranslation(BaseModel):
    x: float
    y: float
    z: float


class PGRotation(BaseModel):
    x: float
    y: float
    z: float
    w: float


class PGTagVertex(BaseModel):
    translation: PGTranslation
    rotation: PGRotation
    id: int


class PGOdomVertex(BaseModel):
    translation: PGTranslation
    rotation: PGRotation
    poseId: int
    adjChi2: Optional[float]
    vizTags: Optional[float]
    neighbors: Optional[List[int]]


class PGWaypointVertex(BaseModel):
    translation: PGTranslation
    rotation: PGRotation
    id: str


class PGDataSet(BaseModel):
    tag_vertices: List[PGTagVertex]
    odometry_vertices: List[PGOdomVertex]
    waypoints_vertices: List[PGWaypointVertex]


class OComputeInfParams(BaseModel):
    lin_vel_var: np.ndarray = Field(default_factory=lambda: np.ones(3))
    ang_vel_var: confloat(ge=0.00001) = 1.0

    class Config:
        arbitrary_types_allowed = True  # Needed to allow numpy arrays to be used as fields
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}

    _check_lin_vel_var_is_correct_length_vector = validator("lin_vel_var", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 3))
    _deserialize_lin_vel_var_vector_if_needed = validator("lin_vel_var", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)


class OConfig(BaseModel):
    """
    is_sba: True if SBA is being used.
    obs_chi2_filter: Removes from the graph (stored in the `graph` instance attribute) observation edges above
     this many standard deviations from the mean observation edge chi2 value in the optimized graph. The graph
     optimization is then re-run with the modified graph. A negative value performs no filtering.
    graph_plot_title: Plot title argument to pass to the visualization routine for the graph visualizations.
    chi2_plot_title: Plot title argument to pass to the visualization routine for the chi2 plot.
    compute_inf_params: Passed down to the `Edge.compute_information` method to specify the edge
     information computation parameters.
    scale_by_edge_amount: Passed on to the `scale_by_edge_amount` argument of the `Graph.set_weights` method. If
     true, then the odom:tag ratio is scaled by the ratio of tag edges to odometry edges
    """

    is_sba: bool
    obs_chi2_filter: float = -1
    compute_inf_params: Optional[OComputeInfParams]
    scale_by_edge_amount: bool = True
    weights: Weights = Weights()
    graph_plot_title: str = ""
    chi2_plot_title: str = ""
