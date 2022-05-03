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
from typing import List, Dict, Union, Optional, Tuple

import numpy as np
from g2o import SE3Quat
from matplotlib import pyplot as plt
from pydantic import BaseModel, conlist, Field, confloat, validator

from map_processing import ASSUMED_FOCAL_LENGTH, VertexType
from map_processing.transform_utils import NEGATE_Y_AND_Z_AXES, transform_matrix_to_vector


def _is_arr_of_right_shape(v: Optional[np.ndarray], shape: Tuple[int, ...], is_optional: bool = False):
    expected_num_dims = len(shape)

    if v is None:
        if is_optional:
            return v
        else:
            raise ValueError("Value provided that was marked as non-optionally None is None")

    if np.any(v == np.nan):
        raise ValueError("Numpy array cannot contain any NaN values")

    v_sqz: np.ndarray = np.squeeze(v)
    if v_sqz.ndim != expected_num_dims:
        raise ValueError(
            f"Field that should have been an array was found to not have the right dimensions (number of dims found to "
            f"be {v_sqz.ndim} after squeezing the array)"
        )
    for dim_idx, dim in enumerate(shape):
        if 0 <= shape[dim_idx] != v_sqz.shape[dim_idx]:
            raise ValueError(
                f"Field that should have had an array of shape {shape} had a shape of {v_sqz.shape} (note that "
                f"negative expected dimensions, if there are any, mean that the matrix can be of any size along that "
                f"axis)"
            )
    return v_sqz


def _is_vector_of_right_length(v: np.ndarray, length: int) -> np.ndarray:
    v_sqz: np.ndarray = np.squeeze(v)

    if np.any(v == np.nan):
        raise ValueError("Numpy array cannot contain any NaN values")

    if v_sqz.ndim != 1:
        raise ValueError(
            f"field that should have been a vector was found to not have the right dimensions (number of dims found to "
            f"be {v_sqz.ndim} after squeezing the array)")
    if v_sqz.size != length:
        raise ValueError(
            f"Expected vector to be of length {length} but instead found the length to be {v_sqz.size}")
    return v_sqz


def _validator_for_numpy_array_deserialization(v: Union[str, np.ndarray]) -> \
        np.ndarray:
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

    @classmethod
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
    tag_position_variance: conlist(Union[float, int], min_items=3, max_items=3) = [0, ] * 3
    tag_orientation_variance: conlist(Union[float, int], min_items=4, max_items=4) = [0, ] * 4
    joint_covar: conlist(Union[float, int], min_items=49, max_items=49) = list(np.eye(7).flatten())

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
    def num_tags(self) -> int:
        return len(np.unique(self.tag_ids))

    @property
    def num_observations(self) -> int:
        num_observations = 0
        for tag_obs_list in self.tag_data:
            num_observations += len(tag_obs_list)
        return num_observations

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
            np.matmul(NEGATE_Y_AND_Z_AXES, np.vstack([[x.tag_pose for x in tags_from_frame] for tags_from_frame in
                                                      self.tag_data]).reshape([-1, 4, 4], order="C"))

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

    @property
    def pose_as_se3quat(self) -> SE3Quat:
        return SE3Quat(np.array([self.pose[i] for i in range(7)]))

    @property
    def pose_as_se3_array(self) -> np.ndarray:
        return self.pose_as_se3quat.to_vector()


class GTDataSet(BaseModel):
    poses: List[GTTagPose] = []

    @property
    def sorted_poses_as_se3quat_list(self) -> List[SE3Quat]:
        return [pose.pose_as_se3quat for pose in sorted(self.poses, key=lambda pose: pose.tag_id)]

    @property
    def pose_ids_as_list(self) -> List[int]:
        return [tag_pose.tag_id for tag_pose in self.poses]

    @property
    def as_dict_of_se3_arrays(self) -> Dict[int, np.ndarray]:
        return {tag_pose.tag_id: tag_pose.pose_as_se3_array for tag_pose in self.poses}
    
    @classmethod
    def gt_data_set_from_dict_of_arrays(cls, dct: Dict[int, np.ndarray]) -> "GTDataSet":
        """Generate a GTDataSet from a dict of tag data.

        Notes:
            Arbitrarily selects one of the poses to be the origin of the data set (i.e., one of the pose transforms in
            the resulting data set will be the identity).

        Args:
            dct: Dictionary mapping tag IDs to their poses in some arbitrary reference frame. The poses are expected to
             either 4x4 matrices or length-7 SE3 vectors.
        """
        if len(dct) == 0:
            return GTDataSet()

        # create new dict in case se3 vectors need to be converted to 4x4 transform matrices
        new_dict: Dict[int, np.ndarray] = {}
        for item in dct.items():
            if item[1].shape == (4, 4):
                new_dict[item[0]] = SE3Quat(transform_matrix_to_vector(item[1])).to_vector()
            else:
                new_dict[item[0]] = item[1]

        ground_truth_tags = []
        for item in new_dict.items():
            ground_truth_tags.append(
                GTTagPose(
                    tag_id=item[0],
                    pose=item[1].tolist()
                )
            )
        return GTDataSet(poses=ground_truth_tags)


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
    Class Attributes:
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

    # Need this class with the `json_encoders` field to be present so that the sub-models' (`Weights` and
    # OComputeInfParams) numpy arrays can be serializable, even though in isolation they are already serializable.
    class Config:
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}

    @classmethod
    def oconfig_sweep_generator(cls, base_oconfig: "OConfig", product_args: List[np.ndarray]) -> \
            Tuple[List[float], "OConfig"]:
        """Generator that yields OConfig objects according to the cartesian product of the `*_sweep` arguments.

        Notes:
             What is meant by "`*_sweep` arguments" is all the arguments whose parameter names end with "_sweep".

        Args:
            base_oconfig: Any optimization configuration parameters not covered by the `*_sweep` are inherited from this
             model.
            product_args: List of arrays to be used as the argument to the cartesian product function.
        """
        len_3_unit_vec = np.ones(3) * np.sqrt(1 / 3)
        for this_product in itertools.product(*product_args, repeat=1):
            yield this_product, OConfig(
                is_sba=base_oconfig.is_sba,
                obs_chi2_filter=base_oconfig.obs_chi2_filter,
                compute_inf_params=OComputeInfParams(
                    lin_vel_var=this_product[0] * np.ones(3),
                    ang_vel_var=this_product[1],
                ),
                scale_by_edge_amount=base_oconfig.scale_by_edge_amount,
                weights=Weights(
                    gravity=len_3_unit_vec * this_product[2],
                    odom_tag_ratio=this_product[3],
                ),
                graph_plot_title=base_oconfig.graph_plot_title,
                chi2_plot_title=base_oconfig.chi2_plot_title,
            )

    def __hash__(self):
        return self.json().__hash__()


class OG2oOptimizer(BaseModel):
    """
    Class Attributes:
        locations: (n, 9) array containing x, y, z, qx, qy, qz, qw locations of the phone as well as the vertex uid at
         n points.
        locationsAdjChi2: Optionally associated with each odometry node is a chi2 calculated from the
         `map_odom_to_adj_chi2` method of the `Graph` class, which is stored in this vector.
    """

    locations: np.ndarray = Field(default_factory=lambda: np.zeros((0, 9)))
    tags: np.ndarray = Field(default_factory=lambda: np.zeros((0, 8)))
    tagpoints: np.ndarray = Field(default_factory=lambda: np.zeros((0, 3)))
    waypoints_arr: np.ndarray = Field(default_factory=lambda: np.zeros((0, 8)))
    waypoints_metadata: List[Dict]
    locationsAdjChi2: Optional[np.ndarray] = None
    visibleTagsCount: Optional[np.ndarray] = None

    class Config:
        arbitrary_types_allowed = True  # Needed to allow numpy arrays to be used as fields
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr.flatten(order="C"))}

    _check_locations_is_correct_shape_matrix = validator("locations", allow_reuse=True)(
        lambda v: _is_arr_of_right_shape(v, (-1, 9)))
    _check_tags_is_correct_shape_matrix = validator("tags", allow_reuse=True)(
        lambda v: _is_arr_of_right_shape(v, (-1, 8)))
    _check_tagpoints_is_correct_shape_matrix = validator("tagpoints", allow_reuse=True)(
        lambda v: _is_arr_of_right_shape(v, (-1, 3)))
    _check_waypoints_arr_is_correct_shape_matrix = validator("waypoints_arr", allow_reuse=True)(
        lambda v: _is_arr_of_right_shape(v, (-1, 8)))
    _check_locationsAdjChi2_is_correct_shape_matrix = validator("locationsAdjChi2", allow_reuse=True)(
        lambda v: _is_arr_of_right_shape(v, (-1, 1), is_optional=True))
    _check_visibleTagsCount_is_correct_shape_matrix = validator("visibleTagsCount", allow_reuse=True)(
        lambda v: _is_arr_of_right_shape(v, (-1, 1), is_optional=True))

    _deserialize_locations_matrix_if_needed = validator("locations", allow_reuse=True, pre=True)(
        lambda v: _validator_for_numpy_array_deserialization(v).reshape([-1, 9]))
    _deserialize_tags_matrix_if_needed = validator("tags", allow_reuse=True, pre=True)(
        lambda v: _validator_for_numpy_array_deserialization(v).reshape([-1, 8]))
    _deserialize_tagpoints_matrix_if_needed = validator("tagpoints", allow_reuse=True, pre=True)(
        lambda v: _validator_for_numpy_array_deserialization(v).reshape([-1, 3]))
    _deserialize_waypoints_arr_matrix_if_needed = validator("waypoints_arr", allow_reuse=True, pre=True)(
            lambda v: _validator_for_numpy_array_deserialization(v).reshape([-1, 8]) if v is not None else None)
    _deserialize_visibleTagsCount_matrix_if_needed = validator("visibleTagsCount", allow_reuse=True, pre=True)(
        lambda v: _validator_for_numpy_array_deserialization(v).reshape([-1, 1]) if v is not None else None)



class OResultChi2Values(BaseModel):
    chi2_all_before: float
    chi2_gravity_before: float
    chi2_all_after: float
    chi2_gravity_after: float

    # Ignore warning about first argument not being self (decorating as a @classmethod appears to prevent validation for
    # some reason...)
    # noinspection PyMethodParameters
    @validator("*")
    def validate_float_is_geq_0(cls, v):
        if isinstance(v, float):
            assert (v >= 0, "all floating point members must be positive")
        return v


class OResult(BaseModel):
    oconfig: OConfig
    map_pre: OG2oOptimizer
    map_opt: OG2oOptimizer
    chi2s: OResultChi2Values
    gt_metric_opt: Optional[float] = None
    gt_metric_pre: Optional[float] = None

    # Need this class with the `json_encoders` field to be present so that the contained numpy arrays can be
    # serializable, even though the models this model is composed of are already serializable on their own.
    class Config:
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}


class OSGPairResult(BaseModel):
    sg1_oresult: OResult
    sg2_oresult: OResult

    # Need this class with the `json_encoders` field to be present so that the contained numpy arrays can be
    # serializable, even though the models this model is composed of are already serializable on their own.
    class Config:
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}

    @property
    def chi2_diff(self) -> float:
        return self.sg2_oresult.chi2s.chi2_all_after - self.sg1_oresult.chi2s.chi2_all_after

class OSweepResults(BaseModel):
    gt_results_arr_shape: List[int]
    sweep_config: Dict[str, List[float]]
    gt_results_list: List[float]
    sweep_config_keys_order: List[str]
    base_oconfig: OConfig

    # Need this class with the `json_encoders` field to be present so that the base_oconfig's numpy arrays can be
    # serializable, even though in isolation base_oconfig is already serializable.
    class Config:
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}

    # Ignore warning about first argument not being self (decorating as a @classmethod appears to prevent validation for
    # some reason...)
    # noinspection PyMethodParameters
    @validator("sweep_config_keys_order")
    def sweep_config_keys_must_be_same_as_sweep_config_keys_order(cls, v, values):
        sweep_config = values["sweep_config"]
        for key in v:
            if key not in sweep_config:
                raise ValueError("sweep_config_keys_order contains a string that is not a key in the sweep_config dict")
        if len(sweep_config) != len(v):
            raise ValueError("the number of items in the sweep_config dictionary must be the same as the number of "
                             "items in the sweep_config_keys_order list")
        return v

    # Ignore warning about first argument not being self (decorating as a @classmethod appears to prevent validation for
    # some reason...)
    # noinspection PyMethodParameters
    @validator("gt_results_list")
    def validate_length_of_gt_results_list(cls, v, values):
        gt_results_arr_shape = values["gt_results_arr_shape"]
        expected_length = np.product(gt_results_arr_shape)
        if len(v) != expected_length:
            raise ValueError(f"gt_results_list cannot be of length {len(v)} if gt_results_arr_shape is "
                             f"{gt_results_arr_shape} (expected length is {expected_length})")
        return v

    @property
    def sweep_config_dict(self) -> Dict[str, List[float]]:
        return {item[0]: item[1] for item in self.sweep_config}

    @property
    def sweep_variables(self) -> List[str]:
        return [item[0] for item in self.sweep_config]

    @property
    def gt_results_arr(self) -> np.ndarray:
        return np.array(self.gt_results_list).reshape(self.gt_results_arr_shape, order="C")

    @property
    def min_gt_result(self) -> float:
        return np.min(self.gt_results_list)

    @property
    def where_min(self) -> Tuple[int, ...]:
        # noinspection PyTypeChecker
        where_min_pre: Tuple[np.ndarray, np.ndarray, np.ndarray] = np.where(self.gt_results_arr == self.min_gt_result)
        return tuple([arr[0] for arr in where_min_pre])  # Select first result if there are multiple

    @property
    def args_producing_min(self) -> Dict[str, float]:
        args_producing_min: Dict[str, float] = {}
        for i, key in enumerate(self.sweep_config_keys_order):
            args_producing_min[key] = np.array(self.sweep_config[key])[self.where_min[i]]
        return args_producing_min

    def visualize_results_heatmap(self) -> plt.Figure:
        """Generate (but do not show) a figure of subplots where each subplot shows a heatmap of the ground truth metric
        for a 2D slice of the search space.

        Raises:
            Exception: If the number of dimensions swept is <2.
        """
        if len(self.sweep_config_keys_order) < 2:
            raise Exception("Cannot create heatmap of results as implemented when <2 variables are swept.")

        num_vars = len(self.sweep_config_keys_order)

        # Generate all possible combinations of slices
        idcs_plot_against_list: List[Tuple[int, int]] = []
        for idx_1 in range(num_vars):
            for idx_2 in range(idx_1 + 1, num_vars):
                idcs_plot_against_list.append((idx_1, idx_2))

        # Figure out dimensions of subplot grid
        subplot_height = int(np.floor(np.sqrt(num_vars)))
        subplot_width = subplot_height
        if num_vars % subplot_width != 0:
            subplot_width += 1

        # In each subplot, make a heatmap from the 2D cross-section of the search space that intersects the minimum
        # value
        where_min = self.where_min
        fig, axs = plt.subplots(subplot_height, subplot_width, figsize=(8, 8), constrained_layout=True)
        for i, ax in enumerate(axs.flat):
            idcs_plot_against = idcs_plot_against_list[i]

            x_vec = self.sweep_config[self.sweep_config_keys_order[idcs_plot_against[0]]]
            y_vec = self.sweep_config[self.sweep_config_keys_order[idcs_plot_against[1]]]

            xx, yy = np.meshgrid(x_vec, y_vec)
            plot_against_dims = sorted(list(set(range(len(self.sweep_config_keys_order))).difference(
                set(idcs_plot_against))))
            zz = self.gt_results_arr.take(indices=where_min[plot_against_dims[0]], axis=plot_against_dims[0])
            zz = zz.take(indices=where_min[plot_against_dims[1] - 1], axis=plot_against_dims[1] - 1).T
            ax.set_xlabel(self.sweep_config_keys_order[idcs_plot_against[0]])
            ax.set_ylabel(self.sweep_config_keys_order[idcs_plot_against[1]])
            c = ax.pcolormesh(xx, yy, zz, shading="auto")

            # Annotate the heatmap with a red dot showing the coordinates that produce the minimum value
            ax.plot(x_vec[where_min[idcs_plot_against[0]]], y_vec[where_min[idcs_plot_against[1]]], "ro")

        # Ignore unbound variable warning for color bar
        # noinspection PyUnboundLocalVariable
        cbar = fig.colorbar(c, ax=axs)
        cbar.set_label("Ground Truth Metric")
        fig.suptitle(f"Cross Section of Search Space Intersecting Min. Ground Truth={self.min_gt_result:0.2e}"
                     f"\n(red dots show min. value coordinates)")
        return fig
