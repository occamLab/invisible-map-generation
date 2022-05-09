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
from enum import Enum
from typing import Union, Optional, Dict, List, Tuple, Any

import numpy as np
from g2o import SE3Quat
from matplotlib import pyplot as plt
from pydantic import BaseModel, conlist, Field, confloat, conint, validator

from map_processing import ASSUMED_FOCAL_LENGTH, VertexType
from map_processing.transform_utils import NEGATE_Y_AND_Z_AXES, transform_matrix_to_vector, LEN_3_UNIT_VEC


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
    orig_gravity: np.ndarray = Field(default_factory=lambda: np.ones(3))
    orig_odometry: np.ndarray = Field(default_factory=lambda: np.ones(6))
    orig_tag: np.ndarray = Field(default_factory=lambda: np.ones(6))
    orig_tag_sba: np.ndarray = Field(default_factory=lambda: np.ones(2))
    odom_tag_ratio: confloat(ge=0.00001) = 1.0
    normalize: bool = False

    class Config:
        arbitrary_types_allowed = True  # Needed to allow numpy arrays to be used as fields
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}

    # Vector validators
    _check_gravity_is_correct_length_vector = validator("orig_gravity", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 3))
    _check_odometry_is_correct_length_vector = validator("orig_odometry", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 6))
    _check_tag_is_correct_length_vector = validator("orig_tag", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 6))
    _check_tag_sba_is_correct_length_vector = validator("orig_tag_sba", allow_reuse=True)(
        lambda v: _is_vector_of_right_length(v, 2))
    _deserialize_gravity_vector_if_needed = validator("orig_gravity", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)
    _deserialize_odometry_vector_if_needed = validator("orig_odometry", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)
    _deserialize_tag_vector_if_needed = validator("orig_tag", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)
    _deserialize_tag_sba_vector_if_needed = validator("orig_tag_sba", allow_reuse=True, pre=True)(
        _validator_for_numpy_array_deserialization)

    @property
    def odometry(self) -> np.ndarray:
        odom_mag = 1
        if self.normalize:
            odom_mag = np.linalg.norm(self.orig_odometry)
            if odom_mag == 0:  # Avoid divide by zero error
                odom_mag = 1
        return self.orig_odometry * self.odom_tag_ratio / odom_mag

    @property
    def gravity(self) -> np.ndarray:
        grav_mag = 1
        if self.normalize:
            grav_mag = np.linalg.norm(self.orig_gravity)
            if grav_mag == 0:
                grav_mag = 1
        return self.orig_gravity / grav_mag

    @property
    def tag(self) -> np.ndarray:
        tag_mag = 1
        if self.normalize:
            tag_mag = np.linalg.norm(self.orig_tag)
            if tag_mag == 0:
                tag_mag = 1
        return self.orig_tag / tag_mag

    @property
    def tag_sba(self) -> np.ndarray:
        tag_sba_mag = 1
        if self.normalize:
            tag_sba_mag = np.linalg.norm(self.orig_tag_sba)
            if tag_sba_mag == 0:
                tag_sba_mag = 1
        return self.orig_tag_sba / (tag_sba_mag * ASSUMED_FOCAL_LENGTH)

    @property
    def tag_odom_ratio(self):
        return 1 / self.odom_tag_ratio

    @classmethod
    def legacy_from_array(cls, array: Union[np.ndarray, List[float]]) -> "Weights":
        return Weights(**cls.legacy_weight_dict_from_array(array))

    @staticmethod
    def legacy_weight_dict_from_array(array: Union[np.ndarray, List[float]]) -> Dict[str, Union[float, np.ndarray]]:
        """Construct a normalized weight dictionary from a given array of values using the legacy approach.

        TODO: refactor places where this is function is used to not use this approach of constructing weights from a
         single numpy array
        """
        weights = Weights().dict()
        length = array.size if isinstance(array, np.ndarray) else len(array)
        half_len = length // 2
        has_ratio = length % 2 == 1

        if length == 1:  # ratio
            weights['orig_odom_tag_ratio'] = array[0]
        elif length == 2:  # tag/odom pose:rot/tag-sba x:y, ratio
            weights['orig_odometry'] = np.array([array[0]] * 3 + [1] * 3)
            weights['orig_tag'] = np.array([array[0]] * 3 + [1] * 3)
            weights['orig_tag_sba'] = np.array([array[0], 1])
            weights['odom_tag_ratio'] = array[1]
        elif length == 3:  # odom pose:rot, tag pose:rot/tag-sba x:y, ratio
            weights['orig_odometry'] = np.array([array[0]] * 3 + [1] * 3)
            weights['orig_tag'] = np.array([array[1]] * 3 + [1] * 3)
            weights['orig_tag_sba'] = np.array([array[1], 1])
            weights['odom_tag_ratio'] = array[2]
        elif half_len == 2:  # odom pose, odom rot, tag pose/tag-sba x, tag rot/tag-sba y, (ratio)
            weights['orig_odometry'] = np.array([array[0]] * 3 + [array[1]] * 3)
            weights['orig_tag'] = np.array([array[2]] * 3 + [array[3]] * 3)
            weights['orig_tag_sba'] = np.array(array[2:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif half_len == 3:  # odom x y z qx qy, tag-sba x, (ratio)
            weights['orig_odometry'] = np.array(array[:5])
            weights['orig_tag_sba'] = np.array([array[5]])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 4:  # odom, tag-sba, (ratio)
            weights['orig_odometry'] = np.array(array[:6])
            weights['orig_tag_sba'] = np.array(array[6:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 5:  # odom x y z qx qy, tag x y z qx qy, (ratio)
            weights['orig_odometry'] = np.array(array[:5])
            weights['orig_tag'] = np.array(array[5:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 6:  # odom, tag, (ratio)
            weights['orig_odometry'] = np.array(array[:6])
            weights['orig_tag'] = np.array(array[6:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        else:
            raise Exception(f'Weight length of {length} is not supported')

        weights["normalize"] = True
        w = Weights(**weights)
        return w.dict()

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


class GenerateParams(BaseModel):
    # noinspection PyUnresolvedReferences
    """
        Attributes:
            dataset_name: String provided as the data set name to the cache manager when the generated data set is
             cached.
            map_id: String provided as the map_id field in the UGDataSet object when exported.
            parameterized_path_args: Dictionary to pass as the second positional argument to the `path_from` argument if
             it is a callable (if the `path_from` argument is not a callable, then this argument is ignored).
            t_max: For a parameterized path, this is the max parameter value to use when evaluating the path.
            n_poses: Number of poses to sample a parameterized path at; if a recorded path is provided, then this
             argument is ignored.
            dist_threshold: Maximum distance from which a tag can be considered observable.
            aoa_threshold: Maximum angle of attack (in radians) from which a tag can be considered observable. The angle
             of attack is calculated as the angle between the z-axis of the tag pose and the vector from the tag to the
             phone.
            tag_size: Height/width dimension of the (square) tags in meters.
            obs_noise_var: Variance parameter for the observation model. Specifies the variance for the distribution
             from which pixel noise is sampled and added to the simulated tag corner pixel observations. Note that the
             simulated tag observation poses are re-derived from these noise pixel observations.
            odometry_noise_var: Dictionary mapping a dimension to which noise is applied to the variance of the Gaussian
             noise in that direction.

        Properties:
            delta_t: For a parameterized path, this gives the time delta used between each of the points. If the path is
             a recorded path, then this value is set to 0 arbitrarily.
        """

    class OdomNoiseDims(str, Enum):
        X = "x"
        Y = "y"
        Z = "z"
        RVERT = "rvert"

        @staticmethod
        def ordering() -> List:
            return [GenerateParams.OdomNoiseDims.X, GenerateParams.OdomNoiseDims.Y, GenerateParams.OdomNoiseDims.Z,
                    GenerateParams.OdomNoiseDims.RVERT]

    dataset_name: str
    map_id: Optional[str] = None
    dist_threshold: confloat(ge=0) = 3.7
    aoa_threshold: confloat(ge=0, le=np.pi) = np.pi / 4
    tag_size: confloat(gt=0) = 0.7
    odometry_noise_var: Dict[OdomNoiseDims, float] = Field(default_factory=lambda: {
            GenerateParams.OdomNoiseDims.X: 0,
            GenerateParams.OdomNoiseDims.Y: 0,
            GenerateParams.OdomNoiseDims.Z: 0,
            GenerateParams.OdomNoiseDims.RVERT: 0,
        })
    obs_noise_var: confloat(ge=0) = 0.0
    t_max: Optional[confloat(gt=0)] = None
    n_poses: Optional[conint(ge=2)] = None
    parameterized_path_args: Optional[Dict[str, Union[float, Tuple[float, float]]]] = None

    # Ignore warning about first argument not being self (decorating as a @classmethod appears to prevent validation for
    # some reason...)
    # noinspection PyMethodParameters
    @validator("parameterized_path_args")
    def validate_interdependent_null_values(cls, v, values):
        v_is_none = v is None
        t_max_is_none = values["t_max"] is None
        n_poses_is_none = values["n_poses"] is None

        if not ((v_is_none and t_max_is_none and n_poses_is_none) or
                (not v_is_none and not t_max_is_none and not n_poses_is_none)):
            raise ValueError("tag_poses_for_parameterized, t_max, n_poses, and parameterized_path_args members must "
                             "both be None or not None.")
        return v

    @property
    def delta_t(self):
        """If t_max is not None, then a delta-time value is computed from t_max and the number of specified poses

        """
        if self.t_max is not None:
            return self.t_max / (self.n_poses - 1)
        else:
            return 0

    class GenerateParamsEnum(str, Enum):
        ODOMETRY_NOISE_VAR_X = "odometry_noise_var_x"
        ODOMETRY_NOISE_VAR_Y = "odometry_noise_var_y"
        ODOMETRY_NOISE_VAR_Z = "odometry_noise_var_z"
        ODOMETRY_NOISE_VAR_RVERT = "odometry_noise_var_rvert"
        OBS_NOISE_VAR = "obs_noise_var"

    # noinspection DuplicatedCode
    @classmethod
    def generate_params_generator(
            cls, param_multiplicands: Dict[GenerateParamsEnum, np.ndarray], param_order: List[GenerateParamsEnum],
            base_generate_params: "GenerateParams") -> Tuple[List[Tuple[Any, ...]], List["GenerateParams"]]:
        """Generator yielding instances of this class according to the cartesian product of the provided parameters.

        Args:
            param_multiplicands: Dictionary mapping parameters to arrays of values whose cartesian product is
             taken.
            param_order: Ordering of the keys in param_multiplicands.
            base_generate_params: Supplies every parameter not prescribed by param_multiplicands.

        Returns:
            A list of each tuple of parameters computed from the cartesian product (the length of which is equivalent to
            the length of param_order) and a list of the generated objects.

        Raises:
            ValueError: If the keys of param_multiplicands elements of param_order are not the same.
        """
        included_params = set(param_order)
        if set(param_multiplicands.keys()) != included_params:
            raise ValueError("The sets of param_multiplicands keys and param_order items must be equal")

        product_args = []
        sweep_param_to_product_idx: Dict[GenerateParams.GenerateParamsEnum, int] = {}
        for i, key in enumerate(param_order):
            product_args.append(param_multiplicands[key])
            sweep_param_to_product_idx[key] = i

        products: List[Tuple[Any, ...]] = []
        generate_params: List[GenerateParams] = []
        for this_product in itertools.product(*product_args, repeat=1):
            products.append(this_product)

            # For each of the x, y, z, and rvert elements of the odometry noise, apply the value stored in the
            # included_params dictionary if it is a key; if not, then default to the value stored in
            # base_generate_params.
            odometry_noise = {}
            if GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X in included_params:
                odometry_noise[GenerateParams.OdomNoiseDims.X] = this_product[
                    sweep_param_to_product_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X]]
            else:
                odometry_noise[GenerateParams.OdomNoiseDims.X] = \
                    base_generate_params.odometry_noise_var[GenerateParams.OdomNoiseDims.X]

            if GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_Y in included_params:
                odometry_noise[GenerateParams.OdomNoiseDims.Y] = \
                    this_product[sweep_param_to_product_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_Y]]
            else:
                odometry_noise[GenerateParams.OdomNoiseDims.Y] = \
                    base_generate_params.odometry_noise_var[GenerateParams.OdomNoiseDims.Y]

            if GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_Z in included_params:
                odometry_noise[GenerateParams.OdomNoiseDims.Z] = this_product[
                    sweep_param_to_product_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_Z]]
            else:
                odometry_noise[GenerateParams.OdomNoiseDims.Z] = \
                    base_generate_params.odometry_noise_var[GenerateParams.OdomNoiseDims.Z]

            if GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_RVERT in included_params:
                odometry_noise[GenerateParams.OdomNoiseDims.RVERT] = this_product[sweep_param_to_product_idx[
                        GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_RVERT]]
            else:
                odometry_noise[GenerateParams.OdomNoiseDims.RVERT] = \
                    base_generate_params.odometry_noise_var[GenerateParams.OdomNoiseDims.RVERT]

            generate_params.append(
                GenerateParams(
                    dataset_name=base_generate_params.dataset_name,
                    dist_threshold=base_generate_params.dist_threshold,
                    aoa_threshold=base_generate_params.aoa_threshold,
                    tag_size=base_generate_params.tag_size,
                    odometry_noise=odometry_noise,
                    obs_noise_var=this_product[sweep_param_to_product_idx[
                        GenerateParams.GenerateParamsEnum.OBS_NOISE_VAR]] if
                    GenerateParams.GenerateParamsEnum.OBS_NOISE_VAR in included_params else
                    base_generate_params.obs_noise_var,
                    t_max=base_generate_params.t_max,
                    n_poses=base_generate_params.n_poses,
                    parameterized_path_args=base_generate_params.parameterized_path_args
                )
            )
        return products, generate_params

    def __hash__(self):
        # TODO: there are more efficient ways to do this, but this works for now
        return self.json().__hash__()


class UGDataSet(BaseModel):
    """Represents an unprocessed graph dataset.

    Notes:
        All attributes except `generated_from` are necessary for deserializing data from the datasets generated by the
         client app. The `generated_from` attribute is only used when the data set generated is synthetic.
    """
    location_data: List[UGLocationDatum] = []
    map_id: str
    plane_data: List = []
    pose_data: List[UGPoseDatum]
    tag_data: List[List[UGTagDatum]] = []
    generated_from: Optional[GenerateParams] = None

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

    class OconfigEnum(str, Enum):
        ODOM_TAG_RATIO = "odom_tag_ratio"
        LIN_VEL_VAR = "lin_vel_var"
        ANG_VEL_VAR = "ang_vel_var"
        GRAV_MAG = "grav_mag"

    # noinspection DuplicatedCode
    @classmethod
    def oconfig_generator(cls, param_multiplicands: Dict[OconfigEnum, np.ndarray], param_order: List[OconfigEnum],
                          base_oconfig: "OConfig") -> Tuple[List[Tuple[Any, ...]], List["OConfig"]]:
        """Generator yielding instances of this class according to the cartesian product of the provided parameters.

        Args:
            param_multiplicands: Dictionary mapping parameters to arrays of values whose cartesian product is
             taken.
            param_order: Ordering of the keys in param_multiplicands.
            base_oconfig: Supplies every parameter not prescribed by param_multiplicands.

        Returns:
            A list of each tuple of parameters computed from the cartesian product (the length of which is equivalent to
            the length of param_order) and a list of the generated objects.

        Raises:
            ValueError: If the keys of param_multiplicands elements of param_order are not the same.
        """
        included_params = set(param_order)
        if set(param_multiplicands.keys()) != included_params:
            raise ValueError("The sets of param_multiplicands keys and param_order items must be equal")

        product_args = []
        sweep_param_to_product_idx: Dict[OConfig.OconfigEnum, int] = {}
        for i, key in enumerate(param_order):
            product_args.append(param_multiplicands[key])
            sweep_param_to_product_idx[key] = i

        products: List[Tuple[Any, ...]] = []
        oconfigs: List[OConfig] = []
        for this_product in itertools.product(*product_args, repeat=1):
            products.append(this_product)
            oconfigs.append(
                OConfig(
                    is_sba=base_oconfig.is_sba,
                    obs_chi2_filter=base_oconfig.obs_chi2_filter,
                    compute_inf_params=OComputeInfParams(
                        lin_vel_var=(this_product[sweep_param_to_product_idx[
                            OConfig.OconfigEnum.LIN_VEL_VAR]] if OConfig.OconfigEnum.LIN_VEL_VAR in
                            included_params else base_oconfig.compute_inf_params.lin_vel_var) * np.ones(3),
                        ang_vel_var=this_product[sweep_param_to_product_idx[
                            OConfig.OconfigEnum.ANG_VEL_VAR]] if OConfig.OconfigEnum.ANG_VEL_VAR in
                        included_params else base_oconfig.compute_inf_params.ang_vel_var,
                    ),
                    scale_by_edge_amount=base_oconfig.scale_by_edge_amount,
                    weights=Weights(
                        orig_gravity=(this_product[sweep_param_to_product_idx[
                            OConfig.OconfigEnum.GRAV_MAG]] if OConfig.OconfigEnum.GRAV_MAG in
                            included_params else base_oconfig.weights.orig_gravity) * LEN_3_UNIT_VEC,
                        odom_tag_ratio=this_product[sweep_param_to_product_idx[
                            OConfig.OconfigEnum.ODOM_TAG_RATIO]] if OConfig.OconfigEnum.ODOM_TAG_RATIO
                        in included_params else base_oconfig.weights.odom_tag_ratio,
                    ),
                    graph_plot_title=base_oconfig.graph_plot_title,
                    chi2_plot_title=base_oconfig.chi2_plot_title,
                )
            )
        return products, oconfigs

    def __hash__(self):
        # TODO: there are more efficient ways to do this, but this works for now
        return self.json().__hash__()


class OG2oOptimizer(BaseModel):
    """Record of map state.

    TODO: The names of (a subset of) these fields were specifically chosen according to previously chosen
     variable names, but they should be refactored to better reflect what they actually represent.

    Class Attributes:
        locations: (n, 9) array containing n odometry poses (x, y, z, qx, qy, qz, qw) well as the vertex
         index and uid in the 7th and 8th positions, respectively.
        tags: (m, 8) array containing m tag poses (x, y, z, qx, qy, qz, qw) as well as the tag id in the
         7th position.
        tagpoints: (m, 3) array containing the m tag corners' global xyz coordinates.
        waypoints_arr: (w, 8) array containing w waypoint poses (x, y, z, qx, qy, qz, qw).
        locationsAdjChi2: Optionally associated with each odometry node is a chi2 calculated from the
         `map_odom_to_adj_chi2` method of the `Graph` class, which is stored in this vector.
        waypoints_metadata: Length-w list of waypoint metadata.
        visibleTagsCount: Optionally associated with each odometry node is an integer indicating the number
         of tags visible at that given pose.
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
    """Container to store the chi2 values
    """

    chi2_all_before: confloat(ge=0)
    chi2_gravity_before: confloat(ge=0)
    chi2_all_after: confloat(ge=0)
    chi2_gravity_after: confloat(ge=0)


class OResult(BaseModel):
    # noinspection PyUnresolvedReferences
    """
    Attributes:
        oconfig: The optimization configuration
        map_pre: The state of the map pre-optimization
        map_opt: The state of the optimized map
        chi2s: The chi2 metrics before and after optimization
        gt_metric_pre: Ground truth metric of the pre-optimized map
        gt_metric_opt: Ground truth metric of the optimized map
    """
    oconfig: OConfig
    map_pre: OG2oOptimizer
    map_opt: OG2oOptimizer
    chi2s: OResultChi2Values
    gt_metric_pre: Optional[float] = None
    gt_metric_opt: Optional[float] = None

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
    """Used to store the results of a parameter sweep.

    Notes:
        For generated data sets, it may be important to know the parameters used to generate them. Though this can be
        achieved by looking up the path of the unprocessed graph json from the map_name attribute, deserializing it, and
        accessing the GenerateParams member, the `generated_params` attribute exists here to be a convenient place to
        store a copy that data.
    """

    gt_results_arr_shape: List[int]
    sweep_config: Dict[str, List[float]]
    gt_results_list: List[float]
    sweep_config_keys_order: List[str]
    base_oconfig: OConfig
    map_name: str
    generated_params: Optional[GenerateParams] = None

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
            remove_dims = sorted(list(set(range(len(self.sweep_config_keys_order))).difference(
                set(idcs_plot_against))))
            zz = self.gt_results_arr
            for ith_dim, remove_dim in enumerate(remove_dims):
                zz = zz.take(indices=where_min[remove_dim], axis=remove_dim - ith_dim)

            ax.set_xlabel(self.sweep_config_keys_order[idcs_plot_against[0]])
            ax.set_ylabel(self.sweep_config_keys_order[idcs_plot_against[1]])
            c = ax.pcolormesh(xx, yy, zz.T, shading="auto")

            # Annotate the heatmap with a red dot showing the coordinates that produce the minimum value
            ax.plot(x_vec[where_min[idcs_plot_against[0]]], y_vec[where_min[idcs_plot_against[1]]], "ro")

        # Ignore unbound variable warning for color bar
        # noinspection PyUnboundLocalVariable
        cbar = fig.colorbar(c, ax=axs)
        cbar.set_label("Ground Truth Metric")
        fig.suptitle(f"Cross Section of Search Space Intersecting Min. Ground Truth={self.min_gt_result:0.2e}"
                     f"\n(red dots show min. value coordinates)")
        return fig


class OMultiSweepResult(BaseModel):
    uid_to_generate_params: Dict[int, GenerateParams]
    sweep_results_by_generate_params_uid: Dict[int, List[OSweepResults]]

    # Need this class with the `json_encoders` field to be present so that the contained numpy arrays can be
    # serializable, even though the models this model is composed of are already serializable on their own.
    class Config:
        json_encoders = {np.ndarray: lambda arr: np.array2string(arr)}
