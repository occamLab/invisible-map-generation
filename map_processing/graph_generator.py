"""
Contains the GraphGenerator class used for generating artificial datasets for optimization. See the generate_datasets.py
script for a CLI interface using this class.
"""

import datetime
import json
from copy import deepcopy
from enum import Enum
from typing import Callable, Tuple, Optional, List, Dict, Union

import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.data_models import (
    UGDataSet,
    UGTagDatum,
    UGPoseDatum,
    GTDataSet,
    GenerateParams,
)
from map_processing.graph_opt_plot_utils import draw_frames
from map_processing.transform_utils import (
    norm_array_cols,
    NEGATE_Y_AND_Z_AXES,
    angle_axis_to_quat,
)


class GraphGenerator:
    """Generates synthetic datasets simulating tag observations.

    Attributes:
        _path_from: Defines a parameterized path. Takes as input an N-length vector of parameters to evaluate the curve
         at, and returns a 3xN array where the rows from top to bottom are the x, y, and z coordinates respectively.
        _path_type: The type of the `_path` attribute as specified by the `GraphGenerator.PathType` enumeration.
        _odometry_t_vec: The vector of pose time stamps, either generated from _t_max and _n_poses in the case of a
         parameterized path or copied from the 'timestamp' fields of the data set otherwise.
        _tag_poses: A Mx4x4 array of M tag poses defined in the global reference frame using homogenous transform
         matrices.
        _odometry_poses: Nx4x4 array of N homogenous transforms representing the poses sampled along the path.
        _observation_poses: A list of length N where N is the number of poses sampled along the path. Each element of
         the list is a dictionary mapping tag IDs to their corresponding transforms that give their position in the
         reference frame of that pose. A key-value pair is only present in the dictionary if it is visible from that
         pose.
        _observation_pixels: Nx2x4 array containing the pixel coordinates corresponding to the pose observations
         recorded in _observation_poses.
        _tag_corners_in_tag: 4x4 array where the first 3 elements of each row give the x, y, and z coordinates of the
         tag corners in the tag's reference frame. The 4th element is always a 1.

    """

    class PathType(Enum):
        RECORDED = 0
        PARAMETERIZED = 1

    CAMERA_INTRINSICS_VEC = [
        1458.0604248046875,  # fx (camera focal length in the x-axis)
        1458.0604248046875,  # fy (camera focal length in the y-axis)
        924.9375,  # cx (principal point offset in the x-axis)
        725.906494140625,  # cy (principal point offset along the y-axis)
    ]

    CAMERA_INTRINSICS = np.array(
        [
            [CAMERA_INTRINSICS_VEC[0], 0, CAMERA_INTRINSICS_VEC[2]],
            [0, CAMERA_INTRINSICS_VEC[1], CAMERA_INTRINSICS_VEC[3]],
            [0, 0, 1],
        ]
    )
    _double_camera_intrinsics = 2 * CAMERA_INTRINSICS

    TAG_CORNERS_SIZE_2 = np.transpose(
        np.array(
            [
                [-1, -1, 0, 1],
                [1, -1, 0, 1],
                [1, 1, 0, 1],
                [-1, 1, 0, 1],
            ]
        ).astype(float)
    )

    TAG_CORNERS_SIZE_2_FOR_CV_PNP = np.array(
        [
            [-1, 1, 0],
            [1, 1, 0],
            [1, -1, 0],
            [-1, -1, 0],
        ]
    ).astype(float)

    NEGATE_ONLY_YZ_BASIS_VECTORS_ELEMENTWISE = np.array(
        [
            [1, -1, -1],
            [1, -1, -1],
            [1, -1, -1],
        ]
    )

    PHONE_IN_FRENET = np.array(
        [[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]]
    )

    PATH_LINEAR_DELTA_T = 0.0001  # Time span over which a parameterized path can be assumed to be approximately linear

    def __init__(
        self,
        path_from: Union[
            Callable[[np.ndarray, Dict[str, float]], np.ndarray], UGDataSet
        ],
        gen_params: GenerateParams,
        tag_poses_for_parameterized: Optional[Dict[int, np.ndarray]] = None,
    ):
        """Initializes a GraphGenerator instance and automatically invokes the `generate` method.

        Args:
            path_from: If a callable, then it defines a parameterized path where the first positional argument of the
             callable is to be an N-length vector of parameters to evaluate the curve at, and returns a 3xN array where
             the rows from top to bottom are the x, y, and z coordinates respectively; the second positional argument is
             a dictionary specifying path parameters (the contents of which is function-specific). If a `UGDataSet`
             instance, then it defines a path and set of tags according to the data in the data set.
            gen_params: Container for parameters of the graph generation.
            tag_poses_for_parameterized: For simulating a data set that is not derived from a recorded data set: A
             dictionary mapping tag IDs to their poses in the global reference frame using homogenous transform
             matrices; if a recorded path is provided, then this argument overrides the data set's tag poses if it is
             not none.

        Raises:
            ValueError - If `path_from` is a `UGDataSet` instance and `t_max` is negative.
            ValueError - If `path_from` is a `UGDataSet` instance and `n_poses` is <2.
            ValueError - If `path_from` is a callable and `gen_params` has a null value for any of its t_max, n_poses,
             or parameterized_path_args attributes.
        """
        self._path_from: Union[
            Callable[[np.ndarray, Dict[str, float]], np.ndarray], UGDataSet
        ] = path_from
        self._path_type = (
            GraphGenerator.PathType.PARAMETERIZED
            if isinstance(self._path_from, Callable)
            else GraphGenerator.PathType.RECORDED
        )

        self._gen_params = gen_params
        self._tag_corners_in_tag = np.array(GraphGenerator.TAG_CORNERS_SIZE_2)
        self._tag_corners_in_tag[:3, :] *= self._gen_params.tag_size / 2
        self._tag_corners_for_pnp = (
            GraphGenerator.TAG_CORNERS_SIZE_2_FOR_CV_PNP * self._gen_params.tag_size / 2
        )
        self._n_poses = (
            self._gen_params.n_poses
            if isinstance(self._path_from, Callable)
            else len(self._path_from.pose_data)
        )

        self._observation_poses: Optional[List[Optional[Dict[int, np.ndarray]]]] = None
        self._obs_from_poses: Optional[np.ndarray] = None
        self._observation_pixels: Optional[List[Optional[Dict[int, np.ndarray]]]] = None

        self._tag_poses: Dict[int, np.ndarray]

        if self._path_type is GraphGenerator.PathType.PARAMETERIZED:
            if (
                self._gen_params.t_max is None
                or self._gen_params.n_poses is None
                or self._gen_params.parameterized_path_args is None
            ):
                raise ValueError(
                    "Provided GenerateParams instance cannot have a null value for any of its t_max, "
                    "n_poses, or parameterized_path_args attributes."
                )
            self._tag_poses = (
                {}
                if tag_poses_for_parameterized is None
                else tag_poses_for_parameterized
            )
        else:
            self._tag_poses = self._path_from.approx_tag_in_global_by_id
        self._tag_poses_orig = deepcopy(self._tag_poses)

        self._orig_tag_poses_arr = np.zeros((len(self._tag_poses), 4, 4))
        for i, pose in enumerate(self._tag_poses.values()):
            self._orig_tag_poses_arr[i, :, :] = pose

        self._odometry_poses: Optional[np.ndarray] = None
        if self._path_type is GraphGenerator.PathType.PARAMETERIZED:
            self._odometry_t_vec = np.arange(
                0,
                self._gen_params.t_max,
                self._gen_params.t_max / self._gen_params.n_poses,
            )
        else:
            self._odometry_t_vec = self._path_from.timestamps
            self._delta_t = 0

        self.generate()

    # -- Public methods --

    def export(self) -> Tuple[UGDataSet, GTDataSet]:
        """
        Returns:
            A UGDataSet object that, when serialized, will contain the unprocessed graph json encoding of this
             artificial dataset generation.
        """
        # Construct data for the UGDataSet initialization
        pose_data: List[UGPoseDatum] = []
        tag_data: List[List[UGTagDatum]] = []
        tag_data_idx = -1
        for pose_idx in range(self._odometry_poses.shape[0]):
            pose_data.append(
                UGPoseDatum(
                    pose=list(self._odometry_poses[pose_idx, :, :].flatten(order="F")),
                    timestamp=self._odometry_t_vec[pose_idx],
                    # Intentionally skipping planes
                    id=pose_idx,
                )
            )

            looped = False
            for tag_id in self._observation_poses[pose_idx].keys():
                if not looped:
                    tag_data.append(list())
                    tag_data_idx += 1
                looped = True
                tag_data[tag_data_idx].append(
                    UGTagDatum(
                        tag_corners_pixel_coordinates=list(
                            self._observation_pixels[pose_idx][tag_id].flatten(
                                order="F"
                            )
                        ),
                        tag_id=tag_id,
                        pose_id=pose_idx,
                        camera_intrinsics=list(
                            np.array(GraphGenerator.CAMERA_INTRINSICS_VEC)
                        ),
                        # Intentionally skipping position and orientation variance
                        timestamp=self._odometry_t_vec[pose_idx],
                        tag_pose=list(
                            self._observation_poses[pose_idx][tag_id].flatten(order="C")
                        ),
                        # Intentionally skipping joint covariance
                    )
                )

        return UGDataSet(
            # Intentionally skipping location data
            map_id="generated_" + datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
            if self._gen_params.map_id is None
            else self._gen_params.map_id,
            # Intentionally skipping plane data
            pose_data=pose_data,
            tag_data=tag_data,
            generated_from=self._gen_params,
        ), GTDataSet.gt_data_set_from_dict_of_arrays(self._tag_poses_orig)

    def export_to_map_processing_cache(
        self, verbose=False, file_name_suffix: str = ""
    ) -> MapInfo:
        """Serializes the graph into a json file and saves it to the target destination as set by the TODO"""
        map_obj, gt_obj = self.export()
        map_dct = map_obj.dict()
        map_str = json.dumps(map_dct, indent=2)

        map_id = map_dct["map_id"]
        if verbose:
            print(
                f"Generated new data set '{map_id}' containing {map_obj.pose_data_len} poses and {map_obj.num_tags} "
                f"tags observed a total of {map_obj.num_observations} "
                f"{'times' if map_obj.num_observations > 1 else 'time'}."
            )

        mi = MapInfo(
            map_name=map_id + file_name_suffix,
            map_json_name=map_id + file_name_suffix,
            map_dct=map_dct,
        )
        CacheManagerSingleton.cache_map(
            parent_folder="generated", map_info=mi, json_string=map_str
        )
        CacheManagerSingleton.cache_ground_truth_data(
            gt_obj,
            dataset_name=self._gen_params.dataset_name,
            corresponding_map_names=[map_id + file_name_suffix],
        )
        return mi

    def visualize(self, plus_minus_lim=5) -> None:
        """Visualizes the generated graph by plotting the path, the poses on the path, the tags, and the observations of
         those tags.

        TODO: add plot legend.

        Args:
            plus_minus_lim: Value for the x-, y-, and z-lim3d parameters of the matplotlib 3d axes.
        """
        path_samples = self._obs_from_poses[:, :3, 3].transpose()

        f: plt.Figure = plt.figure()
        ax: Axes3D = f.add_subplot(projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axes.set_xlim3d(left=-plus_minus_lim, right=plus_minus_lim)
        ax.axes.set_ylim3d(bottom=-plus_minus_lim, top=plus_minus_lim)
        ax.axes.set_zlim3d(bottom=-plus_minus_lim, top=plus_minus_lim)

        plt.plot(
            self._odometry_poses[:, 0, 3],
            self._odometry_poses[:, 1, 3],
            self._odometry_poses[:, 2, 3],
        )
        plt.plot(path_samples[0, :], path_samples[1, :], path_samples[2, :])

        # Get observation vectors in the global frame and plot them
        for i, dct in enumerate(self._observation_poses):
            pose = self._obs_from_poses[i, :, :]
            line_start = pose[:3, 3]
            for obs in dct.values():
                # Apply NEGATE_Y_AND_Z_AXES multiplication here because to mirror what happens when poses are ingested
                # in the `Graph.as_graph` method (via the `UGDataSet.tag_edge_measurements_matrix` property). It undoes
                # the NEGATE_Y_AND_Z_AXES left-multiplication in the `observe_tag_by_pixels` method (it is its own
                # inverse).
                obs_in_global = np.matmul(pose, np.matmul(NEGATE_Y_AND_Z_AXES, obs))
                draw_frames(obs_in_global, plt_axes=ax)
                line_end = obs_in_global[:3, 3]
                ax.plot(
                    xs=[line_start[0], line_end[0]],
                    ys=[line_start[1], line_end[1]],
                    zs=[line_start[2], line_end[2]],
                    color="c",
                )
        plt.show()

    def generate(self) -> None:
        """Populate the _odometry_poses attribute with the poses sampled at the given parameter values, then the
        _observation_poses attribute is populated with the tag observations at each of the poses.

        Raises:
            ValueError - If `_path_from` is not a Callable or a UGDataSet.
        """
        if isinstance(self._path_from, Callable):
            positions = self._path_from(
                self._odometry_t_vec, self._gen_params.parameterized_path_args
            )  # 3xN array
            # Nx3x3 array
            frenet_frames = GraphGenerator.frenet_frames(
                self._odometry_t_vec,
                self._path_from,
                self._gen_params.parameterized_path_args,
            )
            true_poses = np.zeros((len(self._odometry_t_vec), 4, 4))  # Nx4x4
            true_poses[:, 3, 3] = 1
            true_poses[:, :3, 3] = positions.transpose()
            true_poses[:, :3, :3] = np.matmul(
                frenet_frames, GraphGenerator.PHONE_IN_FRENET[:3, :3]
            )
            self._odometry_poses = self._apply_noise(true_poses)
            self._obs_from_poses = true_poses
        elif isinstance(self._path_from, UGDataSet):
            self._odometry_poses = self._apply_noise(self._path_from.pose_matrices)
            self._obs_from_poses = self._path_from.pose_matrices
        else:
            raise ValueError("'_path_from' attribute is not of a known type")

        self._observation_poses = []
        self._observation_pixels = []
        for i in range(self._n_poses):
            self._observation_poses.append(dict())
            self._observation_pixels.append(dict())
            for tag_id in self._tag_poses:
                tag_obs, tag_pixels = self._get_tag_observation(
                    self._obs_from_poses[i, :, :], self._tag_poses[tag_id]
                )
                if tag_obs.shape[0] == 0:  # True if no tags were visible
                    continue

                # If >=1 tags were visible, enter that information into the observation dictionaries
                self._observation_poses[i][tag_id] = tag_obs
                self._observation_pixels[i][tag_id] = tag_pixels

    # -- Private methods --

    def _apply_noise(self, true_poses: np.ndarray) -> np.ndarray:
        if true_poses.shape[0] <= 1:
            raise Exception("To apply noise, there must be >=2 poses")

        # The t^th sub-array of the pose_to_pose array contains the transform from the pose at time t-1 to t.
        pose_to_pose = np.zeros((true_poses.shape[0] - 1, 4, 4))
        for i in range(0, true_poses.shape[0] - 1):
            pose_to_pose[i, :, :] = np.matmul(
                np.linalg.inv(true_poses[i, :, :]), true_poses[i + 1, :, :]
            )

        odom_noise_x = self._gen_params.odometry_noise_var[
            GenerateParams.OdomNoiseDims.X
        ]
        odom_noise_y = self._gen_params.odometry_noise_var[
            GenerateParams.OdomNoiseDims.Y
        ]
        odom_noise_z = self._gen_params.odometry_noise_var[
            GenerateParams.OdomNoiseDims.Z
        ]
        noisy_poses = np.zeros(true_poses.shape)
        noisy_poses[0, :, :] = true_poses[0, :, :]
        for i in range(0, true_poses.shape[0] - 1):
            this_delta_t = self._odometry_t_vec[i + 1] - self._odometry_t_vec[i]
            noise_as_transform = np.zeros((4, 4))
            noise_as_transform[3, 3] = 1
            noise_as_transform[:3, 3] = np.array(
                [
                    np.random.normal(0, this_delta_t * np.sqrt(odom_noise_x)),
                    np.random.normal(0, this_delta_t * np.sqrt(odom_noise_y)),
                    np.random.normal(0, this_delta_t * np.sqrt(odom_noise_z)),
                ]
            )
            theta = np.random.normal(
                0,
                this_delta_t
                * np.sqrt(
                    self._gen_params.odometry_noise_var[
                        GenerateParams.OdomNoiseDims.RVERT
                    ]
                ),
            )

            # Assume rotational noise can be modeled as a normally-distributed rotational error about the gravity axis
            gravity_axis_in_phone_frame = noisy_poses[
                i, 1, :3
            ]  # Select y basis vector of inverted pose
            noise_as_transform[:3, :3] = angle_axis_to_quat(
                theta, gravity_axis_in_phone_frame
            ).R
            noisy_transform = np.matmul(pose_to_pose[i, :, :], noise_as_transform)
            noisy_poses[i + 1, :, :] = np.matmul(noisy_poses[i, :, :], noisy_transform)
        return noisy_poses

    def _get_tag_observation(
        self, obs_from: np.ndarray, tag: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            obs_from: Pose (4x4 homogenous transform) from which the observation is made. Assumed to be from the same
             reference frame as the tag's pose.
            tag: Pose (4x4 homogenous transform) of the tag. Assumed to be from the same reference frame as the obs_from
             pose.

        Returns:
            If the tag is visible: A tuple containing (1) a 4x4 homogenous transform giving the tag pose in the
            reference frame of the phone and (2) a 2x4 array where each column gives the x-y pixel coordinates of the
            pixel projection for a given corner (corner order determined by the TAG_CORNERS_SIZE_2 class attribute).
            Otherwise, two empty arrays are returned.
        """
        tag_in_phone = np.matmul(np.linalg.inv(obs_from), tag)
        if tag_in_phone[2, 3] > 0:  # True if the tag is behind the camera
            return np.array([]), np.array([])

        vector_phone_to_tag = tag_in_phone[:3, 3]
        dist_to_tag = np.linalg.norm(vector_phone_to_tag)
        if dist_to_tag > self._gen_params.dist_threshold:
            return np.array([]), np.array([])

        # Calculate aoa in the range [0, pi] rad. The angle of attack is found by computing the arccos of
        # normalized vector_phone_to_tag vector and the optical axis of the phone's camera (which is the negative z-axis
        # basis vector of the rotation matrix in the tag_in_phone transform)
        dot_for_aoa = -np.dot(vector_phone_to_tag, tag_in_phone[:3, 2])
        if dist_to_tag == 0:
            # Avoid divide by zero by setting angle of attack to pi
            aoa = np.pi
        else:
            aoa = (
                np.arccos(dot_for_aoa / dist_to_tag)
                if dot_for_aoa > 0
                else (np.pi - np.arccos(dot_for_aoa / dist_to_tag))
            )
        if aoa > self._gen_params.aoa_threshold:
            return np.array([]), np.array([])

        tag_pixels = np.zeros((2, 4))
        if not self._observe_tag_by_pixels(tag_in_phone, tag_pixels):
            return np.array([]), np.array([])

        return tag_in_phone, tag_pixels

    def _observe_tag_by_pixels(self, tag_in_phone, pixel_vals: np.ndarray) -> bool:
        """Project points in 3D space into phone space.

        Args:
            tag_in_phone: Transform of the tag observation in the phone's reference frame. This array is modified such
             that it contains the new tag transform after the coordinate projection, noise addition, and pose
             re-computation
            pixel_vals: 2x4 array populated with coordinates of the tag observation after noise is applied.
        Returns:
            True if all points are visible according to the camera intrinsics.
        """
        # Compute the pixel_vals values first. Note the slightly different computation between this and the pixels
        # computed for solvePnP
        pixel_vals_tmp = np.matmul(
            GraphGenerator.CAMERA_INTRINSICS,
            np.matmul(
                np.matmul(NEGATE_Y_AND_Z_AXES, tag_in_phone), self._tag_corners_in_tag
            )[:3, :],
        )
        for col_idx in range(pixel_vals_tmp.shape[1]):
            if pixel_vals_tmp[2, col_idx] == 0:
                # Avoid divide-by-zero by setting pixels to -1 (corresponds to non-visible observation)
                pixel_vals_tmp[:, col_idx] = -1
            pixel_vals_tmp[:, col_idx] = (
                pixel_vals_tmp[:, col_idx] / pixel_vals_tmp[2, col_idx]
            )

        # Add pixel noise and record in provided array
        pixel_vals_tmp[:2, :] += np.random.randn(2, 4) * self._gen_params.obs_noise_var
        pixel_vals[:, :] = pixel_vals_tmp[:2, :]

        # Check if all pixel coordinates are within the sensor's bounds
        if (
            np.any(pixel_vals_tmp[0:2, :] < 0)
            or np.any(
                pixel_vals_tmp[0, :] > GraphGenerator._double_camera_intrinsics[0, 2]
            )
            or np.any(
                pixel_vals_tmp[1, :] > GraphGenerator._double_camera_intrinsics[1, 2]
            )
        ):
            return False

        # Recompute the pose using solvePnP
        t_vec = np.expand_dims(tag_in_phone[:3, 3], 1)
        r_vec = cv2.Rodrigues(tag_in_phone[:3, :3])[0]
        success, r_vec, t_vec = cv2.solvePnP(
            objectPoints=self._tag_corners_for_pnp,
            imagePoints=np.transpose(pixel_vals_tmp[:2, :]),
            cameraMatrix=self.CAMERA_INTRINSICS,
            rvec=r_vec,
            tvec=t_vec,
            distCoeffs=None,
            flags=cv2.SOLVEPNP_IPPE_SQUARE,
            useExtrinsicGuess=True,
        )
        if not success:
            return False  # TODO: figure out why this doesn't work every time that it can reasonably be expected to

        rot_mat, _ = cv2.Rodrigues(r_vec)
        new_transform = np.zeros((4, 4))
        # This operation reverses the effect that the incorrect* tag corner order has on solvePnP (*solvePnP expects
        # corners in a different order to compute the correct pose). However, the sparse bundle adjustment code expects
        # a different tag corner order than solvePnP. Therefore, to maintain consistency in how the additive pixel noise
        # is applied, the tag corner order SBA expects is used, thus necessitating this correction.
        new_transform[:3, :3] = np.multiply(
            GraphGenerator.NEGATE_ONLY_YZ_BASIS_VECTORS_ELEMENTWISE, rot_mat
        )
        new_transform[:3, 3] = t_vec.transpose()
        new_transform[3, 3] = 1
        tag_in_phone[:, :] = new_transform
        return True

    # -- Static methods --

    @staticmethod
    def xz_path_ellipsis_four_by_two(
        t_vec: np.ndarray, path_args: Dict[str, Union[float, Tuple[float, float]]]
    ) -> np.ndarray:
        """Defines a parameterized path that is a counterclockwise ellipses in a plane co-planar to the xz plane.

        Args:
            t_vec: N-length vector of parameters to evaluate the curve at
            path_args: Expects a dictionary containing the keys "e_xw", "e_zw", "e_cp", and "xzp" whose values define
             the ellipse's width in the x-direction, width in the z-direction, centerpoint, and y-value of the plane of
             the path respectively.

        Returns:
            3xN array where the rows from top to bottom are the x, y, and z coordinates respectively.

        Raises:
            ValueError: If the path_args dictionary does not contain the expected keys.
        """
        try:
            cp: Tuple[float, float] = path_args["e_cp"]
            return np.vstack(
                (
                    -(path_args["e_xw"] / 2) * np.sin(t_vec) + cp[0],
                    np.ones(
                        len(t_vec),
                    )
                    * path_args["xzp"],
                    -(path_args["e_zw"] / 2) * np.cos(t_vec) + cp[1],
                )
            )
        except KeyError:
            raise ValueError(
                "path_args argument did not contain the expected keys 'e_xw', 'e_zw', 'e_cp', and 'xzp' "
                "for an elliptical path"
            )

    # noinspection PyUnresolvedReferences
    PARAMETERIZED_PATH_ALIAS_TO_CALLABLE: Dict[
        str, Callable[[np.ndarray, Dict[str, float]], np.ndarray]
    ] = {"e": xz_path_ellipsis_four_by_two.__func__}

    @staticmethod
    def frenet_frames(
        t_vec: np.ndarray,
        ftg: Callable[
            [np.ndarray, Dict[str, Union[float, Tuple[float, float]]]], np.ndarray
        ],
        path_args: Dict[str, Union[float, Tuple[float, float]]],
    ) -> np.ndarray:
        """Computes the provided curve's Frenet frames' basis vectors at given points in time.

        Args:
            t_vec: N-length Vector of parameters to evaluate the curve at
            ftg: Function defining the Frenet frame's position in the global reference frame.
            path_args: Value to be passed as the path_args argument for the path invocation

        Returns:
            Nx3x3 numpy array where, for each 3x3 sub-array, the columns from left to right are the T, N, and B basis
             vectors of unit magnitude.
        """
        t_hat = GraphGenerator.d_curve_dt(
            t_vec, GraphGenerator.PATH_LINEAR_DELTA_T, ftg, path_args
        )  # 3xN
        t_hat = norm_array_cols(t_hat)

        # Approximate the derivative of t_hat to get n_hat
        dt_div_2 = GraphGenerator.PATH_LINEAR_DELTA_T / 2
        t_hat_dt_upper = GraphGenerator.d_curve_dt(
            t_vec + dt_div_2, GraphGenerator.PATH_LINEAR_DELTA_T, ftg, path_args
        )
        t_hat_dt_upper = norm_array_cols(t_hat_dt_upper)
        t_hat_dt_lower = GraphGenerator.d_curve_dt(
            t_vec - dt_div_2, GraphGenerator.PATH_LINEAR_DELTA_T, ftg, path_args
        )
        t_hat_dt_lower = norm_array_cols(t_hat_dt_lower)
        n_hat = (t_hat_dt_upper - t_hat_dt_lower) / GraphGenerator.PATH_LINEAR_DELTA_T
        n_hat = norm_array_cols(n_hat)

        # Compute b_hat through the cross product
        b_hat = np.cross(t_hat, n_hat, axis=0)

        ret = np.zeros((len(t_vec), 3, 3))
        ret[:, :, 0] = t_hat.transpose()
        ret[:, :, 1] = n_hat.transpose()
        ret[:, :, 2] = b_hat.transpose()
        return ret

    @staticmethod
    def d_curve_dt(
        t_vec: np.ndarray,
        dt: float,
        path: Callable[
            [np.ndarray, Dict[str, Union[float, Tuple[float, float]]]], np.ndarray
        ],
        path_args: Dict[str, Union[float, Tuple[float, float]]],
    ) -> np.ndarray:
        """Approximates the derivative of a parameterized curve (using the centralized method) with respect to the
        parameter.

        Args:
            t_vec: N-length Vector of parameters to evaluate the curve at
            dt: Parameter delta used for the linear approximation.
            path: Parameterized curve in 3 dimensions.
            path_args: Value to be passed as the path_args argument for the path invocation

        Returns:
            3xN array giving the derivative in each dimension of the curve.
        """
        return (path(t_vec + dt / 2, path_args) - path(t_vec - dt / 2, path_args)) / dt
