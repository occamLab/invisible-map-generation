"""
Contains the GraphGenerator class, as well as a main routine for testing it.
"""
import random
import json
from typing import Callable, Tuple, Optional, List, Dict
from enum import Enum

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from map_processing.transform_utils import norm_array_cols, FLIP_Y_AND_Z_AXES
from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from ug_json_encoder import UGJsonEncoder, UGTagDatum, UGPoseDatum
from map_processing import ASSUMED_TAG_SIZE

matplotlib.rcParams['figure.dpi'] = 500


class GraphGenerator:
    """Generates synthetic datasets simulating tag observations.

    Attributes:
        _path: Defines a parameterized path. Takes as input a N-length vector of parameters to evaluate the curve at,
         and returns a 3xN array where the rows from top to bottom are the x, y, and z coordinates respectively.
        _t_max: Max parameter value to use when evaluating the path
        _n_poses: Number of poses to sample the path at (referenced as N in the array dimensions described for other
         attributes).
        _tag_poses: A Mx4x4 array of M tag poses defined in the global reference frame using homogenous transform
         matrices.
        _dist_threshold: Maximum distance from which a tag can be considered observable.
        _aoa_threshold: Maximum angle of attack (in radians) from which a tag can be considered observable. The angle
         of attack is calculated as the angle between the z-axis of the tag pose and the vector from the tag to the
         phone.
        _odometry: Nx4x4 array of N homogenous transforms representing the poses sampled along the path.
        _observation_poses: A list of length N where N is the number of poses sampled along the path. Each element of the
         list is a dictionary mapping tag IDs to their corresponding transforms that give their position in the
         reference frame of that pose. A key-value pair is only present in the dictionary if it is visible from that
         pose.
        _observation_pixels: Nx2x4 array containing the pixel coordinates corresponding to the pose observations
         recorded in _observation_poses.
    """

    class OdomNoiseDims(Enum):
        X = 0
        Y = 1
        Z = 2
        RVert = 3

    CAMERA_INTRINSICS_VEC = [
        1458.0604248046875,  # fx (camera focal length in the x-axis)
        1458.0604248046875,  # fy (camera focal length in the y-axis)
        924.9375,  # cx (principal point offset in the x-axis)
        725.906494140625  # cy (principal point offset along the y-axis)
    ]

    CAMERA_INTRINSICS = np.array([
        [CAMERA_INTRINSICS_VEC[0], 0, CAMERA_INTRINSICS_VEC[2]],
        [0, CAMERA_INTRINSICS_VEC[1], CAMERA_INTRINSICS_VEC[3]],
        [0, 0, 1]
    ])
    _double_camera_intrinsics = 2 * CAMERA_INTRINSICS

    TAG_CORNERS_SIZE_1 = np.transpose(np.array([
        [-1, -1, 0, 1],  # Bottom left
        [1, -1, 0, 1],  # Bottom right
        [1, 1, 0, 1],  # Top right
        [-1, 1, 0, 1],  # Top left
    ]).astype(float))

    PHONE_IN_FRENET = np.array([
        [0, -1, 0, 0],
        [0, 0, 1, 0],
        [-1, 0, 0, 0],
        [0, 0, 0, 1]
    ])

    PATH_LINEAR_DELTA_T = 0.0001  # Time span over which the path can be assumed to be approximately linear
    X_HAT_1X3 = np.array(((1, 0, 0),))
    Y_HAT_1X3 = np.array(((0, 1, 0),))
    Z_HAT_1X3 = np.array(((0, 0, 1),))
    BASES_COLOR_CODES = ("r", "g", "b")

    def __init__(self, path: Callable[[np.ndarray], np.ndarray], t_max: float, n_poses: int = 100,
                 tag_poses: Optional[Dict[int, np.ndarray]] = None, dist_threshold: float = 3.7,
                 aoa_threshold: float = np.pi / 4, tag_size: float = 0.7,
                 odometry_noise: Optional[Dict[OdomNoiseDims, float]] = None):
        """
        Args:
            path: Defines a parameterized path. Takes as input a N-length vector of parameters to evaluate the curve at,
             and returns a 3xN array where the rows from top to bottom are the x, y, and z coordinates respectively.
            t_max: Max parameter value to use when evaluating the path
            n_poses: Number of poses to sample the path at.
            tag_poses: A dictionary mapping tag IDs to their poses in the global reference frame using homogenous
             transform matrices.
            dist_threshold: Maximum distance from which a tag can be considered observable.
            aoa_threshold: Maximum angle of attack (in radians) from which a tag can be considered observable. The angle
             of attack is calculated as the angle between the z-axis of the tag pose and the vector from the tag to the
             phone.
            tag_size: Dimensions (height and width) of the tags in meters.
        """
        self._path: Callable[[np.ndarray], np.ndarray] = path
        self._t_max = t_max
        self._n_poses = n_poses
        self._dist_threshold = dist_threshold
        self._aoa_threshold = aoa_threshold

        self._tag_poses: Dict[int, np.ndarray] = tag_poses if tag_poses is not None else {}
        self._tag_poses_arr = np.zeros((len(self._tag_poses), 4, 4))
        for i, pose in enumerate(self._tag_poses.values()):
            self._tag_poses_arr[i, :, :] = pose

        self._poses: Optional[np.ndarray] = None
        self._odometry_t_vec = np.arange(0, self._t_max, self._t_max / self._n_poses)
        self._delta_t = self._t_max / (self._n_poses - 1)
        self._observation_poses: Optional[List[Optional[Dict[int, np.ndarray]]]] = None
        self._observation_pixels: Optional[List[Optional[Dict[int, np.ndarray]]]] = None

        self._tag_corners_in_tag = GraphGenerator.TAG_CORNERS_SIZE_1
        self._tag_corners_in_tag[:3, :] *= tag_size / 2

        self._cms: Optional[CacheManagerSingleton] = None

        self._odometry_noise_var: Dict[GraphGenerator.OdomNoiseDims, float] = odometry_noise if odometry_noise is not \
                                                                                                None else {
            GraphGenerator.OdomNoiseDims.X: 0,
            GraphGenerator.OdomNoiseDims.Y: 0,
            GraphGenerator.OdomNoiseDims.Z: 0,
            GraphGenerator.OdomNoiseDims.RVert: 0,
        }

    # -- Public methods --

    def export(self) -> UGJsonEncoder:
        """
        Returns:
            A UGJsonEncoder object that, when serialized, will contain the unprocessed graph json encoding of this
             artificial dataset generation.
        """
        pose_data: List[UGPoseDatum] = []
        tag_data: List[List[UGTagDatum]] = []
        tag_data_idx = -1
        for pose_idx in range(self._poses.shape[0]):
            pose_data.append(
                UGPoseDatum(
                    pose=tuple(self._poses[pose_idx, :, :].flatten(order="F")),
                    timestamp=self._odometry_t_vec[pose_idx],
                    # Intentionally skipping planes
                    pose_id=pose_idx
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
                        tag_corners_pixel_coordinates=tuple(
                            self._observation_pixels[pose_idx][tag_id].flatten(order="F")),
                        tag_id=tag_id,
                        pose_id=pose_idx,
                        camera_intrinsics=tuple(np.array(GraphGenerator.CAMERA_INTRINSICS_VEC)),
                        # Intentionally skipping position and orientation variance
                        timestamp=self._odometry_t_vec[pose_idx],
                        tag_pose=tuple(np.matmul(FLIP_Y_AND_Z_AXES,
                                                 self._observation_poses[pose_idx][tag_id]).flatten(order="C")),
                        # Intentionally skipping joint covariance
                    )
                )
        return UGJsonEncoder(
            # Intentionally skipping location data
            map_id="generated_" + str(random.randint(0, int(1e9))),  # arbitrary integer for unique id-ing
            # Intentionally skipping plane data
            pose_data=pose_data,
            tag_data=tag_data
        )

    def export_to_map_processing_cache(self) -> None:
        """Serializes the graph into a json file and saves it to the target destination as set by the TODO
        """
        export_obj = self.export()
        map_dict = export_obj.default(export_obj)
        map_name = map_dict["map_id"]
        json_string = json.dumps(map_dict, indent=2)

        if self._cms is None:
            self._cms = CacheManagerSingleton()
        self._cms.cache_map(
            parent_folder="generated",
            map_info=MapInfo(
                map_name=map_name,
                map_json_name=map_name,
            ),
            json_string=json_string
        )

    def visualize(self, plus_minus_lim=5) -> None:
        """Visualizes the generated graph by plotting the path, the poses on the path, the tags, and the observations of
         those tags.

         TODO: add plot legend.

        Args:
            plus_minus_lim: Value for the x-, y-, and z-lim3d parameters of the matplotlib 3d axes.
        """
        path_t_vec = np.arange(0, 2 * np.pi, 0.01)
        path_samples = self._path(path_t_vec)

        f: plt.Figure = plt.figure()
        ax: Axes3D = f.add_subplot(projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.axes.set_xlim3d(left=-plus_minus_lim, right=plus_minus_lim)
        ax.axes.set_ylim3d(bottom=-plus_minus_lim, top=plus_minus_lim)
        ax.axes.set_zlim3d(bottom=-plus_minus_lim, top=plus_minus_lim)

        plt.plot(path_samples[0, :], path_samples[1, :], path_samples[2, :])
        gg.draw_frames((self._poses[:, :3, 3]).transpose(), self._poses[:, :3, :3], ax)
        gg.draw_frames((self._tag_poses_arr[:, :3, 3]).transpose(), self._tag_poses_arr[:, :3, :3], ax,
                       colors=("m", "m", "m"))

        # Get observation vectors in the global frame and plot them
        for i, dct in enumerate(self._observation_poses):
            pose = self._poses[i, :, :]
            line_start = pose[:3, 3]
            for obs in dct.values():
                # If transforms are computed correctly, then obs_in_global should be equivalent to the original tag pose
                # definition
                obs_in_global = np.matmul(pose, obs)
                line_end = obs_in_global[:3, 3]
                ax.plot(
                    xs=[line_start[0], line_end[0]],
                    ys=[line_start[1], line_end[1]],
                    zs=[line_start[2], line_end[2]],
                    color="c"
                )
        plt.show()

    def generate(self) -> None:
        """Populate the _poses attribute with the poses sampled at the given parameter values, then the
        _observation_poses attribute is populated with the tag observations at each of the poses.
        """
        positions = self._path(self._odometry_t_vec)  # 3xN
        frenet_frames = GraphGenerator.frenet_frames(self._odometry_t_vec, self._path)  # Nx3x3
        true_poses = np.zeros((len(self._odometry_t_vec), 4, 4))  # Nx4x4
        true_poses[:, 3, 3] = 1
        true_poses[:, :3, 3] = positions.transpose()
        true_poses[:, :3, :3] = np.matmul(frenet_frames, GraphGenerator.PHONE_IN_FRENET[:3, :3])

        self._poses = self._apply_noise(true_poses)

        self._observation_poses = []
        self._observation_pixels = []
        for i in range(self._n_poses):
            self._observation_poses.append(dict())
            self._observation_pixels.append(dict())
            for tag_id in self._tag_poses:
                tag_obs, tag_pixels = self._get_tag_observation(self._poses[i, :, :], self._tag_poses[tag_id])
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
            pose_to_pose[i, :, :] = np.matmul(np.linalg.inv(true_poses[i, :, :]), true_poses[i + 1, :, :])

        # Noise model: assume that noise variance is proportional to the elapsed time. noisy_transforms contains the
        # pose-to-pose transforms except with noise applied.
        noisy_transforms = np.zeros((true_poses.shape[0] - 1, 4, 4))
        for i in range(0, true_poses.shape[0] - 1):
            noise_as_transform = np.zeros((4, 4))
            noise_as_transform[3, 3] = 1
            noise_as_transform[:3, 3] = np.array([
                np.random.normal(0, np.sqrt(self._delta_t * self._odometry_noise_var[GraphGenerator.OdomNoiseDims.X])),
                np.random.normal(0, np.sqrt(self._delta_t * self._odometry_noise_var[GraphGenerator.OdomNoiseDims.Y])),
                np.random.normal(0, np.sqrt(self._delta_t * self._odometry_noise_var[GraphGenerator.OdomNoiseDims.Z]))
            ])
            theta = np.random.normal(0, np.sqrt(self._delta_t *
                                                self._odometry_noise_var[GraphGenerator.OdomNoiseDims.RVert]))
            # Interpret rotational noise as noise w.r.t. the rotation about the vertical (x) axis
            noise_as_transform[:3, :3] = np.array([
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)]
            ])
            noisy_transforms[i, :, :] = np.matmul(pose_to_pose[i, :, :], noise_as_transform)

        # Reconstruct new list of poses from the noisy pose-to-pose transforms (dead-reckon, where the initial pose is
        # the same as the true initial pose)
        noisy_poses = np.zeros(true_poses.shape)
        noisy_poses[0, :, :] = true_poses[0, :, :]
        for i in range(0, true_poses.shape[0] - 1):
            noisy_poses[i + 1, :, :] = np.matmul(noisy_poses[i, :, :], noisy_transforms[i, :, :])
        return noisy_poses

    def _generate_time_series_for_equidistant_poses(self, n_odom, t_end):
        raise NotImplementedError()

    def _get_tag_observation(self, obs_from: np.ndarray, tag: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Args:
            obs_from: Pose (4x4 homogenous transform) from which the observation is made. Assumed to be from the same
             reference frame as the tag's pose.
            tag: Pose (4x4 homogenous transform) of the tag. Assumed to be from the same reference frame as the obs_from
             pose.

        Returns:
            If the tag is visible: A tuple containing (1) a 4x4 homogenous transform giving the tag pose in the
            reference frame of the phone and (2) a 2x4 array where each column gives the x-y pixel coordinates of the
            pixel projection for a given corner (corner order determined by the TAG_CORNERS_SIZE_1 class attribute).
            Otherwise, two empty arrays are returned.
        """
        tag_in_phone = np.matmul(np.linalg.inv(obs_from), tag)
        vector_phone_to_tag = tag_in_phone[:3, 3]
        dist_to_tag = np.linalg.norm(vector_phone_to_tag)
        if dist_to_tag > self._dist_threshold:
            return np.array([]), np.array([])

        # Dotting this vector with the tag's z-axis is the same as selecting the z-element of the vector
        dot_for_aoa = -vector_phone_to_tag[2]

        # Calculate aoa in the range [0, pi] rad.
        aoa = np.arccos(dot_for_aoa / dist_to_tag) if dot_for_aoa > 0 else \
            (np.pi - np.arccos(dot_for_aoa / dist_to_tag))
        if aoa > self._aoa_threshold:
            return np.array([]), np.array([])

        # Flip the y and z because the math for the camera intrinsics assumes that +z is increasing depth from the camera
        # (and the AR kit has +z facing out of the screen).
        flipped_tag_in_phone = np.matmul(FLIP_Y_AND_Z_AXES, tag_in_phone)
        tag_corners_in_phone = np.matmul(flipped_tag_in_phone, self._tag_corners_in_tag)

        tag_pixels = np.zeros((2, 4))
        if not GraphGenerator.project_and_check_visible(tag_corners_in_phone[:3, :], tag_pixels):
            return np.array([]), np.array([])

        return tag_in_phone, tag_pixels

    # -- Static methods --

    @staticmethod
    def project_and_check_visible(points: np.ndarray, pixel_vals: Optional[np.ndarray] = None) -> bool:
        """Project points in 3D space into phone space.

        Args:
            points: 3xN array of N vectors (or 1-D length-3 array) (in the phone's reference frame) to points in 3D
             space.
            pixel_vals: If not none, then the array provided by this argument is expected to be a 2xN array (must be a
             2-D array; cannot be a 1-D array). The columns of this array will be populated with the pixel coordinates
             of the resulting projection of the points into pixel space. Row order for coordinates is x then y.
        Returns:
            True if all points are visible according to the camera intrinsics.
        """
        # 3xN array of N points' pixel coordinates (accurate only after subsequent normalization)
        pixel_coords = np.matmul(GraphGenerator.CAMERA_INTRINSICS, points)
        if len(pixel_coords.shape) == 1:
            pixel_coords = np.expand_dims(pixel_coords, axis=1)

        # Normalize values so that first and second rows contain the actual pixel values
        for col_idx in range(pixel_coords.shape[1]):
            pixel_coords[:, col_idx] = pixel_coords[:, col_idx] / pixel_coords[2, col_idx]

        # Copy values into argument if provided
        if isinstance(pixel_vals, np.ndarray):
            pixel_vals[:, :] = pixel_coords[:2, :]

        # Check if all pixel coordinates are within the sensor's bounds
        return not (
                np.any(pixel_coords[0:2, :] < 0) or
                np.any(pixel_coords[0, :] > GraphGenerator._double_camera_intrinsics[0, 2]) or
                np.any(pixel_coords[1, :] > GraphGenerator._double_camera_intrinsics[1, 2])
        )

    @staticmethod
    def path_ellipsis_four_by_two(t_vec: np.ndarray) -> np.ndarray:
        """Defines a parameterized path that is a counterclockwise ellipses of width 4 and height 2 in the x-y plane.

        Args:
            t_vec: N-length vector of parameters to evaluate the curve at

        Returns:
            3xN array where the rows from top to bottom are the x, y, and z coordinates respectively.
        """
        return np.vstack((-4 * np.sin(t_vec), np.zeros(len(t_vec), ), -2 * np.cos(t_vec)))

    @staticmethod
    def draw_frames(offsets: np.ndarray, frames: np.ndarray, plt_axes: plt.Axes,
                    colors: Tuple[str, str, str] = ("r", "g", "b")) -> None:
        """Draw N reference frames at given translation offsets.

        Args:
            offsets: Translation offsets of the frames. Expected to be a 3xN matrix where the rows from top to bottom
             encode the translation offset in the first, second, and third dimensions, respectively.
            frames: Nx3x3 array of rotation matrices.
            plt_axes: Matplotlib axes to plot on
            colors: Tuple of color codes to use for the first, second, and third dimensions' basis vector arrows,
             respectively.
        """
        for b in range(3):
            basis_vectors_transposed = (frames[:, :, b]).transpose()
            plt_axes.quiver(
                offsets[0, :],
                offsets[1, :],
                offsets[2, :],
                # For each basis vector, dot it with the corresponding basis vector of the reference frame it is within
                np.matmul(GraphGenerator.X_HAT_1X3, basis_vectors_transposed),
                np.matmul(GraphGenerator.Y_HAT_1X3, basis_vectors_transposed),
                np.matmul(GraphGenerator.Z_HAT_1X3, basis_vectors_transposed),
                length=0.5,
                arrow_length_ratio=0.3,
                normalize=True,
                color=colors[b],
            )

    @staticmethod
    def frenet_frames(t_vec: np.ndarray, ftg: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Computes the provided curve's Frenet frames' basis vectors at given points in time.

        Args:
            t_vec: N-length Vector of parameters to evaluate the curve at
            ftg: Function defining the Frenet frame's position in the global reference frame.

        Returns:
            Nx3x3 numpy array where, for each 3x3 sub-array, the columns from left to right are the T, N, and B basis
             vectors of unit magnitude.
        """
        t_hat = GraphGenerator.d_curve_dt(t_vec, GraphGenerator.PATH_LINEAR_DELTA_T, ftg)  # 3xN
        t_hat = norm_array_cols(t_hat)

        # Approximate the derivative of t_hat to get n_hat
        dt_div_2 = GraphGenerator.PATH_LINEAR_DELTA_T / 2
        t_hat_dt_upper = GraphGenerator.d_curve_dt(t_vec + dt_div_2, GraphGenerator.PATH_LINEAR_DELTA_T, ftg)
        t_hat_dt_upper = norm_array_cols(t_hat_dt_upper)
        t_hat_dt_lower = GraphGenerator.d_curve_dt(t_vec - dt_div_2, GraphGenerator.PATH_LINEAR_DELTA_T, ftg)
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
    def d_curve_dt(t_vec: np.ndarray, dt: float, curve: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Approximates the derivative of a parameterized curve (using the centralized method) with respect to the
        parameter.

        Args:
            t_vec: N-length Vector of parameters to evaluate the curve at
            dt: Time delta used for the linear approximation.
            curve: Parameterized curve in 3 dimensions.

        Returns:
            3xN array giving the derivative in each dimension of the curve.
        """
        return (curve(t_vec + dt / 2) - curve(t_vec - dt / 2)) / dt


if __name__ == "__main__":
    gg = GraphGenerator(
        path=GraphGenerator.path_ellipsis_four_by_two,
        tag_poses={
            0: np.array([
                [1, 0, 0, -3],
                [0, 1, 0, 0],
                [0, 0, 1, -4],
                [0, 0, 0, 1]
            ]),
            1: np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, -4],
                [0, 0, 0, 1]
            ]),
            2: np.array([
                [1, 0, 0, 3],
                [0, 1, 0, 0],
                [0, 0, 1, -4],
                [0, 0, 0, 1]
            ])
        },
        t_max=6 * np.pi,
        n_poses=600,
        tag_size=ASSUMED_TAG_SIZE
    )
    gg.generate()
    gg.visualize()
    gg.export_to_map_processing_cache()
