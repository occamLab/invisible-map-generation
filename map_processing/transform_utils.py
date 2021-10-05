"""
Utilities for manipulating transformations and providing other helpful matrix operations.
"""

from typing import List

import g2o
from typing import Tuple
import numpy as np
from g2o import SE3Quat, Quaternion
from scipy.spatial.transform import Rotation as Rot


def get_tag_estimate_in_global_frame_when_sba(estimate: np.ndarray) -> np.ndarray:
    """Derives tag pose in the global reference frame from one reported when using SBA.

    Tag observations when SBA is used result in a +1m offset being applied along the tag's Z axis; this function
    accounts for that.

    Args:
        estimate: A length-7 vector that contains the pose estimate of the tag in the camera reference frame.
    Returns:
        A vector adjusted by -1m along its local Z axis, then inverted.
    """
    return np.append(
        (SE3Quat([0, 0, -1, 0, 0, 0, 1]) * SE3Quat(estimate[:-1]).inverse()).inverse().to_vector(),
        estimate[-1]
    )


def se3_quat_average(transforms: List[SE3Quat]) -> SE3Quat:
    """Computes the average transform from a list of transforms.

    Args:
        transforms: List of transforms

    Returns:
        Average transform
    """
    translation_average = sum([t.translation() / len(transforms) for t in transforms])
    epsilons = np.ones(len(transforms), )
    converged = False
    quat_average = None
    while not converged:
        quat_sum = sum(np.array([t.orientation().x(), t.orientation().y(), t.orientation().z(), t.orientation().w()]) \
                       * epsilons[idx] for idx, t in enumerate(transforms))
        quat_average = quat_sum / np.linalg.norm(quat_sum)
        same_epsilon = [np.linalg.norm(epsilons[idx] * np.array([t.orientation().x(), t.orientation().y(),
                                                                 t.orientation().z(), t.orientation().w()]) - \
                                       quat_average) for idx, t in enumerate(transforms)]
        swap_epsilon = [np.linalg.norm(-epsilons[idx] * np.array([t.orientation().x(), t.orientation().y(),
                                                                  t.orientation().z(), t.orientation().w()]) - \
                                       quat_average) for idx, t in enumerate(transforms)]

        change_mask = np.greater(same_epsilon, swap_epsilon)
        epsilons[change_mask] = -epsilons[change_mask]
        converged = not np.any(change_mask)
    average_as_quat = Quaternion(quat_average[3], quat_average[0], quat_average[1], quat_average[2])
    return SE3Quat(average_as_quat, translation_average)


def transform_vector_to_matrix(transform_vector: np.ndarray) -> np.ndarray:
    """Convert a vectorized transform into a transform matrix.

    Args:
        transform_vector: 7-element vector in the form of [x_trans, y_trans, z_trans, rot_x, rot_y, rot_z, rot_w]

    Returns:
        4x4 matrix containing the corresponding homogenous transform.
    """
    transformation = np.eye(4)
    transformation[:3, 3] = transform_vector[:3]
    transformation[:3, :3] = Rot.from_quat(transform_vector[3:7]).as_matrix()
    return transformation


def translation_vector_to_matrix(translation_vector: np.ndarray) -> np.ndarray:
    """Convert a vectorized translation into a transform matrix.

    Args:
        translation_vector: 3-element vector in the form of [x, y, z]

    Returns:
        4x4 matrix containing the corresponding homogenous transform
    """
    transformation = np.zeros(4)
    transformation[3, 3] = 1
    transformation[:3, 3] = translation_vector
    return transformation


def pose_to_isometry(pose: np.ndarray) -> g2o.Isometry3d:
    """Convert a pose vector to a g2o.Isometry3d object.

    Args:
        pose: A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and qw respectively.
    Returns:
        A :class: g2o.Isometry3d instance encoding the same information as the input pose.
    """
    return g2o.Isometry3d(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])


def pose_to_se3quat(pose: np.ndarray) -> g2o.Isometry3d:
    """Convert a pose vector to a g2o.Isometry3d object.

    Args:
        pose: A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and qw respectively.
    Returns:
        A :class: g2o.Isometry3d instance encoding the same information as the input pose.
    """
    return g2o.SE3Quat(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])


def isometry_to_pose(isometry: g2o.Isometry3d) -> np.ndarray:
    """Convert a :class: g2o.Isometry3d to a vector containing a pose.

    Args:
        isometry: A :class: g2o.Isometry3d instance.
    Returns:
        A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and qw respectively.
    """
    return np.concatenate([isometry.translation(), isometry.rotation().coeffs()])


def global_yaw_effect_basis(rotation, gravity_axis='z'):
    """Form a basis which describes the effect of a change in global yaw on a local transform_vector's qx, qy, and qz.

    Since the accelerometer measures gravitational acceleration, it can accurately measure the global z-azis but its
    transform_vector of the orthogonal axis are less reliable.

    Args:
        rotation: A :class: scipy.spatial.transform.Rotation encoding a local rotation.
        gravity_axis: A character specifying the gravity axis (e.g., 'z')

    Returns:
        A 3x3 numpy array where the columns are the new basis.
    """
    rotation1 = Rot.from_euler(gravity_axis, 0.05) * rotation
    change = rotation1.as_quat()[:3] - rotation.as_quat()[:3]
    return np.linalg.svd(change[:, np.newaxis])[0]


def locations_from_transforms(locations):
    """TODO: documentation

    Args:
        locations:

    Returns:

    """
    for i in range(locations.shape[0]):
        locations[i, :7] = SE3Quat(locations[i, :7]).inverse().to_vector()
    return locations


def transform_matrix_to_vector(pose: np.ndarray, invert=False) -> np.ndarray:
    """Convert a pose/multiple poses in homogenous transform matrix form to [x, y, z, qx, qy, qz, qw].

    Args:
        pose (np.ndarray): Pose or array of poses in matrix form. The poses are converted along the last two axes.
        invert (bool): If inverted, then the return enum_value will be inverted
    Returns:
      Converted pose or array of poses (the output has one fewer dimensions than the input)
    """
    translation = pose[..., :3, 3]
    if pose.shape[0] != 0:
        rotation = Rot.from_matrix(pose[..., :3, :3]).as_quat()
    else:
        rotation = np.zeros([0, 4])
    ret_val = np.concatenate([translation, rotation], axis=-1)
    if invert:
        ret_val = np.vstack(list(map(lambda measurement: SE3Quat(measurement).inverse().to_vector(), ret_val)))
    return ret_val


def pose2diffs(poses):
    """Convert an array of poses in the odom frame to an array of
    transformations from the last pose.

    Args:
      poses (np.ndarray): Pose or array of poses.
    Returns:
      An array of transformations
    """
    diffs = []
    for previous_pose, current_pose in zip(poses[:-1], poses[1:]):
        diffs.append(np.linalg.inv(previous_pose).dot(current_pose))
    diffs = np.array(diffs)
    return diffs


def make_sba_tag_arrays(tag_size) -> Tuple[np.ndarray, np.ndarray]:
    """Generate tag coordinates given a specified tag size (assuming relative to a reference frame 1m in front of the
    tag).

    Args:
        tag_size: Size of the square tag's side length.

    Returns:
        true_3d_points: A 4x3 array of the tag's 4 xyz coordinates in the reference frame that is offset from the center
         of the tag by -1 in the Z axis. Order of tags is bottom-left, bottom-right, top-right, and top-left.
        true_tag_center: The length-3 vector containing (0, 0, 1).
    """
    pos_tag_sz_div_2 = tag_size / 2
    neg_tag_sz_div_2 = - pos_tag_sz_div_2

    true_3d_points = np.array(
        [
            [neg_tag_sz_div_2, neg_tag_sz_div_2, 1],  # Bottom-left
            [pos_tag_sz_div_2, neg_tag_sz_div_2, 1],  # Bottom-right
            [pos_tag_sz_div_2, pos_tag_sz_div_2, 1],  # Top-right
            [neg_tag_sz_div_2, pos_tag_sz_div_2, 1]  # Top-left
        ]
    )

    true_3d_tag_center = np.array([0, 0, 1])
    return true_3d_points, true_3d_tag_center


def norm_array_cols(arr: np.ndarray) -> np.ndarray:
    """Normalize each column of the array.

    Args:
        arr: 2-dimensional array.

    The arr argument is not modified (because of how numpy arrays are passed).
    """
    # Is there a better way to do this?
    norm = np.linalg.norm(arr, axis=0)
    for i in range(arr.shape[0]):
        arr[i, :] = np.divide(arr[i, :], norm)
    return arr


FLIP_Y_AND_Z_AXES = np.array(
    [
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ]
)