"""
Utilities for manipulating transformations.
"""

from typing import List

import g2o
import numpy as np
from g2o import SE3Quat, Quaternion
from scipy.spatial.transform import Rotation as Rot


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


def measurement_to_matrix(transform_vector: np.ndarray) -> np.ndarray:
    """Convert a vectorized transform into a transform matrix.

    Args:
        transform_vector: 7-element 1D numpy array in the form of [translation, rotation]

    Returns:
        4x4 numpy array representing the transform
    """
    transformation = np.eye(4)
    transformation[:3, 3] = transform_vector[:3]
    transformation[:3, :3] = Rot.from_quat(transform_vector[3:7]).as_matrix()
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
    return np.concatenate(
        [isometry.translation(), isometry.rotation().coeffs()])


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


def matrix2measurement(pose, invert=False):
    """Convert a pose or array of poses in matrix form to [x, y, z,
    qx, qy, qz, qw].

    The output will have one fewer dimension than the input.

    Args:
        pose (np.ndarray): Pose or array of poses in matrix form.
         The poses are converted along the last two axes.
        invert (bool): If inverted, then the return enum_value will be inverted
    Returns:
      Converted pose or array of poses.
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