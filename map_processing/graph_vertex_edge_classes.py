"""
Vertex, VertexType, and Edge classes which are used in the Graph class.
"""

import pdb
from typing import Union, Dict, Any, Optional, Tuple

import numpy as np
from g2o import SE3Quat

from . import VertexType
from .data_models import OComputeInfParams


class Vertex:
    """A class to contain a vertex of the optimization graph.

    It contains the :class: VertexType of the vertex as well as the pose.
    """

    def __init__(self, mode: VertexType, estimate: np.ndarray, fixed: bool,
                 meta_data: Union[Dict[str, Any], None] = None):
        """The vertex class.

        Args:
            mode: The :class: VertexType of the vertex.
            estimate: The estimate of where the vertex is
        """
        self.mode: VertexType = mode
        self.estimate: np.ndarray = estimate
        self.fixed: bool = fixed
        self.meta_data: Dict = {} if meta_data is None else dict(meta_data)

    def __repr__(self):
        return f"<{'Fixed' if self.fixed else ''}{str(self.mode)[len(VertexType.__name__) + 1:]} Vertex>"


class Edge:
    """A class for graph edges.

    It encodes UIDs for the start and end vertices, the transform_vector, and the information matrix.

    Args:
        startuid: The UID of the starting vertex. This can be any hashable such as an int.
        enduid: The UID of the ending vertex. This can be any hashable such as an int.
        corner_verts: an array of UIDs for each of the tag corner vertices. This only applies in the SBA case
        information_prescaling: A 6 element numpy array encoding the diagonal of a matrix that pre-multiplies the
         edge information matrix specified by the weights. If None is past, then the 6 element vector is assumed to
         be all ones
        camera_intrinsics: [fx, fy, cx, cy] (only applies to tag edges)
        measurement: A 7 element numpy array encoding the measured transform from the start vertex to the end vertex
         in the start vertex's coordinate frame. The format is [x, y, z, qx, qy, qz, qw].
        start_end: A tuple containing the start and end vertices of this edge.

    Attributes:
        information: This edge's information matrix.
    """

    MIN_QUAT_AXIS_VALUES = 0.1 * np.ones(3)

    def __init__(self, startuid: int, enduid: Optional[int], corner_verts: Optional[Dict[int, Vertex]],
                 information_prescaling: Optional[np.ndarray], camera_intrinsics: Optional[np.ndarray],
                 measurement: np.ndarray, start_end: Tuple[Vertex, Optional[Vertex]]):
        self.startuid: int = startuid
        self.enduid: Optional[int] = enduid
        self.corner_verts: Optional[Dict[int, Vertex]] = corner_verts if corner_verts is not None else None
        self.information_prescaling: Optional[np.ndarray] = np.array(information_prescaling)
        self.camera_intrinsics: Optional[np.ndarray] = np.array(camera_intrinsics)
        self.measurement: Optional[np.ndarray] = np.array(measurement)
        self.start_end: Tuple[Vertex, Optional[Vertex]] = start_end
        self.information: np.ndarray = np.eye(2 if corner_verts is not None else (3 if start_end[1] is None else 6))

    def compute_information(self, weights_vec: np.ndarray, compute_inf_params: OComputeInfParams, using_sba: bool) \
            -> None:
        """Computes the information matrix for the edge.

        Notes:
            Depending on the modes indicated in the vertices in the `start_end` and `corner_verts` instance attributes,
             the corresponding method is called. If the corner_verts instance attribute is not None, then it is inferred
             that the edge is to a tag whose corner pixel values are known and being used for sba; the
             `_compute_information_sba` method is subsequently called. If the start and end vertices are both of the
             odometry type, then `_compute_information_se3_nonzero_delta_t` is used. Otherwise, it is assumed that the
             edge is some observation (i.e., both the start and end vertices were sampled at the same time), so the
             `_compute_information_se3_obs` method is called.

        Args:
            weights_vec: A vector of weights that is used to scale the information matrix. Passed as the argument to the
             corresponding downstream `compute_information*` method.
            compute_inf_params: Contains parameters for edge information computation. If both the start
             and end vertices of the edge are odometry vertices, then 'ang_vel_var' and 'lin_vel_var' fields is used
             to specify the angular velocity variance and linear velocity variance, respectively, for the
             `_compute_information_se3_nonzero_delta_t` method. See that method for more information.
            using_sba: True if SBA is used (relevant for the fact that odometry poses are inverted when using SBA)

        Raises:
            ValueError: If the weights_vec argument or resulting information matrix contain any negative values.
        """
        if np.any(weights_vec < 0):
            raise ValueError("The input weight vector should not contain negative values.")

        if self.corner_verts is not None:  # sba corner edge
            self._compute_information_sba(weights_vec, compute_inf_params.tag_sba_var)
        elif self.start_end[1] is None:  # gravity edge
            self._compute_information_gravity(weights_vec)
        else:
            if self.start_end[1].mode == VertexType.ODOMETRY:
                self._compute_information_se3_nonzero_delta_t(
                    weights_vec, using_sba=using_sba, lin_vel_var=compute_inf_params.lin_vel_var,
                    ang_vel_var=compute_inf_params.ang_vel_var)
            else:
                self._compute_information_se3_obs(weights_vec, compute_inf_params.tag_var)

        if np.any(self.information < 0):
            raise ValueError("The information matrix should not contain negative values")
        if len(self.information.shape) != 2:
            raise ValueError(f"The information matrix was computed to be an array with {len(self.information.shape)} "
                             f"dimensions instead of 2 (weights vector argument was an array of shape "
                             f"{weights_vec.shape}")

    def _compute_information_se3_nonzero_delta_t(
            self, weights_vec: np.ndarray, using_sba: bool, ang_vel_var: float = 1.0,
            lin_vel_var: np.array = np.ones(3)) -> None:
        """Compute the 6x6 information matrix for the edges that represent a transform over some nonzero time span.

        Args:
            weights_vec: Length-6 vector used to scale the diagonal of the information matrix.
            ang_vel_var: Scalar used as the angular velocity variance.
            lin_vel_var: Length-3 vector where the elements correspond to the x, y, and z-direction's linear velocity
             variance, respectively.
        """
        self.information = np.diag(weights_vec)
        delta_t_sq = (self.start_end[1].meta_data["timestamp"] - self.start_end[0].meta_data["timestamp"]) ** 2

        # Assume rotational noise can be modeled as a normally-distributed rotational error about the gravity axis.
        # Acquire the gravity axis by selecting the y basis vector of the inverted pose
        if using_sba:  # Poses are inverted when using SBA
            gravity_axis_in_phone_frame = SE3Quat(self.start_end[1].estimate).Quaternion().R[:3, 1]
        else:
            gravity_axis_in_phone_frame = SE3Quat(self.start_end[1].estimate).Quaternion().R[1, :3]

        # Rotation component
        self.information[3:, 3:] *= \
            np.diag(4 / (np.maximum(np.abs(gravity_axis_in_phone_frame), Edge.MIN_QUAT_AXIS_VALUES) *
                         ang_vel_var * delta_t_sq))
        # self.information[3:, 3:] *= np.diag(1 / (np.ones(3) * delta_t_sq * ang_vel_var ** 2))

        # Translation component
        self.information[:3, :3] *= np.diag(1 / (np.ones(3) * delta_t_sq * lin_vel_var ** 2))

    def _compute_information_se3_obs(self, weights_vec: np.ndarray, tag_var: float = 1.0, tag_pos_rot_ratio: float = 1) -> None:
        self.information = np.diag(weights_vec)
        self.information[:3, :3] /= tag_var
        self.information[3:, 3:] /= (tag_var*tag_pos_rot_ratio)

    def _compute_information_sba(self, weights_vec: np.ndarray, tag_sba_var: float = 1.0) -> None:
        self.information = np.diag(weights_vec) * np.diag([1 / tag_sba_var] * 2)

    def _compute_information_gravity(self, weights_vec: np.ndarray) -> None:
        self.information = np.diag(weights_vec)

    def get_start_vertex_type(self, vertices: Dict[int, Vertex]) -> VertexType:
        return vertices[self.startuid].mode

    def get_end_vertex_type(self, vertices: Dict[int, Vertex]) -> Optional[VertexType]:
        if self.enduid is None:  # Is the case when the edge is a gravity edge
            return None
        else:
            return vertices[self.enduid].mode

    def __repr__(self):
        return f"<Edge: {self.start_end[0].__repr__()} -> {self.start_end[1].__repr__()}>"
