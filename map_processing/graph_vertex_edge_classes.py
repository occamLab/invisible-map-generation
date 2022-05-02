"""
Vertex, VertexType, and Edge classes which are used in the Graph class.
"""

from typing import List, Union, Dict, Any, Optional, Tuple

import numpy as np

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


class Edge:
    """A class for graph edges.

    It encodes UIDs for the start and end vertices, the transform_vector, and the information matrix.

    Args:
        startuid: The UID of the starting vertex. This can be any hashable such as an int.
        enduid: The UID of the ending vertex. This can be any hashable such as an int.
        corner_ids: an array of UIDs for each of the tag corner vertices. This only applies in the SBA case
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

    WEIGHTS_ONLY = False
    """
    If true, the information matrices are exclusively computed as diagonal matrices from the weight vectors.
    """

    def __init__(self, startuid: int, enduid: Optional[int], corner_ids: Optional[List[int]],
                 information_prescaling: Optional[np.ndarray], camera_intrinsics: Optional[np.ndarray],
                 measurement: np.ndarray, start_end: Tuple[Vertex, Optional[Vertex]]):
        self.startuid: int = startuid
        self.enduid: Optional[int] = enduid
        self.corner_ids: Optional[List[int]] = list(corner_ids) if corner_ids is not None else None
        self.information_prescaling: Optional[np.ndarray] = np.array(information_prescaling)
        self.camera_intrinsics: Optional[np.ndarray] = np.array(camera_intrinsics)
        self.measurement: Optional[np.ndarray] = np.array(measurement)
        self.start_end: Tuple[Vertex, Vertex] = start_end
        self.information: np.ndarray = np.eye(2 if corner_ids is not None else (3 if start_end[1] is None else 6))

    def compute_information(self, weights_vec: np.ndarray, compute_inf_params: Optional[OComputeInfParams] = None) -> \
            None:
        """Computes the information matrix for the edge.

        Notes:
            Depending on the modes indicated in the vertices in the `start_end` and `corner_ids` instance attributes,
             the corresponding method is called. If the corner_ids instance attribute is not None, then it is inferred
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

        Raises:
            ValueError: If the weights_vec argument or resulting information matrix contain any negative values.
        """
        if np.any(weights_vec < 0):
            raise ValueError("The input weight vector should not contain negative values.")

        if compute_inf_params is None:
            compute_inf_params = OComputeInfParams()

        if self.corner_ids is not None:  # sba corner edge
            self._compute_information_sba(weights_vec)
        elif self.start_end[1] is None:  # gravity edge
            self._compute_information_gravity(weights_vec)
        else:
            if self.start_end[1].mode == VertexType.ODOMETRY:
                self._compute_information_se3_nonzero_delta_t(weights_vec, lin_vel_var=compute_inf_params.lin_vel_var,
                                                              ang_vel_var=compute_inf_params.ang_vel_var)
            else:
                self._compute_information_se3_obs(weights_vec)

        if np.any(self.information < 0):
            raise ValueError("The information matrix should not contain negative values")
        if len(self.information.shape) != 2:
            raise ValueError(f"The information matrix was computed to be an array with {len(self.information.shape)} "
                             f"dimensions instead of 2 (weights vector argument was an array of shape "
                             f"{weights_vec.shape}")

    def _compute_information_se3_nonzero_delta_t(self, weights_vec: np.ndarray, ang_vel_var: float = 1,
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

        # TODO: Should the pose-to-pose transforms be used here in some way?
        self.information[3:, 3:] *= np.diag(4 * np.ones(3) / (ang_vel_var ** 2 * delta_t_sq))
        self.information[:3, :3] *= np.diag(1 / (delta_t_sq * lin_vel_var ** 2))

    def _compute_information_se3_obs(self, weights_vec: np.ndarray) -> None:
        self.information = np.diag(weights_vec)

    def _compute_information_sba(self, weights_vec: np.ndarray) -> None:
        self.information = np.diag(weights_vec)

    def _compute_information_gravity(self, weights_vec: np.ndarray) -> None:
        self.information = np.diag(weights_vec)

    def get_start_vertex_type(self, vertices: Dict[int, Vertex]) -> VertexType:
        return vertices[self.startuid].mode

    def get_end_vertex_type(self, vertices: Dict[int, Vertex]) -> Optional[VertexType]:
        if self.enduid is None:  # Is the case when the edge is a gravity edge
            return None
        else:
            return vertices[self.enduid].mode
