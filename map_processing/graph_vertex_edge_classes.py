"""
Vertex, VertexType, and Edge classes which are used in the Graph class.
"""

from enum import Enum
from typing import List, Union, Dict, Any

import numpy as np


class VertexType(Enum):
    """An enumeration containing the vertex types ODOMETRY, TAG, DUMMY, and WAYPOINT.
    """
    ODOMETRY = 0
    TAG = 1
    TAGPOINT = 2
    DUMMY = 3
    WAYPOINT = 4


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
        self.meta_data: Dict = {} if meta_data is None else meta_data


class Edge:
    """A class for graph edges.

    It encodes UIDs for the start and end vertices, the transform_vector, and the information matrix.
    """

    def __init__(self, startuid: int, enduid: int, corner_ids: Union[None, List], information: np.ndarray,
                 information_prescaling: Union[None, np.ndarray], camera_intrinsics: Union[None, np.ndarray],
                 measurement: np.ndarray):
        """The edge class.

        The arguments are a startuid, enduid, an information matrix represented by a 6x6 numpy array, and a transform_vector.

        Args:
            startuid: The UID of the starting vertex. This can be any hashable such as an int.
            enduid: The UID of the ending vertex. This can be any hashable such as an int.
            corner_ids: an array of UIDs for each of the tag corner vertices. This only applies to edges to a tag
            information: A 6x6 numpy array encoding transform_vector information. The rows and columns encode x, y, z, qx,
             qy, and qz information.
            information_prescaling: A 6 element numpy array encoding the diagonal of a matrix that pre-multiplies the
             edge information matrix specified by the weights. If None is past, then the 6 element vector is assumed to
             be all ones
            camera_intrinsics: [fx, fy, cx, cy] (only applies to tag edges)
            measurement: A 7 element numpy array encoding the measured transform from the start vertex to the end vertex
             in the start vertex's coordinate frame. The format is [x, y, z, qx, qy, qz, qw].
        """
        self.startuid = startuid
        self.enduid = enduid
        self.corner_ids = corner_ids
        self.information = information
        self.information_prescaling = information_prescaling
        self.camera_intrinsics = camera_intrinsics
        self.measurement = measurement

    def get_start_vertex_type(self, vertices: Dict[int, Vertex]) -> VertexType:
        return vertices[self.startuid].mode

    def get_end_vertex_type(self, vertices: Dict[int, Vertex]) -> VertexType:
        return vertices[self.enduid].mode
