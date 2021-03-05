"""
Vertex, VertexType, and Edge classes which are used in the Graph class.
"""

from enum import Enum


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

    def __init__(self, mode, estimate, fixed):
        """The vertex class.

        Args:
            mode: The :class: VertexType of the vertex.
            estimate: The estimate of where the vertex is
        """
        self.mode = mode
        self.estimate = estimate
        self.fixed = fixed
        self.meta_data = {}


class Edge:
    """A class for graph edges.

    It encodes UIDs for the start and end vertices, the measurement, and the information matrix.
    """

    def __init__(self,
                 startuid,
                 enduid,
                 corner_ids,
                 information,
                 information_prescaling,
                 camera_intrinsics,
                 measurement):
        """The edge class.

        The arguments are a startuid, enduid, an information matrix represented by a 6x6 numpy array, and a measurement.

        Args:
            startuid: The UID of the starting vertex. This can be any hashable such as an int.
            enduid: The UID of the ending vertex. This can be any hashable such as an int.
            corner_ids: an array of UIDs for each of the tag corner vertices. This only applies to edges to a tag
            information: A 6x6 numpy array encoding measurement information. The rows and columns encode x, y, z, qx,
             qy, and qz information.
            information_prescaling: A 6 element numpy array encoding he diagonal of a matrix that pre-multiplies the
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
