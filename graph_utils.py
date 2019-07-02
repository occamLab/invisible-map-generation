"""Some helpful functions for visualizing and analyzing graphs.
"""
import numpy as np
from graph import VertexType


def optimizer_to_map(vertices, optimizer):
    """Convert a :class: g2o.SparseOptimizer to a dictionary
    containing locations of the phone, tags, and waypoints.

    Args:
        vertices: A dictionary of vertices.
            This is used to lookup the type of vertex pulled from the
            optimizer.
        optimizer: a :class: g2o.SparseOptimizer containing a map.

    Returns:
        A dictionary with fields 'locations', 'tags', and 'waypoints'.
        The 'locations' key covers a (n, 3) array containing x, y, and
        z locations of the phone at n points.
        The 'tags' and 'waypoints' keys cover the locations of the
        tags and waypoints in the same format.
    """
    locations = np.reshape([], [0, 3])
    tags = np.reshape([], [0, 3])
    waypoints = np.reshape([], [0, 3])

    for i in optimizer.vertices():
        mode = vertices[i].mode
        location = optimizer.vertex(i).estimate().translation()
        if mode == VertexType.ODOMETRY:
            locations = np.vstack([locations, location])
        elif mode == VertexType.TAG:
            tags = np.vstack([tags, location])
        elif mode == VertexType.WAYPOINT:
            waypoints = np.vstack([waypoints, location])

    return {'locations': np.array(locations), 'tags': np.array(tags),
            'waypoints': np.array(waypoints)}
