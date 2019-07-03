"""Some helpful functions for visualizing and analyzing graphs.
"""
import itertools
import numpy as np
from graph import VertexType, Graph


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


def connected_components(graph):
    """Return a list of graphs representing connecting components of
    the input graph.

    If the graph is connected, there should only be one element in the
    output.

    Args:
        graph: A :class: Graph to be separated into connected
            components.

    Returns: A list of :class: Graph containing the connected
        components.
    """
    groups = []
    for uid in graph.edges:
        edge = graph.edges[uid]
        uids = {edge.startuid, edge.enduid}
        membership = []
        for i, group in enumerate(groups):
            if group[0] & uids:
                membership.append(i)

        new_group = set.union(uids, *[groups[i][0] for i in membership]), \
            set.union({uid}, *[groups[i][1] for i in membership])

        membership.reverse()

        for i in membership:
            del groups[i]

        groups.append(new_group)

    return [Graph(vertices={k: graph.vertices[k] for k in group[0]},
                  edges={k: graph.edges[k] for k in group[1]})
            for group in groups]
