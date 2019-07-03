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


def ordered_odometry_edges(graph):
    """Generate a list of a list of edges ordered by start of path to end.

    The lists are different connected paths.
    As long as the graph is connected, the output list should only one
    list of edges.

    Args:
        graph: The graph to extract the ordered edges from.

    Returns: A list of lists of edge UIDs, where each sublist is a
        sequence of connected edges.
    """
    segments = []

    for uid in graph.edges:
        edge = graph.edges[uid]
        if graph.vertices[edge.startuid].mode != VertexType.ODOMETRY \
                or graph.vertices[edge.enduid].mode != VertexType.ODOMETRY:
            continue
        start_found = end_found = False
        start_found_idx = end_found_idx = 0

        for i in range(len(segments) - 1, -1, -1):
            current_start_found = edge.startuid == graph.edges[segments[i][-1]].enduid
            current_end_found = edge.enduid == graph.edges[segments[i][0]].startuid

            if current_start_found:
                start_found = True
                start_found_idx = i

            elif current_end_found:
                end_found = True
                end_found_idx = i

            if current_start_found and end_found:
                segments[i].append(uid)
                segments[i].extend(segments[end_found_idx])
                del segments[end_found_idx]
                break

            elif current_end_found and start_found:
                segments[start_found_idx].append(uid)
                segments[i] = segments[start_found_idx] + segments[i]
                del segments[start_found_idx]
                break

        if start_found and not end_found:
            segments[start_found_idx].append(uid)

        elif end_found and not start_found:
            segments[end_found_idx].insert(0, uid)

        elif not (start_found or end_found):
            segments.append([uid])

    return segments
