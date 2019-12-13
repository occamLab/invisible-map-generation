"""Some helpful functions for visualizing and analyzing graphs.
"""
import numpy as np
from graph import VertexType, Graph
from scipy.spatial.transform import Rotation as R


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
        The 'locations' key covers a (n, 8) array containing x, y, z,
        qx, qy, qz, qw locations of the phone as well as the vertex
        uid at n points.
        The 'tags' and 'waypoints' keys cover the locations of the
        tags and waypoints in the same format.
    """
    locations = np.reshape([], [0, 8])
    tags = np.reshape([], [0, 8])
    waypoints = np.reshape([], [0, 8])

    for i in optimizer.vertices():
        mode = vertices[i].mode
        location = optimizer.vertex(i).estimate().translation()
        rotation = optimizer.vertex(i).estimate().rotation().coeffs()
        pose = np.concatenate([location, rotation, [i]])

        if mode == VertexType.ODOMETRY:
            locations = np.vstack([locations, pose])
        elif mode == VertexType.TAG:
            tags = np.vstack([tags, pose])
        elif mode == VertexType.WAYPOINT:
            waypoints = np.vstack([waypoints, pose])

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
            current_start_found = \
                edge.startuid == graph.edges[segments[i][-1]].enduid
            current_end_found = \
                edge.enduid == graph.edges[segments[i][0]].startuid

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


def get_subgraph(graph, start_vertex_uid, end_vertex_uid):
    edges = ordered_odometry_edges(graph)
    start_found = False
    ret_graph = Graph({}, {})
    for i, edgeuid in enumerate(edges[0]):
        edge = graph.edges[edgeuid]
        if edge.startuid == start_vertex_uid:
            start_found = True

        if start_found:
            ret_graph.vertices[edge.enduid] = graph.vertices[edge.enduid]
            ret_graph.vertices[edge.startuid] = graph.vertices[edge.startuid]
            ret_graph.edges[edgeuid] = edge

        if edge.enduid == end_vertex_uid:
            break

    # Find tags and edges connecting to the found vertices
    for edgeuid in graph.edges:
        edge = graph.edges[edgeuid]
        if graph.vertices[edge.startuid].mode == VertexType.TAG and edge.enduid in ret_graph.vertices:
                ret_graph.edges[edgeuid] = edge
                ret_graph.vertices[edge.startuid] = graph.vertices[edge.startuid]

        if graph.vertices[edge.enduid].mode == VertexType.TAG and edge.startuid in ret_graph.vertices:
            ret_graph.edges[edgeuid] = edge
            ret_graph.vertices[edge.enduid] = graph.vertices[edge.enduid]

    return ret_graph

def get_tags_all_position_estimate(graph, start_vertex_uid, end_vertex_uid):
    tags = np.reshape([], [0, 8]) # [x, y, z, qx, qy, qz, 1, id]
    for edgeuid in graph.edges:
        edge = graph.edges[edgeuid]
        if graph.vertices[edge.startuid].mode == VertexType.ODOMETRY and graph.vertices[edge.enduid].mode == VertexType.TAG:
            odom_transform = np.eye(4)
            odom_position = graph.vertices[edge.enduid].estimate
            odom_rotation = R.from_quat(odom_position[3:]).as_dcm()
            odom_transform[:3, 3] = odom_position[:3]
            odom_transform[:3, :3] = odom_rotation

            edge_transform = np.eye(4)
            edge_rotation = R.from_quat(edge.measurement[3:]).as_dcm()
            edge_transform[:3, 3] = edge.measurement[:3]
            edge_transform[:3, :3] = edge_rotation

            tag_transform = odom_transform.dot(edge_transform)
            #tag_transform = edge_transform.dot(odom_transform)
            tag_translation = tag_transform[:3, 3]
            tag_rotation = R.from_dcm(tag_transform[:3, :3]).as_quat()
            tag_pose = np.concatenate([tag_translation, tag_rotation, [edge.enduid]])
            tags = np.vstack([tags, tag_pose])
    return tags



