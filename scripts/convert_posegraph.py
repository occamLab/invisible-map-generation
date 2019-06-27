from pose_graph import PoseGraph
import itertools
import numpy as np
from graph import Vertex, Edge, Graph, VertexType
from scipy.spatial.transform import Rotation as R
import pickle


def convertVertex(vertex):
    if vertex.fix_status:
        vertextype = VertexType.DUMMY
    elif vertex.type == 'tag':
        vertextype = VertexType.TAG
    elif vertex.type == 'odometry':
        vertextype = VertexType.ODOMETRY
    elif vertex.type == 'waypoint':
        vertextype = VertexType.WAYPOINT
    else:
        raise Exception("Vertex type {} not recognized".format(vertex.type))

    return (vertex.id, Vertex(mode=vertextype,
                              value=np.concatenate
                              ([vertex.translation, (vertex.rotation)])
                              ))


def convertEdge(edge):
    return Edge(startuid=edge.start.id, enduid=edge.end.id,
                importance=edge.importance_matrix,
                change=np.concatenate
                ([edge.translation, edge.rotation]))


def convert(posegraph):
    vertices = {}
    edges = {}
    edgeUid = 0

    for startid in posegraph.odometry_edges:
        for endid in posegraph.odometry_edges[startid]:
            edge = posegraph.odometry_edges[startid][endid]
            endpoints = [edge.start, edge.end]

            for vertex in endpoints:
                uid, converted = convertVertex(vertex)
                vertices[uid] = converted

            edges[edgeUid] = convertEdge(edge)
            edgeUid += 1

    for startid in posegraph.odometry_tag_edges:
        for endid in posegraph.odometry_tag_edges[startid]:
            edge = posegraph.odometry_tag_edges[startid][endid]
            endpoints = [edge.start, edge.end]

            for vertex in endpoints:
                uid, converted = convertVertex(vertex)
                vertices[uid] = converted

            edges[edgeUid] = convertEdge(edge)
            edgeUid += 1

    for startid in posegraph.odometry_waypoints_edges:
        for endid in posegraph.odometry_waypoints_edges[startid]:
            edge = posegraph.odometry_waypoints_edges[startid][endid]
            endpoints = [edge.start, edge.end]

            for vertex in endpoints:
                uid, converted = convertVertex(vertex)
                vertices[uid] = converted

            edges[edgeUid] = convertEdge(edge)
            edgeUid += 1

    return Graph(vertices=vertices, edges=edges)
