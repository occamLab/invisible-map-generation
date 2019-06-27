import itertools
import g2o
import numpy as np


def pose2Isometry(pose):
    return g2o.Isometry3d(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])


def graph2Optimizer(graph):
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
        g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())))

    for i in graph.vertices:
        vertex = g2o.VertexSE3()
        vertex.set_id(i)
        vertex.set_estimate(pose2Isometry(graph.vertices[i].value))
        vertex.set_fixed(graph.vertices[i].mode == VertexType.DUMMY)
        optimizer.add_vertex(vertex)

    for i in graph.edges:
        edge = g2o.EdgeSE3()

        for j, k in enumerate([i.startuid, i.enduid]):
            edge.set_vertex(j, optimizer.vertex(k))

        edge.set_measurement(pose2Isometry(i.change))
        edge.set_information(i.importance)

        optimizer.add_edge(edge)

    return optimizer


class VertexType:
    ODOMETRY = 0
    TAG = 1
    DUMMY = 2
    WAYPOINT = 3


class Vertex:
    def __init__(self, mode, value):
        self.mode = mode
        self.value = value


class Edge:
    def __init__(self, startuid, enduid, importance, change):
        self.startuid = startuid
        self.enduid = enduid
        self.importance = importance
        self.change = change


class Graph:
    def __init__(self, vertices, edges):
        self.edges = edges
        self.vertices = vertices

    def connectedComponents(self):
        groups = []
        for edge in self.edges:
            uids = {edge.startuid, edge.enduid}
            membership = []
            for i, group in enumerate(groups):
                if len(group[0] & uids) > 0:
                    membership.append(i)

            newGroup = (set.union(uids, *[groups[i][0] for i in membership]),
                        [edge] + list(itertools.chain.from_iterable
                                      ([groups[i][1] for i in membership])))

            membership.reverse()

            for i in membership:
                del groups[i]

            groups.append(newGroup)

        return [Graph(vertices={k: self.vertices[k] for k in group[0]},
                      edges=group[1]) for group in groups]

    def odometryGraph(self):
        odometryEdges = []
        odometryVertices = {}
        for edge in self.edges:
            if self.vertices[edge.startuid].mode == VertexType.ODOMETRY \
               and self.vertices[edge.enduid].mode == VertexType.ODOMETRY:
                odometryVertices[edge.startuid] = self.vertices[edge.startuid]
                odometryVertices[edge.enduid] = self.vertices[edge.enduid]
                odometryEdges.append(edge)

        return Graph(vertices={k: self.vertices[k] for k in odometryVertices},
                     edges=odometryEdges)

    def orderedOdometryEdges(self):
        segments = []

        for edge in self.edges:
            if self.vertices[edge.startuid].mode != VertexType.ODOMETRY \
               or self.vertices[edge.enduid].mode != VertexType.ODOMETRY:
                continue
            startFound = endFound = False
            startFoundIdx = endFoundIdx = 0

            for i in range(len(segments) - 1, -1, -1):
                currentStartFound = edge.startuid == segments[i][-1].enduid
                currentEndFound = edge.enduid == segments[i][0].startuid

                if currentStartFound:
                    startFound = True
                    startFoundIdx = i

                elif currentEndFound:
                    endFound = True
                    endFoundIdx = i

                if currentStartFound and endFound:
                    segments[i].append(edge)
                    segments[i].extend(segments[endFoundIdx])
                    del segments[endFoundIdx]
                    break

                elif currentEndFound and startFound:
                    segments[startFoundIdx].append(edge)
                    segments[i] = segments[startFoundIdx] + segments[i]
                    del segments[startFoundIdx]
                    break

            if startFound and not endFound:
                segments[startFoundIdx].append(edge)

            elif endFound and not startFound:
                segments[endFoundIdx].insert(0, edge)

            elif not (startFound or endFound):
                segments.append([edge])

        return segments
