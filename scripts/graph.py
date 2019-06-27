import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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


def optimizer2map(vertices, optimizer):
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

    def generateUnoptimizedGraph(self):
        self.unoptimizedGraph = graph2Optimizer(self)
        return self.unoptimizedGraph

    def optimizeGraph(self):
        self.optimizedGraph = graph2Optimizer(self)

        initStatus = self.optimizedGraph.initialize_optimization()
        runStatus = self.optimizedGraph.optimize(20)

        return initStatus and runStatus

    def plotMap(self):
        unoptimized = optimizer2map(self.vertices, self.unoptimizedGraph)
        optimized = optimizer2map(self.vertices, self.optimizedGraph)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        tagMarker = '^'
        waypointMarker = 's'
        locationMarker = '.'

        ax.plot(unoptimized['locations'][:, 0], unoptimized['locations'][:, 1],
                unoptimized['locations'][:, 2], locationMarker,
                label='Uncorrected Path')
        ax.plot(optimized['locations'][:, 0], optimized['locations'][:, 1],
                optimized['locations'][:, 2], locationMarker,
                label='Corrected Path')

        ax.plot(unoptimized['tags'][:, 0], unoptimized['tags'][:, 1],
                unoptimized['tags'][:, 2], tagMarker, label='Uncorrected Tags')
        ax.plot(optimized['tags'][:, 0], optimized['tags'][:, 1],
                optimized['tags'][:, 2], tagMarker, label='Corrected Tags')

        ax.plot(unoptimized['waypoints'][:, 0], unoptimized['waypoints'][:, 1],
                unoptimized['waypoints'][:, 2], waypointMarker,
                label='Uncorrected Waypoints')
        ax.plot(optimized['waypoints'][:, 0], optimized['waypoints'][:, 1],
                optimized['waypoints'][:, 2], waypointMarker,
                label='Corrected Waypoints')

        ax.legend()

        return fig

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
