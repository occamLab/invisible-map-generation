import itertools
from maximization_model import maxweights
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import g2o
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


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

        for j, k in enumerate([graph.edges[i].startuid,
                               graph.edges[i].enduid]):
            edge.set_vertex(j, optimizer.vertex(k))

        edge.set_measurement(pose2Isometry(graph.edges[i].change))
        edge.set_information(graph.edges[i].information)
        edge.set_id(i)

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


def globalYawEffectBasis(rotation):
    rotation1 = R.from_euler('z', 0.05) * rotation
    change = rotation1.as_quat()[:3] - rotation.as_quat()[:3]
    return np.linalg.svd(change[:, np.newaxis])[0]


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
    def __init__(self, startuid, enduid, information, change):
        self.startuid = startuid
        self.enduid = enduid
        self.information = information
        self.change = change


class Graph:
    def __init__(self, vertices, edges):
        self.edges = edges
        self.vertices = vertices
        self.generateBasisMatrices()

    def generateBasisMatrices(self):
        basisMatrices = {}

        for uid in self.edges:
            if (self.vertices[self.edges[uid].startuid].mode
                    == VertexType.DUMMY) \
                    != (self.vertices[self.edges[uid].enduid].mode
                        == VertexType.DUMMY):

                basisMatrices[uid] = np.zeros([6,6])
                basisMatrices[uid][3:6, 3:6] = globalYawEffectBasis(
                    R.from_quat(self.edges[uid].change[3:7]))

            else:
                basisMatrices[uid] = np.eye(6)

        self.basisMatrices = basisMatrices

    def generateUnoptimizedGraph(self):
        self.unoptimizedGraph = graph2Optimizer(self)
        return self.unoptimizedGraph

    def optimizeGraph(self):
        self.optimizedGraph = graph2Optimizer(self)

        initStatus = self.optimizedGraph.initialize_optimization()
        runStatus = self.optimizedGraph.optimize(256)

        self.g2oStatus =  initStatus and runStatus

    def generateMaximizationParams(self):
        errors = np.array([])
        observations = np.reshape([], [0, 18])
        optimizedEdges = {edge.id(): edge for edge in list(
            self.optimizedGraph.edges())}

        for uid in self.edges:
            edge = self.edges[uid]
            startMode = self.vertices[edge.startuid].mode
            endMode = self.vertices[edge.enduid].mode

            if endMode != VertexType.WAYPOINT:
                errors = np.hstack(
                    [errors, self.basisMatrices[uid].T.dot(
                        optimizedEdges[uid].error())])

            if startMode == VertexType.ODOMETRY:
                if endMode == VertexType.ODOMETRY:
                    observations = np.vstack([observations, np.eye(6, 18)])
                elif endMode == VertexType.TAG:
                    observations = np.vstack([observations, np.eye(6, 18, 6)])
                elif endMode == VertexType.DUMMY:
                    observations = np.vstack([observations, np.eye(6, 18, 12)])
                elif endMode == VertexType.WAYPOINT:
                    pass
                else:
                    raise Exception("Unspecified handling for edge of start"
                                    " type {} and end type {}"
                                    .format(startMode, endMode))

            else:
                raise Exception("Unspecified handling for edge of start type"
                                " {} and end type {}"
                                .format(startMode, endMode))

        self.errors = errors
        self.observations = observations
        return errors, observations

    def tuneWeights(self):
        results = maxweights(self.observations, self.errors,
                             np.zeros(self.observations.shape[1]))
        self.maximizationSuccess = results.success
        self.weights = results.x
        self.maximizationResults = results

        for uid in self.edges:
            edge = self.edges[uid]
            startMode = self.vertices[edge.startuid].mode
            endMode = self.vertices[edge.enduid].mode

            if startMode == VertexType.ODOMETRY:
                if endMode == VertexType.ODOMETRY:
                    self.edges[uid].information = np.diag(np.sqrt(np.exp(-self.weights[:6])))
                if endMode == VertexType.TAG:
                    self.edges[uid].information = np.diag(np.sqrt(np.exp(-self.weights[6:12])))
                if endMode == VertexType.DUMMY:
                    basis = self.basisMatrices[uid][3:6, 3:6]
                    cov = np.diag(np.exp(-self.weights[15:18] / 2))
                    information = basis.dot(cov).dot(basis.T)
                    template = np.zeros([6,6])
                    template[3:6, 3:6] = information
                    self.edges[uid].information = template


        return results

    def emOnce(self):
        self.generateUnoptimizedGraph()
        self.optimizeGraph()
        self.generateMaximizationParams()
        self.tuneWeights()

    def em(self, maxIter=10, tol=1):
        prevChi2 = self.optimizedGraph.chi2()
        newChi2 = 0
        i = 0
        while i < maxIter:
            self.emOnce()
            newChi2 = self.optimizedGraph.chi2()

            if np.abs(newChi2 - prevChi2) < tol:
                return i

            prevChi2 = newChi2
            i += 1

        if np.abs(newChi2 - prevChi2) < tol:
            return i

        return i

    def plotMap(self):
        unoptimized = optimizer2map(self.vertices, self.unoptimizedGraph)
        optimized = optimizer2map(self.vertices, self.optimizedGraph)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        tagMarker = '^'
        waypointMarker = 's'
        locationMarker = '.'

        ax.set_title(r'Map ($\chi^2 = {:.4g}$)'.format(
            self.optimizedGraph.chi2()))

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

    def plotErrors(self):
        tabulatedErrors = self.errors.reshape(-1, 6)
        fig, axs = plt.subplots(3, 2)

        for i, (title, ax) in enumerate(zip(["x", "y", "z", "qx", "qy", "qz"], itertools.chain.from_iterable(axs))):
            ax.set_title("{} Error".format(title))
            ax.hist(tabulatedErrors[:, i])

        return fig

    def connectedComponents(self):
        groups = []
        for i in self.edges:
            uids = {self.edges[i].startuid, self.edges[i].enduid}
            membership = []
            for j, group in enumerate(groups):
                if len(group[0] & uids) > 0:
                    membership.append(j)

            newGroup = (set.union(uids, *[groups[k][0] for k in membership]),
                        [i] + list(itertools.chain.from_iterable
                                   ([groups[k][1] for k in membership])))

            membership.reverse()

            for i in membership:
                del groups[i]

            groups.append(newGroup)

        return [Graph(vertices={k: self.vertices[k] for k in group[0]},
                      edges={k: self.edges[k] for k in group[1]})
                for group in groups]

    def odometryGraph(self):
        odometryEdges = {}
        odometryVertices = {}
        for i in self.edges:
            if self.vertices[self.edges[i].startuid].mode \
               == VertexType.ODOMETRY \
               and self.vertices[self.edges[i].enduid].mode \
               == VertexType.ODOMETRY:

                odometryVertices[self.edges[i].startuid] \
                    = self.vertices[self.edges[i].startuid]
                odometryVertices[self.edges[i].enduid] \
                    = self.vertices[self.edges[i].enduid]
                odometryEdges[i] = self.edges[i]

        return Graph(vertices=odometryVertices, edges=odometryEdges)
