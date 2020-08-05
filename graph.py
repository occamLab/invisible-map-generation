"""Store a map in graph form and optimize it using EM.
"""

from enum import Enum
import g2o
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import OptimizeResult

from maximization_model import maxweights


def pose_to_isometry(pose):
    """Convert a pose vector to a :class: g2o.Isometry3d instance.

    Args:
        pose: A 7 element 1-d numpy array encoding x, y, z, qx, qy,
        qz, and qw respectively.
    Returns:
        A :class: g2o.Isometry3d instance encoding the same
        information as the input pose.
    """
    return g2o.Isometry3d(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])

def pose_to_se3quat(pose):
    """Convert a pose vector to a :class: g2o.Isometry3d instance.

    Args:
        pose: A 7 element 1-d numpy array encoding x, y, z, qx, qy,
        qz, and qw respectively.
    Returns:
        A :class: g2o.Isometry3d instance encoding the same
        information as the input pose.
    """
    return g2o.SE3Quat(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])



def isometry_to_pose(isometry):
    """Convert a :class: g2o.Isometry3d to a vector containing a pose.

    Args:
        isometry: A :class: g2o.Isometry3d instance.
    Returns:
        A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and
        qw respectively.
    """
    return np.concatenate(
        [isometry.translation(), isometry.rotation().coeffs()])


def graph_to_optimizer(graph, damping_status=False):
    """Convert a :class: graph to a :class: g2o.SparseOptimizer.

    Args:
        graph: A :class: graph to be converted.
            Only the edges and vertices fields need to be filled out.
    Returns:
        A :class: g2o.SparseOptimizer that can be optimized via its
        optimize class method.

    """
    optimizer = g2o.SparseOptimizer()
    optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
        g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())))

    if graph.is_sparse_bundle_adjustment:
        for i in graph.vertices:
            if graph.vertices[i].mode == VertexType.TAGPOINT:
                vertex = g2o.VertexSBAPointXYZ()
                vertex.set_estimate(graph.vertices[i].estimate[:3])
            else:
                vertex = g2o.VertexSE3Expmap()
                vertex.set_estimate(pose_to_se3quat(graph.vertices[i].estimate))
            vertex.set_id(i)
            vertex.set_fixed(graph.vertices[i].fixed)
            optimizer.add_vertex(vertex)
        cam_idx = 0
        for i in graph.edges:
            if graph.edges[i].corner_ids is None:
                edge = g2o.EdgeSE3Expmap()
                for j, k in enumerate([graph.edges[i].startuid,
                                       graph.edges[i].enduid]):
                    edge.set_vertex(j, optimizer.vertex(k))
                    edge.set_measurement(pose_to_se3quat(graph.edges[i].measurement))
                    edge.set_information(graph.edges[i].information)
                optimizer.add_edge(edge)
            else:
                # Note: we only use the focal length in the x direction since: (a) that's all that g2o supports and (b) it is always the same in ARKit (at least currently)
                cam = g2o.CameraParameters(graph.edges[i].camera_intrinsics[0], graph.edges[i].camera_intrinsics[2:], 0)
                cam.set_id(cam_idx)
                optimizer.add_parameter(cam)
                for corner_idx, corner_id in enumerate(graph.edges[i].corner_ids):
                    edge = g2o.EdgeProjectPSI2UV()
                    edge.resize(3)
                    edge.set_vertex(0, optimizer.vertex(corner_id))
                    edge.set_vertex(1, optimizer.vertex(graph.edges[i].startuid))
                    edge.set_vertex(2, optimizer.vertex(graph.edges[i].enduid))
                    edge.set_information(graph.edges[i].information)
                    edge.set_measurement(graph.edges[i].measurement[corner_idx*2:corner_idx*2+2])
                    edge.set_parameter_id(0, cam_idx)
                    if graph.use_huber:
                        edge.set_robust_kernel(g2o.RobustKernelHuber(graph.huber_delta))
                    optimizer.add_edge(edge)
                cam_idx += 1
    else:
        for i in graph.vertices:
            vertex = g2o.VertexSE3()
            vertex.set_id(i)
            vertex.set_estimate(pose_to_isometry(graph.vertices[i].estimate))
            vertex.set_fixed(graph.vertices[i].fixed)
            optimizer.add_vertex(vertex)

        for i in graph.edges:
            edge = g2o.EdgeSE3()

            for j, k in enumerate([graph.edges[i].startuid,
                                   graph.edges[i].enduid]):
                edge.set_vertex(j, optimizer.vertex(k))

            edge.set_measurement(pose_to_isometry(graph.edges[i].measurement))
            edge.set_information(graph.edges[i].information)
            edge.set_id(i)

            optimizer.add_edge(edge)
    return optimizer


def global_yaw_effect_basis(rotation, gravity_axis='z'):
    """Form a basis which describes the effect of a change in global
    yaw on a local measurement's qx, qy, and qz.

    Since the accelerometer measures gravitational acceleration, it
    can accurately measure the global z-azis but its measurement of
    the orthogonal axis are less reliable.

    Args:
        rotation: A :class: scipy.spatial.transform.Rotation encoding
        a local rotation.

    Returns:
        A 3x3 numpy array where the columns are the new basis.
    """
    rotation1 = R.from_euler(gravity_axis, 0.05) * rotation
    change = rotation1.as_quat()[:3] - rotation.as_quat()[:3]
    return np.linalg.svd(change[:, np.newaxis])[0]


class VertexType(Enum):
    """An enumeration containing the vertex types ODOMETRY, TAG,
    DUMMY, and WAYPOINT.
    """
    ODOMETRY = 0
    TAG = 1
    TAGPOINT = 2
    DUMMY = 3
    WAYPOINT = 4


class Vertex:
    """A class to contain a vertex of the optimization graph.

    It contains the :class: VertexType of the vertex as well as the
    pose.
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

    It encodes UIDs for the start and end vertices, the measurement,
    and the information matrix.
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

        The arguments are a startuid, enduid, an information matrix
        represented by a 6x6 numpy array, and a measurement.

        Args:
            startuid: The UID of the starting vertex.
                This can be any hashable such as an int.
            enduid: The UID of the ending vertex.
                This can be any hashable such as an int.
            corner_ids: an array of UIDs for each of the tag corner vertices.
                This only applies to edges to a tag
            information: A 6x6 numpy array encoding measurement
                information. The rows and columns encode x, y, z, qx,
                qy, and qz information.
            information_prescaling: A 6 element numpy array encoding
                the diagonal of a matrix that premultiplies the edge
                information matrix specified by the weights.  If None
                is past, then the 6 element vector is assumed to be all
                ones
            camera_intrinsics: [fx, fy, cx, cy] (only applies to tag edges)
            measurement: A 7 element numpy array encoding the measured
                transform from the start vertex to the end vertex in
                the start vertex's coordinate frame.
                The format is [x, y, z, qx, qy, qz, qw].
        """
        self.startuid = startuid
        self.enduid = enduid
        self.corner_ids = corner_ids
        self.information = information
        self.information_prescaling = information_prescaling
        self.camera_intrinsics = camera_intrinsics
        self.measurement = measurement


class Graph:
    """A class for the graph encoding a map with class methods to
    optimize it.
    """

    def __init__(self, vertices, edges, weights=np.zeros(18), gravity_axis='z', is_sparse_bundle_adjustment=False, use_huber=False, huber_delta=None, damping_status=False):
        """The graph class.
        The graph contains a dictionary of vertices and edges, the
        keys being UIDs such as ints.
        The start and end UIDs in each edge refer to the vertices in
        the vertices dictionary.

        Args:
            vertices: A dictionary of vertices indexed by UIDs.
                The UID-vertices associations are referred to by the
                startuid and enduid fields of the :class: Edge class.
            edges: A dictionary of edges indexed by UIDs.

        Kwargs:
            weights: An initial guess for what the weights of the model are.
                The weights correspond to x, y, z, qx, qy, and qz
                measurements for odometry edges, tag edges, and dummy
                edges and has 18 elements [odometry x, odometry y,
                ..., dummy qz]
                The weights are related to variance by variance = exp(w).
        """

        self.is_sparse_bundle_adjustment = is_sparse_bundle_adjustment
        self.edges = edges
        self.original_vertices = vertices
        self.vertices = vertices
        self.weights = weights
        self.gravity_axis = gravity_axis
        self.generate_basis_matrices()

        self.g2o_status = -1
        self.maximization_success_status = False
        self.errors = np.array([])
        self.observations = np.reshape([], [0, self.weights.size])
        self.maximization_success = False
        self.maximization_results = OptimizeResult

        self.unoptimized_graph = None
        self.optimized_graph = None
        self.damping_status = damping_status
        self.use_huber = use_huber
        self.huber_delta = huber_delta
        self.update_edges()

    def generate_basis_matrices(self):
        """Generate basis matrices used to show how a change in global
        yaw changes the values of a local measurement.

        This is used for dummy edges.
        For other edge types, the basis is simply the identity matrix.
        """
        basis_matrices = {}

        for uid in self.edges:
            if (self.vertices[self.edges[uid].startuid].mode
                    == VertexType.DUMMY) \
                    != (self.vertices[self.edges[uid].enduid].mode
                        == VertexType.DUMMY):

                basis_matrices[uid] = np.eye(6)
                if not self.is_sparse_bundle_adjustment:
                    basis_matrices[uid][3:6, 3:6] = global_yaw_effect_basis(
                        R.from_quat(self.vertices[self.edges[uid].enduid].estimate[3:7]), self.gravity_axis)
            else:
                basis_matrices[uid] = np.eye(6)

        self.basis_matrices = basis_matrices

    def generate_unoptimized_graph(self):
        """Generate the unoptimized g2o graph from the current vertex
        and edge assignments.

        This can be optimized using :func: optimize_graph.
        """
        self.unoptimized_graph = graph_to_optimizer(self)

    def check_optimized_edges(self, g):
        total_chi2 = 0.0
        for edge in g.edges():
            if type(edge) == g2o.EdgeProjectPSI2UV:
                cam = edge.parameter(0)
                error = edge.measurement() - cam.cam_map(edge.vertex(1).estimate()*edge.vertex(2).estimate().inverse()*edge.vertex(0).estimate())
                error_chi2 = error.dot(edge.information()).dot(error)
            elif type(edge) == g2o.EdgeSE3Expmap:
                error = edge.vertex(1).estimate().inverse() * edge.measurement() * edge.vertex(0).estimate()
                error_chi2 = error.log().T.dot(edge.information()).dot(error.log())
            elif type(edge) == g2o.EdgeSE3:
                delta = edge.measurement().inverse() * edge.vertex(0).estimate().inverse() * edge.vertex(1).estimate()
                error = np.hstack((delta.translation() ,delta.orientation().coeffs()[:-1]))
                error_chi2 = error.dot(edge.information()).dot(error)
            total_chi2 += error_chi2
        print("total chi2", total_chi2)
        return total_chi2

    def optimize_graph(self):
        """Optimize the graph using g2o.

        It sets the g2o_status attribute to the g2o success output.
        """
        self.optimized_graph = graph_to_optimizer(self)

        self.optimized_graph.initialize_optimization()
        run_status = self.optimized_graph.optimize(1024)
        print("checking unoptimized edges")
        self.check_optimized_edges(self.unoptimized_graph)
        print("checking optimized edges")
        self.check_optimized_edges(self.optimized_graph)
        self.g2o_status = run_status

    def generate_maximization_params(self):
        """Generate the arrays to be processed by the maximization model.

        It sets the error field to an array of errors, as well as a
        2-d array populated by 1-hot 18 element observation vectors
        indicating the type of measurement.
        The meaning of the position of the one in the observation
        vector corresponds to the layout of the weights vector.
        """
        errors = np.array([])
        observations = np.reshape([], [0, 18])
        optimized_edges = {edge.id(): edge for edge in list(
            self.optimized_graph.edges())}

        for uid in self.edges:
            edge = self.edges[uid]
            start_mode = self.vertices[edge.startuid].mode
            end_mode = self.vertices[edge.enduid].mode

            if end_mode != VertexType.WAYPOINT:
                errors = np.hstack(
                    [errors, self.basis_matrices[uid].T.dot(
                        optimized_edges[uid].error())])

            if start_mode == VertexType.ODOMETRY:
                if end_mode == VertexType.ODOMETRY:
                    observations = np.vstack([observations, np.eye(6, 18)])
                elif end_mode == VertexType.TAG:
                    observations = np.vstack([observations, np.eye(6, 18, 6)])
                elif end_mode == VertexType.DUMMY:
                    observations = np.vstack([observations, np.eye(6, 18, 12)])
                elif end_mode == VertexType.WAYPOINT:
                    pass
                else:
                    raise Exception("Unspecified handling for edge of start"
                                    " type {} and end type {}"
                                    .format(start_mode, end_mode))

            else:
                raise Exception("Unspecified handling for edge of start type"
                                " {} and end type {}"
                                .format(start_mode, end_mode))

        self.errors = errors
        self.observations = observations
        return errors, observations

    def tune_weights(self):
        """Tune the weights to maximize the likelihood of the errors
        found between the unoptimized and optimized graphs.
        """
        results = maxweights(self.observations, self.errors, self.weights)
        self.maximization_success = results.success
        self.weights = results.x
        self.maximization_results = results
        self.update_edges()
        return results

    def update_edges(self):
        for uid in self.edges:
            edge = self.edges[uid]
            start_mode = self.vertices[edge.startuid].mode
            end_mode = self.vertices[edge.enduid].mode
            if start_mode == VertexType.ODOMETRY:
                if end_mode == VertexType.ODOMETRY:
                    self.edges[uid].information = np.diag(
                        np.exp(-self.weights[:6]))
                elif end_mode == VertexType.TAG:
                    if self.is_sparse_bundle_adjustment:
                        self.edges[uid].information = np.diag(
                            np.exp(-self.weights[6:8]))
                    else:
                        self.edges[uid].information = np.diag(
                            np.exp(-self.weights[6:12]))
                elif end_mode == VertexType.DUMMY:
                    # TODO: this basis is not very pure and results in weight on each dimension of the quaternion (seems to work though)
                    basis = self.basis_matrices[uid][3:6, 3:6]
                    cov = np.diag(np.exp(-self.weights[15:18]))
                    information = basis.dot(cov).dot(basis.T)
                    template = np.zeros([6, 6])
                    if self.is_sparse_bundle_adjustment:
                        template[:3, :3] = information
                    else:
                        template[3:6, 3:6] = information
                    if self.damping_status:
                        self.edges[uid].information = template
                    else:
                        self.edges[uid].information = np.zeros_like(template)
                elif end_mode == VertexType.WAYPOINT:
                    # TODO: not sure what this should be
                    self.edges[uid].information = np.eye(6,6)
                else:
                    raise Exception(
                        'Edge of end type {} not recognized.'.format(end_mode))
                if self.edges[uid].information_prescaling is not None:
                    prescaling_matrix = self.edges[uid].information_prescaling
                    if prescaling_matrix.ndim == 1:
                        prescaling_matrix = np.diag(prescaling_matrix)
                    self.edges[uid].information = prescaling_matrix*self.edges[uid].information
            else:
                raise Exception(
                    'Edge of start type {} not recognized.'.format(start_mode))

    def expectation_maximization_once(self):
        """Run one cycle of expectation maximization.

        It generates an unoptimized graph from current vertex
        estimates and edge measurements and importances, and optimizes
        the graph.  Using the errors, it tunes the weights so that the
        variances maximize the likelihood of each error by type.
        """
        self.generate_unoptimized_graph()
        self.optimize_graph()
        self.update_vertices()
        self.generate_maximization_params()
        self.tune_weights()

    def expectation_maximization(self, maxiter=10, tol=1):
        """Run many iterations of expectation maximization.

        Kwargs:
            maxiter (int): The maximum amount of iterations.
            tol (float): The maximum magnitude of the change in weight
                vectors that will signal the end of the cycle.
        """
        previous_weights = self.weights
        new_weights = self.weights

        i = 0
        while i < maxiter:
            self.expectation_maximization_once()
            new_weights = self.weights

            if np.linalg.norm(new_weights - previous_weights) < tol:
                return i

            previous_weights = new_weights
            i += 1

        return i

    def update_vertices(self):
        """Update the initial vertices elements with the optimized graph values.
        """
        for uid in self.optimized_graph.vertices():
            if self.is_sparse_bundle_adjustment:
                if type(self.optimized_graph.vertex(uid).estimate()) == np.ndarray:
                    self.vertices[uid].estimate = self.optimized_graph.vertex(uid).estimate()
                else:
                    self.vertices[uid].estimate = self.optimized_graph.vertex(uid).estimate().to_vector()
            else:
                self.vertices[uid].estimate = isometry_to_pose(
                    self.optimized_graph.vertices()[uid].estimate())
