"""Store a map in graph form and optimize it using EM.
"""

from enum import Enum
import g2o
import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.optimize import OptimizeResult
from graph_utils import pose_to_isometry, pose_to_se3quat, global_yaw_effect_basis, isometry_to_pose, \
    measurement_to_matrix

from maximization_model import maxweights


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
                the diagonal of a matrix that pre-multiplies the edge
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

    def __init__(self, vertices, edges, weights=np.zeros(18), gravity_axis='z', is_sparse_bundle_adjustment=False,
                 use_huber=False, huber_delta=None, damping_status=False):
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
        self.unoptimized_graph = self.graph_to_optimizer()

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
        self.optimized_graph = self.graph_to_optimizer()

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

    def connected_components(self):
        """Return a list of graphs representing connecting components of
        the input graph.

        If the graph is connected, there should only be one element in the
        output.

        Returns: A list of :class: Graph containing the connected
            components.
        """
        groups = []
        for uid in self.edges:
            edge = self.edges[uid]
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
        # TODO: copy over other information from the graph
        return [Graph(vertices={k: self.vertices[k] for k in group[0]},
                      edges={k: self.edges[k] for k in group[1]})
                for group in groups]

    def graph_to_optimizer(self):
        """Convert a :class: graph to a :class: g2o.SparseOptimizer.  Only the edges and vertices fields need to be
        filled out.

        Returns:
            A :class: g2o.SparseOptimizer that can be optimized via its
            optimize class method.
        """
        optimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())))

        if self.is_sparse_bundle_adjustment:
            for i in self.vertices:
                if self.vertices[i].mode == VertexType.TAGPOINT:
                    vertex = g2o.VertexSBAPointXYZ()
                    vertex.set_estimate(self.vertices[i].estimate[:3])
                else:
                    vertex = g2o.VertexSE3Expmap()
                    vertex.set_estimate(pose_to_se3quat(self.vertices[i].estimate))
                vertex.set_id(i)
                vertex.set_fixed(self.vertices[i].fixed)
                optimizer.add_vertex(vertex)
            cam_idx = 0
            for i in self.edges:
                if self.edges[i].corner_ids is None:
                    edge = g2o.EdgeSE3Expmap()
                    for j, k in enumerate([self.edges[i].startuid,
                                           self.edges[i].enduid]):
                        edge.set_vertex(j, optimizer.vertex(k))
                        edge.set_measurement(pose_to_se3quat(self.edges[i].measurement))
                        edge.set_information(self.edges[i].information)
                    optimizer.add_edge(edge)
                else:
                    # Note: we only use the focal length in the x direction since: (a) that's all that g2o supports and
                    # (b) it is always the same in ARKit (at least currently)
                    cam = g2o.CameraParameters(self.edges[i].camera_intrinsics[0],
                                               self.edges[i].camera_intrinsics[2:], 0)
                    cam.set_id(cam_idx)
                    optimizer.add_parameter(cam)
                    for corner_idx, corner_id in enumerate(self.edges[i].corner_ids):
                        edge = g2o.EdgeProjectPSI2UV()
                        edge.resize(3)
                        edge.set_vertex(0, optimizer.vertex(corner_id))
                        edge.set_vertex(1, optimizer.vertex(self.edges[i].startuid))
                        edge.set_vertex(2, optimizer.vertex(self.edges[i].enduid))
                        edge.set_information(self.edges[i].information)
                        edge.set_measurement(self.edges[i].measurement[corner_idx * 2:corner_idx * 2 + 2])
                        edge.set_parameter_id(0, cam_idx)
                        if self.use_huber:
                            edge.set_robust_kernel(g2o.RobustKernelHuber(self.huber_delta))
                        optimizer.add_edge(edge)
                    cam_idx += 1
        else:
            for i in self.vertices:
                vertex = g2o.VertexSE3()
                vertex.set_id(i)
                vertex.set_estimate(pose_to_isometry(self.vertices[i].estimate))
                vertex.set_fixed(self.vertices[i].fixed)
                optimizer.add_vertex(vertex)

            for i in self.edges:
                edge = g2o.EdgeSE3()

                for j, k in enumerate([self.edges[i].startuid,
                                       self.edges[i].enduid]):
                    edge.set_vertex(j, optimizer.vertex(k))

                edge.set_measurement(pose_to_isometry(self.edges[i].measurement))
                edge.set_information(self.edges[i].information)
                edge.set_id(i)

                optimizer.add_edge(edge)
        return optimizer

    def integrate_path(self, edgeuids, initial=np.array([0, 0, 0, 0, 0, 0, 1])):
        poses = [initial]
        for edgeuid in edgeuids:
            old_pose = measurement_to_matrix(poses[-1])
            transform = measurement_to_matrix(self.edges[edgeuid].measurement)
            new_pose = old_pose.dot(transform)
            translation = new_pose[:3, 3]
            rotation = R.from_matrix(new_pose[:3, :3]).as_quat()
            poses.append(np.concatenate([translation, rotation]))
        return np.array(poses)

    def get_tags_all_position_estimate(self):
        tags = np.reshape([], [0, 8])  # [x, y, z, qx, qy, qz, 1, id]
        for edgeuid in self.edges:
            edge = self.edges[edgeuid]
            if self.vertices[edge.startuid].mode == VertexType.ODOMETRY and self.vertices[
                edge.enduid].mode == VertexType.TAG:
                odom_transform = measurement_to_matrix(
                    self.vertices[edge.startuid].estimate)
                edge_transform = measurement_to_matrix(edge.measurement)

                tag_transform = odom_transform.dot(edge_transform)
                tag_translation = tag_transform[:3, 3]
                tag_rotation = R.from_matrix(tag_transform[:3, :3]).as_quat()
                tag_pose = np.concatenate(
                    [tag_translation, tag_rotation, [edge.enduid]])
                tags = np.vstack([tags, tag_pose])
        return tags

    def get_subgraph(self, start_vertex_uid, end_vertex_uid):
        edges = self.ordered_odometry_edges()
        start_found = False
        ret_graph = Graph({}, {})
        for i, edgeuid in enumerate(edges[0]):
            edge = self.edges[edgeuid]
            if edge.startuid == start_vertex_uid:
                start_found = True

            if start_found:
                ret_graph.vertices[edge.enduid] = self.vertices[edge.enduid]
                ret_graph.vertices[edge.startuid] = self.vertices[edge.startuid]
                ret_graph.edges[edgeuid] = edge

            if edge.enduid == end_vertex_uid:
                break

        # Find tags and edges connecting to the found vertices
        for edgeuid in self.edges:
            edge = self.edges[edgeuid]
            if self.vertices[edge.startuid].mode == VertexType.TAG and edge.enduid in ret_graph.vertices:
                ret_graph.edges[edgeuid] = edge
                ret_graph.vertices[edge.startuid] = self.vertices[edge.startuid]

            if self.vertices[edge.enduid].mode == VertexType.TAG and edge.startuid in ret_graph.vertices:
                ret_graph.edges[edgeuid] = edge
                ret_graph.vertices[edge.enduid] = self.vertices[edge.enduid]

        return ret_graph

    def ordered_odometry_edges(self):
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

        for uid in self.edges:
            edge = self.edges[uid]
            if self.vertices[edge.startuid].mode != VertexType.ODOMETRY \
                    or self.vertices[edge.enduid].mode != VertexType.ODOMETRY:
                continue
            start_found = end_found = False
            start_found_idx = end_found_idx = 0

            for i in range(len(segments) - 1, -1, -1):
                current_start_found = \
                    edge.startuid == self.edges[segments[i][-1]].enduid
                current_end_found = \
                    edge.enduid == self.edges[segments[i][0]].startuid

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
