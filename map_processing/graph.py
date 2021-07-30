"""Store a map in graph form and optimize it using EM.
"""

from __future__ import annotations

import copy
import math
from typing import Union, Set, Tuple

import g2o
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation as Rot

import map_processing.graph_utils as graph_utils
from expectation_maximization.maximization_model import maxweights
from map_processing.graph_utils import pose_to_isometry, pose_to_se3quat, global_yaw_effect_basis, isometry_to_pose, \
    measurement_to_matrix
from map_processing.graph_vertex_edge_classes import *


class Graph:
    """A class for the graph encoding a map with class methods to optimize it.
    """

    def __init__(self, vertices: Dict[int, Vertex],
                 edges: Dict[int, Edge],
                 weights: Dict[str, np.ndarray] = {'odometry': np.zeros(6),
                                                   'tag_sba': np.zeros(2),
                                                   'tag': np.zeros(6),
                                                   'dummy': np.zeros(3)},
                 gravity_axis="y",
                 is_sparse_bundle_adjustment: bool = False,
                 use_huber: bool = False,
                 huber_delta=None,
                 damping_status: bool = False):
        """The graph class

        The graph contains a dictionary of vertices and edges, the keys being UIDs such as ints. The start and end UIDs
        in each edge refer to the vertices in the vertices dictionary.

        Args:
            vertices: A dictionary of vertices indexed by UIDs. The UID-vertices associations are referred to by the
             startuid and enduid fields of the :class: Edge  class.
            edges: A dictionary of edges indexed by UIDs.

        Kwargs:
            weights: An initial guess for what the weights of the model are. The weights correspond to x, y, z, qx, qy,
             and qz measurements for odometry edges, tag edges, and dummy edges and has 18 elements [odometry x,
             odometry y, ..., dummy qz]. The weights are related to variance by variance = exp(w).
        """
        self.edges: Dict[int, Edge] = copy.deepcopy(edges)
        self.vertices: Dict[int, Vertex] = copy.deepcopy(vertices)
        self.original_vertices = copy.deepcopy(vertices)
        self.huber_delta: bool = copy.deepcopy(huber_delta)
        self.gravity_axis: str = copy.deepcopy(gravity_axis)
        self.is_sparse_bundle_adjustment: bool = is_sparse_bundle_adjustment
        self.damping_status: bool = damping_status
        self.use_huber: bool = use_huber

        self._weights: Dict[str, np.ndarray] = {}
        self.set_weights(copy.deepcopy(weights))
        
        self._verts_to_edges: Dict[int, Set[int]] = {}
        self._generate_verts_to_edges_mapping()
        self._basis_matrices: Dict[int, np.ndarray] = {}
        self._generate_basis_matrices()

        self.g2o_status = -1
        self.maximization_success_status = False
        self.errors = np.array([])
        self.observations = np.reshape([], [0, len(weights) * 6]) # TODO ensure the shape is right
        self.maximization_success: bool = False
        self.maximization_results = OptimizeResult
        self.unoptimized_graph: Union[g2o.SparseOptimizer, None] = None
        self.optimized_graph: Union[g2o.SparseOptimizer, None] = None
        self.update_edges()

        # This is populated in graph_to_optimizer and is currently no updated anywhere else
        self.our_edges_to_g2o_edges: Dict[int, Union[g2o.EdgeProjectPSI2UV, g2o.EdgeSE3Expmap, g2o.EdgeSE3]] = {}

    # -- Optimization-related methods --

    def generate_unoptimized_graph(self) -> None:
        """Generate the unoptimized g2o graph from the current vertex and edge assignments.

        This can be optimized using :func: optimize_graph.
        """
        self.unoptimized_graph = self.graph_to_optimizer()

    def _generate_verts_to_edges_mapping(self) -> None:
        """Populates the `_verts_to_edges` attribute such that it maps vertex UIDs to incident edge UIDs (regardless
        of whether the edge is incoming or outgoing).
        """
        for edge_uid in self.edges:
            edge = self.edges[edge_uid]
            for vertex_uid in [edge.startuid, edge.enduid]:
                if self._verts_to_edges.__contains__(vertex_uid):
                    self._verts_to_edges[vertex_uid].add(edge_uid)
                else:
                    self._verts_to_edges[vertex_uid] = {edge_uid, }

    def get_chi2_by_edge_type(self, graph: g2o.SparseOptimizer, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Iterates through the edges and calculates the chi2 of each, sorting them into categories based on the end vertex

        Args:
            graph: a g2o.SparseOptimizer object
            verbose (bool): Boolean for whether or not to print the chi2 values

        Returns:
            A dict mapping 'odometry', 'tag', and 'dummy' to the chi2s corresponding to that edge type
        """
        chi2s = {
            'odometry': {
                'sum': 0.0,
                'edges': 0
            },
            'tag': {
                'sum': 0.0,
                'edges': 0
            },
            'dummy': {
                'sum': 0.0,
                'edges': 0
            },
        }
        for edge in graph.edges():
            end_mode = self.vertices[self.edges[edge.id()].enduid].mode
            if end_mode == VertexType.ODOMETRY:
                chi2s['odometry']['sum'] += Graph.get_chi2_of_edge(edge)
                chi2s['odometry']['edges'] += 1
            elif end_mode == VertexType.TAG or end_mode == VertexType.TAGPOINT:
                chi2s['tag']['sum'] += Graph.get_chi2_of_edge(edge)
                chi2s['tag']['edges'] += 1
            elif end_mode == VertexType.DUMMY:
                chi2s['dummy']['sum'] += Graph.get_chi2_of_edge(edge)
                chi2s['dummy']['edges'] += 1
            start_mode = self.vertices[self.edges[edge.id()].startuid].mode
            if start_mode != VertexType.ODOMETRY:
                raise Exception(f'Original is not odometry. Edge type: {type(edge)}. Start: {start_mode}. End: {end_mode}')
        for edge_type in chi2s:
            chi2s[edge_type]['average'] = chi2s[edge_type]['sum'] / chi2s[edge_type]['edges']
        if verbose:
            print(chi2s)
        return chi2s

    @staticmethod
    def check_optimized_edges(graph: g2o.SparseOptimizer, verbose: bool = True) -> float:
        """Iterates through edges in the g2o sparse optimizer object and sums the chi2 values for all of the edges.

        Args:
            graph: A g2o.SparseOptimizer object
            verbose (bool): Boolean for whether or not to print the total chi2 value

        Returns:
            Sum of the chi2 values associated with each edge
        """
        total_chi2 = 0.0
        for edge in graph.edges():
            total_chi2 += Graph.get_chi2_of_edge(edge)

        if verbose:
            print("Total chi2:", total_chi2)

        return total_chi2

    @staticmethod
    def get_chi2_of_edge(edge: Union[g2o.EdgeProjectPSI2UV, g2o.EdgeSE3Expmap, g2o.EdgeSE3]) -> float:
        """Computes the chi2 value associated with the provided edge

        Arguments:
            edge (Union[g2o.EdgeProjectPSI2UV, g2o.EdgeSE3Expmap, g2o.EdgeSE3]): A g2o edge

        Returns:
            Chi2 value associated with the provided edge

        Raises:
            Exception if an edge is encountered that is not handled (handled edges are g2o.EdgeProjectPSI2UV,
             g2o.EdgeSE3Expmap, and g2o.EdgeSE3)
        """
        if isinstance(edge, g2o.EdgeProjectPSI2UV):
            # if np.abs(edge.measurement()[1] - 939.6618652) < 0.00001:
            #     print('test')
            cam = edge.parameter(0)
            camera_coords = edge.vertex(1).estimate() * edge.vertex(2).estimate().inverse() * edge.vertex(0).estimate()
            pixel_coords = cam.cam_map(camera_coords)
            error = edge.measurement() - cam.cam_map(
                edge.vertex(1).estimate() * edge.vertex(2).estimate().inverse() * edge.vertex(0).estimate())
            # if edge.vertex(2).id() == 0:
            #     print(f"Camera: {camera_coords}")
            #     print(f"Pixel: {pixel_coords}")
            #     print(f"Edge: {edge.measurement()}")
            #     print(f"Error: {error}")

            return error.dot(edge.information()).dot(error)
        elif isinstance(edge, g2o.EdgeSE3Expmap):
            error = edge.vertex(1).estimate().inverse() * edge.measurement() * edge.vertex(0).estimate()
            chi2 = error.log().T.dot(edge.information()).dot(error.log())
            if math.isnan(chi2):
                # print(f'vertex 0 estimate:\n{edge.vertex(0).estimate().matrix()}\nfull:\n{edge.vertex(0).estimate().adj()}')
                # print(f'vertex 1 estimate:\n{edge.vertex(1).estimate().matrix()}\nfull:\n{edge.vertex(1).estimate().adj()}')
                # print(f'measurement:\n{edge.measurement().matrix()}\nfull:\n{edge.measurement().adj()}')
                # print(f'calc. error:\n{error.matrix()}\nfull:{error.adj()}')
                # print(f'omega:\n{edge.information()}')
                # if weights is not None:
                #     print(f'weights:\n{weights}')
                raise Exception('chi2 is NaN for an edge of type EdgeSE3Expmap')
            return chi2
        elif isinstance(edge, g2o.EdgeSE3):
            delta = edge.measurement().inverse() * edge.vertex(0).estimate().inverse() * edge.vertex(1).estimate()
            error = np.hstack((delta.translation(), delta.orientation().coeffs()[:-1]))
            return error.dot(edge.information()).dot(error)
        else:
            raise Exception("Unhandled edge type for chi2 calculation")

    def map_odom_to_adj_chi2(self, vertex_uid: int) -> Tuple[float, int]:
        """Computes odometry-adjacent chi2 value

        Arguments:
            vertex_uid (int): Vertex integer corresponding to an odometry node

        Returns:
            Tuple containing two elements:
            - Float that is the sum of the chi2 values of the two edges (as calculated through the `get_chi2_of_edge`
              static method) that are incident to both the specified odometry node and two other odometry nodes. If
              there is only one such incident edge, then only that edge's chi2 value is returned.
            - Integer indicating how many tag vertices are visible from the specified odometry node

        Raises:
            ValueError if `vertex_uid` does not correspond to an odometry node.
            Exception if there appear to be more than two incident edges that connect the specified node to other
             odometry nodes.
        """
        if self.vertices[vertex_uid].mode != VertexType.ODOMETRY:
            raise ValueError("Specified vertex type is not an odometry vertex")

        odom_edge_uids = []
        num_tags_visible = 0
        for e in self._verts_to_edges[vertex_uid]:
            edge = self.edges[e]
            start_vertex = self.vertices[edge.startuid]
            end_vertex = self.vertices[edge.enduid]
            if start_vertex.mode == VertexType.ODOMETRY and end_vertex.mode == VertexType.ODOMETRY:
                odom_edge_uids.append(e)
            elif start_vertex.mode == VertexType.TAG or end_vertex.mode == VertexType.TAG:
                num_tags_visible += 1

        if len(odom_edge_uids) > 2:
            raise Exception("Odometry vertex appears to be incident to > two odometry vertices")

        adj_chi2 = 0.0
        for our_edge in odom_edge_uids:
            g2o_edge = self.our_edges_to_g2o_edges[our_edge]
            adj_chi2 += self.get_chi2_of_edge(g2o_edge)
        return adj_chi2, num_tags_visible

    def optimize_graph(self) -> float:
        """Optimize the graph using g2o.

        The g2o_status attribute is set to to the g2o success output.

        Returns:
            Chi2 sum of optimized graph as returned by the call to `self.check_optimized_edges(self.optimized_graph)`
        """
        self.optimized_graph: g2o.SparseOptimizer = self.graph_to_optimizer()
        self.optimized_graph.initialize_optimization()
        run_status = self.optimized_graph.optimize(1024)

        print("checking unoptimized edges")
        self.check_optimized_edges(self.unoptimized_graph)
        print("checking optimized edges")
        optimized_chi_sqr = self.check_optimized_edges(self.optimized_graph)

        self.g2o_status = run_status
        return optimized_chi_sqr

    def graph_to_optimizer(self) -> g2o.SparseOptimizer:
        """Convert a :class: graph to a :class: g2o.SparseOptimizer.  Only the edges and vertices fields need to be
        filled out.

        Vertices' ids in the resulting g2o.SparseOptimizer match their UIDs in the self.vertices attribute.

        Returns:
            A :class: g2o.SparseOptimizer that can be optimized via its optimize class method.
        """
        optimizer: g2o.SparseOptimizer = g2o.SparseOptimizer()
        optimizer.set_algorithm(g2o.OptimizationAlgorithmLevenberg(
            g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())))
        cpp_bool_ret_val_check = True

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
                cpp_bool_ret_val_check &= optimizer.add_vertex(vertex)
            cam_idx = 0
            for i in self.edges:
                if self.edges[i].corner_ids is None:
                    edge = g2o.EdgeSE3Expmap()
                    for j, k in enumerate([self.edges[i].startuid,
                                           self.edges[i].enduid]):
                        edge.set_vertex(j, optimizer.vertex(k))
                        edge.set_measurement(pose_to_se3quat(self.edges[i].measurement))
                        edge.set_information(self.edges[i].information)
                        edge.set_id(i)
                    cpp_bool_ret_val_check &= optimizer.add_edge(edge)
                    self.our_edges_to_g2o_edges[i] = edge
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
                        edge.set_id(i)
                        cpp_bool_ret_val_check &= edge.set_parameter_id(0, cam_idx)
                        if self.use_huber:
                            cpp_bool_ret_val_check &= edge.set_robust_kernel(g2o.RobustKernelHuber(self.huber_delta))
                        cpp_bool_ret_val_check &= optimizer.add_edge(edge)

                        self.our_edges_to_g2o_edges[i] = edge
                    cam_idx += 1
        else:
            for i in self.vertices:
                vertex = g2o.VertexSE3()
                vertex.set_id(i)
                vertex.set_estimate(pose_to_isometry(self.vertices[i].estimate))
                vertex.set_fixed(self.vertices[i].fixed)
                cpp_bool_ret_val_check &= optimizer.add_vertex(vertex)

            for i in self.edges:
                edge = g2o.EdgeSE3()

                for j, k in enumerate([self.edges[i].startuid, self.edges[i].enduid]):
                    edge.set_vertex(j, optimizer.vertex(k))

                edge.set_measurement(pose_to_isometry(self.edges[i].measurement))
                edge.set_information(self.edges[i].information)
                edge.set_id(i)

                cpp_bool_ret_val_check &= optimizer.add_edge(edge)
                self.our_edges_to_g2o_edges[i] = edge

        if not cpp_bool_ret_val_check:
            raise Exception("A g2o optimizer method returned false in the graph_to_optimizer method")

        return optimizer

    def delete_tag_vertex(self, vertex_uid: int):
        """Deletes a tag vertex from relevant attributes.

        Deletes the tag vertex from the following instance attributes:
        - `_verts_to_edges`
        - `vertices`

        All incident edges to the vertex are deleted from the following instance attributes:
        - `edges`
        - `our_edges_to_g2o_edges`
        - The sets stored as the values in the `_verts_to_edges` dictionary

        No edges or vertices are modified in either of the attributes that are g2o graphs.

        Arguments:
            vertex_uid (int): UID of vertex to delete which must be of a VertexType.TAG type.

        Raises:
            ValueError if the specified vertex to delete is not of a VertexType.TAG type.
        """
        if self.vertices[vertex_uid].mode != VertexType.TAG:
            raise ValueError("Specified vertex for deletion is not a tag vertex")

        # Delete connected edge(s)
        connected_edges = self._verts_to_edges[vertex_uid]
        for edge_uid in connected_edges:
            if self.our_edges_to_g2o_edges.__contains__(edge_uid):
                self.our_edges_to_g2o_edges.__delitem__(edge_uid)
            if self.edges[edge_uid].startuid != vertex_uid:
                self._verts_to_edges[self.edges[edge_uid].startuid].remove(edge_uid)
            else:
                self._verts_to_edges[self.edges[edge_uid].enduid].remove(edge_uid)
            self.edges.__delitem__(edge_uid)

        # Delete vertex
        self._verts_to_edges.__delitem__(vertex_uid)
        self.vertices.__delitem__(vertex_uid)
    
    def remove_edge(self, edge_id: int):
        """
        Removes the specified edge from this graph
        """
        edge = self.edges[edge_id]
        self._verts_to_edges[edge.startuid].remove(edge_id)
        self._verts_to_edges[edge.enduid].remove(edge_id)
        del self.edges[edge_id]

    # -- Utility methods --

    def update_edges(self) -> None:
        """Populates the information attribute of each of the edges.

        Raises:
            Exception if an edge is encountered whose start mode is not an odometry node
            Exception if an edge has an unhandled end node type
        """
        for uid in self.edges:
            edge = self.edges[uid]
            start_mode = self.vertices[edge.startuid].mode
            end_mode = self.vertices[edge.enduid].mode
            if start_mode == VertexType.ODOMETRY:
                if end_mode == VertexType.ODOMETRY:
                    self.edges[uid].information = np.diag(np.exp(-self._weights['odometry']))
                elif end_mode == VertexType.TAG:
                    if self.is_sparse_bundle_adjustment:
                        self.edges[uid].information = np.diag(np.exp(-self._weights['tag_sba']))
                    else:
                        self.edges[uid].information = np.diag(np.exp(-self._weights['tag']))
                elif end_mode == VertexType.DUMMY:
                    # TODO: this basis is not very pure and results in weight on each dimension of the quaternion (seems
                    #  to work though)
                    basis = self._basis_matrices[uid][3:6, 3:6]
                    cov = np.diag(np.exp(-self._weights['dummy']))
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
                    self.edges[uid].information = np.eye(6, 6)
                else:
                    raise Exception("Edge of end type {} not recognized.".format(end_mode))
                if self.edges[uid].information_prescaling is not None:
                    prescaling_matrix = self.edges[uid].information_prescaling
                    if prescaling_matrix.ndim == 1:
                        prescaling_matrix = np.diag(prescaling_matrix)
                    self.edges[uid].information = prescaling_matrix * self.edges[uid].information
            else:
                raise Exception("Edge of start type {} not recognized.".format(start_mode))

    def update_vertices(self) -> None:
        """Update the vertices' estimate attributes with the optimized graph values' estimates.
        """
        for uid in self.optimized_graph.vertices():
            if self.is_sparse_bundle_adjustment:
                if type(self.optimized_graph.vertex(uid).estimate()) == np.ndarray:
                    self.vertices[uid].estimate = self.optimized_graph.vertex(uid).estimate()
                else:
                    self.vertices[uid].estimate = self.optimized_graph.vertex(uid).estimate().to_vector()
            else:
                self.vertices[uid].estimate = isometry_to_pose(self.optimized_graph.vertices()[uid].estimate())

    def _generate_basis_matrices(self) -> None:
        """Generate basis matrices used to show how a change in global yaw changes the values of a local measurement.

        This is used for dummy edges. For other edge types, the basis is simply the identity matrix.
        """
        basis_matrices = {}

        for uid in self.edges:
            if (self.vertices[self.edges[uid].startuid].mode == VertexType.DUMMY) \
                    != (self.vertices[self.edges[uid].enduid].mode == VertexType.DUMMY):
                basis_matrices[uid] = np.eye(6)
                if not self.is_sparse_bundle_adjustment:
                    basis_matrices[uid][3:6, 3:6] = global_yaw_effect_basis(
                        Rot.from_quat(self.vertices[self.edges[uid].enduid].estimate[3:7]), self.gravity_axis)
            else:
                basis_matrices[uid] = np.eye(6)
        self._basis_matrices = basis_matrices

    def connected_components(self) -> List[Graph]:
        """Return a list of graphs representing connecting components of the input graph.

        If the graph is connected, there should only be one element in the output.

        Returns:
            A list of :class: Graph containing the connected components.
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
        return [Graph(vertices={k: self.vertices[k] for k in group[0]}, edges={k: self.edges[k] for k in group[1]})
                for group in groups]

    def integrate_path(self, edgeuids, initial=np.array([0, 0, 0, 0, 0, 0, 1])) -> np.ndarray:
        """Returns an array of vectors containing translation and rotation information for the prescribed edge UIDs.
        """
        poses = [initial]
        for edgeuid in edgeuids:
            old_pose = measurement_to_matrix(poses[-1])
            transform = measurement_to_matrix(self.edges[edgeuid].measurement)
            new_pose = old_pose.dot(transform)
            translation = new_pose[:3, 3]
            rotation = Rot.from_matrix(new_pose[:3, :3]).as_quat()
            poses.append(np.concatenate([translation, rotation]))
        return np.array(poses)

    # -- Getters & Setters --
    
    def set_weights(self, weights: Dict[str, np.ndarray], scale_by_edge_amount: bool = True):
        if scale_by_edge_amount:
            odom_edges = 0
            tag_edges = 0
            for edge_id, edge in self.edges.items():
                if edge.get_end_vertex_type(self.vertices) == VertexType.ODOMETRY:
                    odom_edges += 1
                elif edge.get_end_vertex_type(self.vertices) in (VertexType.TAG, VertexType.TAGPOINT):
                    tag_edges += 1
            weights['odom_tag_ratio'] = weights.get('odom_tag_ratio', 1) * tag_edges / odom_edges
            self._weights = graph_utils.normalize_weights(weights, is_sba=self.is_sparse_bundle_adjustment)
        else:
            self._weights = weights

    def get_weights(self):
        return self._weights

    def get_tags_all_position_estimate(self) -> np.ndarray:
        """Returns an array position estimates for every edge that connects an odometry vertex to a tag vertex.
        """
        tags = np.reshape([], [0, 8])  # [x, y, z, qx, qy, qz, 1, id]
        for edgeuid in self.edges:
            edge = self.edges[edgeuid]
            if self.vertices[edge.startuid].mode == VertexType.ODOMETRY and self.vertices[edge.enduid].mode == \
                    VertexType.TAG:
                odom_transform = measurement_to_matrix(
                    self.vertices[edge.startuid].estimate)
                edge_transform = measurement_to_matrix(edge.measurement)

                tag_transform = odom_transform.dot(edge_transform)
                tag_translation = tag_transform[:3, 3]
                tag_rotation = Rot.from_matrix(tag_transform[:3, :3]).as_quat()
                tag_pose = np.concatenate(
                    [tag_translation, tag_rotation, [edge.enduid]])
                tags = np.vstack([tags, tag_pose])
        return tags

    def get_subgraph(self, start_vertex_uid, end_vertex_uid) -> Graph:
        """Returns a Graph instance that is a subgraph created from the specified range of odometry vertices

        Args:
            start_vertex_uid: First vertex in range of vertices from which to create a subgraph
            end_vertex_uid: Last vertex in range of vertices from which to create a subgraph

        Returns:
            A Graph object that was constructed from all odometry vertices within the prescribed range and any
            incident tag vertices.
        """
        start_found = False
        edges: Dict[int, Edge] = {}
        vertices: Dict[int, Vertex] = {}
        for i, edgeuid in enumerate(self.get_ordered_odometry_edges()[0]):
            edge = self.edges[edgeuid]
            if edge.startuid == start_vertex_uid:
                start_found = True
            if start_found:
                vertices[edge.enduid] = self.vertices[edge.enduid]
                vertices[edge.startuid] = self.vertices[edge.startuid]
                edges[edgeuid] = edge
            if edge.enduid == end_vertex_uid:
                break

        # Find tags and edges connecting to the found vertices
        for edgeuid in self.edges:
            edge = self.edges[edgeuid]
            if self.vertices[edge.startuid].mode in (VertexType.TAG, VertexType.DUMMY) and edge.enduid in vertices:
                edges[edgeuid] = edge
                vertices[edge.startuid] = self.vertices[edge.startuid]
            if self.vertices[edge.enduid].mode in (VertexType.TAG, VertexType.DUMMY) and edge.startuid in vertices:
                edges[edgeuid] = edge
                vertices[edge.enduid] = self.vertices[edge.enduid]

        for (vert_id, vert) in self.vertices.items():
            if vert.mode == VertexType.TAGPOINT:
                vertices[vert_id] = vert

        ret_graph = Graph(vertices, edges,
                          weights=self._weights,
                          gravity_axis=self.gravity_axis,
                          is_sparse_bundle_adjustment=self.is_sparse_bundle_adjustment,
                          use_huber=self.use_huber,
                          huber_delta=self.huber_delta,
                          damping_status=self.damping_status)
        return ret_graph

    def get_tag_verts(self):
        """Return a list of of the tag vertices
        """
        tag_verts = []
        for vertex in self.vertices:
            if self.vertices[vertex].mode == VertexType.TAG:
                tag_verts.append(vertex)
        return tag_verts

    def get_ordered_odometry_edges(self) -> List[List[int]]:
        """Generate a list of a list of edges ordered by start of path to end.

        The lists are different connected paths. As long as the graph is connected, the output list should only contain
        one list of edges.

        Returns:
            A list of lists of edge UIDs, where each sublist is a sequence of connected edges.
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
                current_start_found = edge.startuid == self.edges[segments[i][-1]].enduid
                current_end_found = edge.enduid == self.edges[segments[i][0]].startuid

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

    # -- Expectation maximization-related methods  --

    def expectation_maximization_once(self) -> Dict[str, np.ndarray]:
        """Run one cycle of expectation maximization.

        It generates an unoptimized graph from current vertex estimates and edge measurements and importances, and
        optimizes the graph. Using the errors, it tunes the weights so that the variances maximize the likelihood of
        each error by type.
        """
        self.generate_unoptimized_graph()
        self.optimize_graph()
        self.update_vertices()
        self.generate_maximization_params()
        return self.tune_weights()

    def expectation_maximization(self, maxiter=10, tol=1) -> int:
        """Run many iterations of expectation maximization.

        Kwargs:
            maxiter (int): The maximum amount of iterations.
            tol (float): The maximum magnitude of the change in weight vectors that will signal the end of the cycle.

        Returns:
            Number of iterations ran
        """
        previous_weights = self._weights
        i = 0
        while i < maxiter:
            self.expectation_maximization_once()
            new_weights = self._weights
            for weight_type in new_weights:
                if np.linalg.norm(new_weights[weight_type] - previous_weights[weight_type]) < tol:
                    return i
            previous_weights = new_weights
            i += 1
        return i

    def generate_maximization_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate the arrays to be processed by the maximization model.

        Sets the error field to an array of errors, as well as a 2-d array populated by 1-hot 18 element observation
        vectors indicating the type of measurement. The meaning of the position of the one in the observation vector
        corresponds to the layout of the weights vector.

        Returns:
            Errors and observations
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
                    [errors, self._basis_matrices[uid].T.dot(
                        optimized_edges[uid].error())])

            if start_mode == VertexType.ODOMETRY:
                if end_mode == VertexType.ODOMETRY:
                    observations = np.vstack([observations, np.eye(6, 18)])
                elif end_mode == VertexType.TAG:
                    observations = np.vstack([observations, np.eye(6, 18, 6)])
                elif end_mode == VertexType.DUMMY:
                    observations = np.vstack([observations, np.eye(6, 18, 12)])
                elif end_mode == VertexType.WAYPOINT:
                    continue
                else:
                    raise Exception("Unspecified handling for edge of start type {} and end type {}".format(start_mode,
                                                                                                            end_mode))
            else:
                raise Exception("Unspecified handling for edge of start type {} and end type {}".format(start_mode,
                                                                                                        end_mode))

        self.errors = errors
        self.observations = observations
        return errors, observations

    def tune_weights(self):
        """Tune the weights to maximize the likelihood of the errors found between the unoptimized and optimized graphs.
        """
        results = maxweights(self.observations, self.errors, self._weights)
        self.maximization_success = results.success
        self._weights = graph_utils.weight_dict_from_array(results.x)
        self.maximization_results = results
        self.update_edges()
        return self._weights
