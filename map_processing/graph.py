"""
Contains the Graph class which store a map in graph form and optimizes it.
"""

from __future__ import annotations

import copy
import itertools
from collections import defaultdict
from typing import *

import g2o
from g2o import SE3Quat, SparseOptimizer, EdgeProjectPSI2UV, EdgeSE3Expmap, EdgeSE3
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation as Rot

import map_processing.graph_opt_utils
from expectation_maximization.maximization_model import maxweights
from . import PrescalingOptEnum, graph_opt_utils, ASSUMED_TAG_SIZE
from .graph_vertex_edge_classes import *
from .transform_utils import pose_to_isometry, pose_to_se3quat, global_yaw_effect_basis, isometry_to_pose, \
    transform_vector_to_matrix, transform_matrix_to_vector, se3_quat_average, make_sba_tag_arrays, FLIP_Y_AND_Z_AXES


class Graph:
    """A class for the graph encoding a map with class methods to optimize it.

    Makes use of g2o and, optionally, expectation maximization.
    """

    _chi2_dict_template = {
        "odometry": {
            "sum": 0.0,
            "edges": 0
        },
        "tag": {
            "sum": 0.0,
            "edges": 0
        },
        "dummy": {
            "sum": 0.0,
            "edges": 0
        },
    }

    def __init__(
            self,
            vertices: Dict[int, Vertex],
            edges: Dict[int, Edge],
            weights: Optional[Dict[str, np.ndarray]] = None,
            gravity_axis: str = "y",
            is_sparse_bundle_adjustment: bool = False,
            use_huber: bool = False,
            huber_delta=None,
            damping_status: bool = False
    ):
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
        if weights is None:
            weights = {
                'odometry': np.ones(6),
                'tag_sba':  np.ones(2),
                'tag':      np.ones(6),
                'dummy':    np.ones(3)
            }

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
        self.observations = np.reshape([], [0, len(weights) * 6])  # TODO ensure the shape is right
        self.maximization_success: bool = False
        self.maximization_results = OptimizeResult
        self.unoptimized_graph: Union[SparseOptimizer, None] = None
        self.optimized_graph: Union[SparseOptimizer, None] = None
        self.update_edge_information()

        # This is populated in graph_to_optimizer and is currently no updated anywhere else
        self.our_edges_to_g2o_edges: Dict[int, Union[EdgeProjectPSI2UV, EdgeSE3Expmap, EdgeSE3]] = {}

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

    def get_chi2_by_edge_type(self, graph: SparseOptimizer, verbose: bool = True) -> Dict[str, Dict[str, float]]:
        """
        Iterates through the edges and calculates the chi2 of each, sorting them into categories based on the end vertex

        Args:
            graph: a SparseOptimizer object
            verbose (bool): Boolean for whether or not to print the chi2 values

        Returns:
            A dict mapping 'odometry', 'tag', and 'dummy' to the chi2s corresponding to that edge type
        """
        chi2s = dict(Graph._chi2_dict_template)
        for edge in graph.edges():
            end_mode = self.vertices[self.edges[edge.id()].enduid].mode
            if end_mode == VertexType.ODOMETRY:
                chi2s['odometry']['sum'] += graph_opt_utils.get_chi2_of_edge(edge)
                chi2s['odometry']['edges'] += 1
            elif end_mode == VertexType.TAG or end_mode == VertexType.TAGPOINT:
                chi2s['tag']['sum'] += graph_opt_utils.get_chi2_of_edge(edge)
                chi2s['tag']['edges'] += 1
            elif end_mode == VertexType.DUMMY:
                chi2s['dummy']['sum'] += graph_opt_utils.get_chi2_of_edge(edge)
                chi2s['dummy']['edges'] += 1
            start_mode = self.vertices[self.edges[edge.id()].startuid].mode
            if start_mode != VertexType.ODOMETRY:
                raise Exception(f'Original is not odometry. Edge type: {type(edge)}. Start: {start_mode}. End: '
                                f'{end_mode}')
        for edge_type in chi2s:
            chi2s[edge_type]['average'] = chi2s[edge_type]['sum'] / chi2s[edge_type]['edges']
        if verbose:
            print(chi2s)
        return chi2s

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
            adj_chi2 += graph_opt_utils.get_chi2_of_edge(g2o_edge)
        return adj_chi2, num_tags_visible

    def optimize_graph(self) -> float:
        """Optimize the graph using g2o (optimization result is a SparseOptimizer object, which is stored in the
        optimized_graph attribute). The g2o_status attribute is set to to the g2o success output.

        Returns:
            Chi2 sum of optimized graph as returned by the call to `self.sum_optimized_edges_chi2(self.optimized_graph)`
        """
        self.optimized_graph: SparseOptimizer = self.graph_to_optimizer()
        self.optimized_graph.initialize_optimization()
        run_status = self.optimized_graph.optimize(1024)

        print("checking unoptimized edges")
        graph_opt_utils.sum_optimized_edges_chi2(self.unoptimized_graph)
        print("checking optimized edges")
        optimized_chi_sqr = graph_opt_utils.sum_optimized_edges_chi2(self.optimized_graph)

        self.g2o_status = run_status
        return optimized_chi_sqr

    def graph_to_optimizer(self) -> SparseOptimizer:
        """Convert a :class: graph to a :class: SparseOptimizer.  Only the edges and vertices fields need to be
        filled out.

        Vertices' ids in the resulting SparseOptimizer match their UIDs in the self.vertices attribute.

        Returns:
            A :class: SparseOptimizer that can be optimized via its optimize class method.
        """
        optimizer: SparseOptimizer = SparseOptimizer()
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
                    edge = EdgeSE3Expmap()
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
                        edge = EdgeProjectPSI2UV()
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
                edge = EdgeSE3()

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

    # -- Utility methods --

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

    def filter_out_high_chi2_observation_edges(self, filter_std_dv_multiple: float) -> None:
        """Calls remove_edge on every edge whose associated chi2 value in the optimized_graph attribute is above the
        specified threshold.

        The threshold is given by m + filter_std_dv_multiple * s where m is the mean, and s is the standard deviation of
        the optimized graph's edges' chi2 values for the edges between tags and tagpoints.

        If the optimized_graph attribute is not assigned, then no action is taken.
        """
        if self.optimized_graph is None:
            return

        chi2_by_edge = {}
        chi2s = []
        for edge in self.optimized_graph.edges():
            end_mode = self.vertices[self.edges[edge.id()].enduid].mode
            start_mode = self.vertices[self.edges[edge.id()].startuid].mode
            if end_mode in (VertexType.TAG, VertexType.TAGPOINT) or start_mode in (VertexType.TAG,
                                                                                   VertexType.TAGPOINT):
                chi2 = graph_opt_utils.get_chi2_of_edge(edge)
                chi2_by_edge[edge.id()] = chi2
                chi2s.append(chi2)
        chi2s = np.array(chi2s)
        mean_chi2 = chi2s.mean()
        std_chi2 = chi2s.std()

        # Filter out chi2 within the specified std devs of the mean
        max_chi2 = mean_chi2 + filter_std_dv_multiple * std_chi2

        for edge_id, chi2 in chi2_by_edge.items():
            if chi2 > max_chi2:
                print(f'Removing edge {edge_id} - chi2 is {chi2}. Goes to Tag '
                      f'{self.vertices[self.edges[edge_id].enduid].meta_data["tag_id"]}')
                self.remove_edge(edge_id)

    def update_edge_information(self) -> None:
        """Sets the information attribute of each of the edges. Values in the _weights dictionary are used to scale
        the information values that are computed.

        Raises:
            Exception if an edge is encountered whose start mode is not an odometry node
            Exception if an edge has an unhandled end node type
        """
        for uid in self.edges:
            edge = self.edges[uid]
            start_mode = self.vertices[edge.startuid].mode
            end_mode = self.vertices[edge.enduid].mode
            if start_mode != VertexType.ODOMETRY:
                raise Exception("Edge of start type {} not recognized.".format(start_mode))

            if end_mode == VertexType.ODOMETRY:
                self.edges[uid].information = np.diag(self._weights['odometry'])
            elif end_mode == VertexType.TAG:
                if self.is_sparse_bundle_adjustment:
                    self.edges[uid].information = np.diag(self._weights['tag_sba'])
                else:
                    self.edges[uid].information = np.diag(self._weights['tag'])
            elif end_mode == VertexType.DUMMY:
                # TODO: this basis is not very pure and results in weight on each dimension of the quaternion (seems
                #  to work though)
                basis = self._basis_matrices[uid][3:6, 3:6]
                cov = np.diag(self._weights['dummy'])
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
                self.edges[uid].information = np.eye(6, 6)  # TODO: set to something other than identity?
            else:
                raise Exception("Edge of end type {} not recognized.".format(end_mode))

            if self.edges[uid].information_prescaling is not None:
                prescaling_matrix = self.edges[uid].information_prescaling
                if prescaling_matrix.ndim == 1:
                    prescaling_matrix = np.diag(prescaling_matrix)
                self.edges[uid].information *= prescaling_matrix

    def update_vertices_estimates(self) -> None:
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
        """Generate basis matrices used to show how a change in global yaw changes the values of a local
        transform_vector.

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
            old_pose = transform_vector_to_matrix(poses[-1])
            transform = transform_vector_to_matrix(self.edges[edgeuid].measurement)
            new_pose = old_pose.dot(transform)
            translation = new_pose[:3, 3]
            rotation = Rot.from_matrix(new_pose[:3, :3]).as_quat()
            poses.append(np.concatenate([translation, rotation]))
        return np.array(poses)

    # -- Getters & Setters --

    def set_weights(self, weights: Dict[str, np.ndarray], scale_by_edge_amount: bool = True) -> None:
        """Sets the weights for the graph representation within this instance (i.e., does not apply the weights to the
        optimizer object; this must be done through the update_edge_information instance method of the Graph class).

        Args:
            weights:
            scale_by_edge_amount: If true, then the weights dictionary used is modified by computing the ratio
             of odometry to tag edges and then applying the weight normalization function (see
             map_processing.graph_opt_utils.normalize_weights). If false, then the weights set are simply equal to the
             provided weights.
        """
        if not scale_by_edge_amount:
            self._weights = dict(weights)
            return

        # Count the number of odometry and tag edges
        num_odom_edges = 0
        num_tag_edges = 0
        for edge_id, edge in self.edges.items():
            if edge.get_end_vertex_type(self.vertices) == VertexType.ODOMETRY:
                num_odom_edges += 1
            elif edge.get_end_vertex_type(self.vertices) in (VertexType.TAG, VertexType.TAGPOINT):
                num_tag_edges += 1

        # Compute the ratio and normalize
        weights['odom_tag_ratio'] = weights.get('odom_tag_ratio', 1) * num_tag_edges / num_odom_edges
        self._weights = map_processing.graph_opt_utils.normalize_weights(weights,
                                                                         is_sba=self.is_sparse_bundle_adjustment)

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
                odom_transform = transform_vector_to_matrix(self.vertices[edge.startuid].estimate)
                edge_transform = transform_vector_to_matrix(edge.measurement)

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

    def get_tag_verts(self) -> List[int]:
        """
        Returns:
            A list of of the tag vertices' UIDs
        """
        tag_verts = []
        for vertex in self.vertices.keys():
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

    def get_optimizer_vertices_dict_by_types(self, types: Optional[Set[VertexType]] = None) -> Dict[int, Vertex]:
        """
        Args:
            types: Vertex types to filter by. If None is passed, then the default filtering is only TAG vertices.

        Returns:
            UID to Vertex mapping of vertices from the optimized graph where the vertices are filtered by type.
        """
        if types is None:
            types = {VertexType.TAG, }
        return {
            uid: Vertex(
                self.vertices[uid].mode,
                self.optimized_graph.vertex(uid).estimate().vector(),
                self.vertices[uid].fixed,
                self.vertices[uid].meta_data
            ) for uid in self.optimized_graph.vertices() if self.vertices[uid].mode in types
        }

    def get_map_tag_id_to_optimizer_pose_estimate(self) -> Dict[int, np.ndarray]:
        return {
            self.vertices[uid].meta_data["tag_id"]: self.optimized_graph.vertex(uid).estimate().vector()
                for uid in self.optimized_graph.vertices() if self.vertices[uid].mode == VertexType.TAG
        }

    # -- Expectation maximization-related methods  --

    def expectation_maximization_once(self) -> Dict[str, np.ndarray]:
        """Run one cycle of expectation maximization.

        It generates an unoptimized graph from current vertex estimates and edge measurements and importances, and
        optimizes the graph. Using the errors, it tunes the weights so that the variances maximize the likelihood of
        each error by type.
        """
        self.generate_unoptimized_graph()
        self.optimize_graph()
        self.update_vertices_estimates()
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
        vectors indicating the type of transform_vector. The meaning of the position of the one in the observation
        vector corresponds to the layout of the weights vector.

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
        self._weights = map_processing.graph_opt_utils.weight_dict_from_array(results.x)
        self.maximization_results = results
        self.update_edge_information()
        return self._weights

    @staticmethod
    def as_graph(dct: Dict, fixed_vertices: Union[VertexType, Tuple[VertexType]] = (),
                 prescaling_opt: PrescalingOptEnum = PrescalingOptEnum.USE_SBA) -> Graph:
        """Convert a dictionary decoded from JSON into a Graph object.

        Args:
            dct (dict): The dictionary to convert to a
            fixed_vertices (tuple): Determines which vertex types to set to fixed. Dummy and Tagpoints are always fixed
                regardless of their presence in the tuple.
            prescaling_opt (PrescalingOptEnum): Selects which logical branches to use. If it is equal to
            `PrescalingOptEnum.USE_SBA`, then sparse bundle adjustment is used; otherwise, the outcome only differs
             between the remaining enum values by how the tag edge prescaling matrix is selected. Read the
             PrescalingOptEnum class documentation for more information.

        Returns:
            A graph derived from the input dictionary.

        Raises:
            Exception: if prescaling_opt is an enum_value that is not handled.
            Exception: if no pose data was provided to the dictionary
            KeyError: exceptions from missing keys that are expected to be in dct (i.e., KeyErrors are not caught)

        Notes:
            Coordinate system expectations:
            - Odometry measurements are expected to be in a right-handed coordinate system (which follows Apple's ARKit)
            - Tag observations are expected to be in the phone's left-handed coordinate system.

            For those familiar with the old structure of the repository...
            This function was created by combining the as_graph functions from convert_json.py and convert_json_sba.py.
            Because the two implementations shared a lot of code but also deviated in a few important ways, the entire
            functionality of each was preserved in this function by using the prescaling_opt argument to toggle on/off
            logical branches according to the implementation in convert_json.py and the implementation in
            convert_json_sba.py.
        """
        # Pull out this equality from the enum (this equality is checked many times)
        use_sba = prescaling_opt == PrescalingOptEnum.USE_SBA

        # Used only if use_sba is false:
        tag_joint_covar = None
        tag_position_variances = None
        tag_orientation_variances = None
        tag_edge_prescaling = None
        previous_pose_matrix = None

        # Used only if use_sba is true:
        camera_intrinsics_for_tag: Union[np.ndarray, None] = None
        tag_corners = None
        true_3d_tag_center: Union[None, np.ndarray] = None
        true_3d_tag_points: Union[None, np.ndarray] = None
        tag_transform_estimates = None
        tag_corner_ids_by_tag_vertex_id = None
        initialize_with_averages = None

        if isinstance(fixed_vertices, VertexType):
            fixed_vertices = (fixed_vertices,)

        if use_sba:
            true_3d_tag_points, true_3d_tag_center = make_sba_tag_arrays(ASSUMED_TAG_SIZE)

        frame_ids = [pose['id'] for pose in dct['pose_data']]
        if len(dct['pose_data']) == 0:
            raise Exception("No pose data in the provided dictionary")

        pose_matrices = np.array([pose['pose'] for pose in dct['pose_data']]).reshape((-1, 4, 4), order="F")
        odom_vertex_estimates = transform_matrix_to_vector(pose_matrices, invert=use_sba)

        # Extract data from the dictionary. If no tag data exists, then generate placeholder vectors of the right shape
        # containing 0s
        if len(dct['tag_data']) > 0:
            good_tag_detections = dct['tag_data']
            # good_tag_detections = list(filter(lambda l: len(l) > 0,
            #                              [[tag_data for tag_data in tags_from_frame
            #                           if np.linalg.norm(np.asarray([tag_data['tag_pose'][i] for i in (3, 7, 11)])) < 1
            #                              and tag_data['tag_pose'][10] < 0.7] for tags_from_frame in dct['tag_data']]))

            tag_pose_flat = np.vstack([[x['tag_pose'] for x in tags_from_frame] for tags_from_frame in
                                       good_tag_detections])

            if use_sba:
                camera_intrinsics_for_tag = np.vstack([[x['camera_intrinsics'] for x in tags_from_frame]
                                                       for tags_from_frame in good_tag_detections])
                tag_corners = np.vstack([[x['tag_corners_pixel_coordinates'] for x in tags_from_frame] for
                                         tags_from_frame in good_tag_detections])
            else:
                tag_joint_covar = np.vstack([[x['joint_covar'] for x in tags_from_frame] for tags_from_frame in
                                             good_tag_detections])
                tag_position_variances = np.vstack([[x['tag_position_variance'] for x in tags_from_frame] for
                                                    tags_from_frame in good_tag_detections])
                tag_orientation_variances = np.vstack([[x['tag_orientation_variance'] for x in tags_from_frame] for
                                                       tags_from_frame in dct['tag_data']])

            tag_ids = np.vstack(list(itertools.chain(*[[x['tag_id'] for x in tags_from_frame] for tags_from_frame in
                                                       good_tag_detections])))
            pose_ids = np.vstack(list(itertools.chain(*[[x['pose_id'] for x in tags_from_frame] for tags_from_frame in
                                                        good_tag_detections])))
        else:
            tag_pose_flat = np.zeros((0, 16))
            tag_ids = np.zeros((0, 1), dtype=np.int64)
            pose_ids = np.zeros((0, 1), dtype=np.int64)

            if use_sba:
                camera_intrinsics_for_tag = np.zeros((0, 4))
                tag_corners = np.zeros((0, 8))
            else:
                tag_joint_covar = np.zeros((0, 49), dtype=np.double)
                tag_position_variances = np.zeros((0, 3), dtype=np.double)
                tag_orientation_variances = np.zeros((0, 4), dtype=np.double)

        unique_tag_ids = np.unique(tag_ids)
        if use_sba:
            tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(0, unique_tag_ids.size * 5, 5)))
        else:
            tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(unique_tag_ids.size)))

        # The camera axis used to get tag measurements is flipped relative to the phone frame used for odom
        # measurements. Additionally, note that the matrix here is recorded in row-major format.
        tag_edge_measurements_matrix = np.matmul(FLIP_Y_AND_Z_AXES, tag_pose_flat.reshape([-1, 4, 4]))
        tag_edge_measurements = transform_matrix_to_vector(tag_edge_measurements_matrix)
        n_pose_ids = pose_ids.shape[0]

        if not use_sba:
            if prescaling_opt == PrescalingOptEnum.FULL_COV:
                # Note that we are ignoring the variance deviation of qw since we use a compact quaternion
                # parameterization of orientation
                tag_joint_covar_matrices = tag_joint_covar.reshape((-1, 7, 7))

                # TODO: for some reason we have missing measurements (all zeros).  Throw those out
                tag_edge_prescaling = np.array([np.linalg.inv(covar[:-1, :-1]) if np.linalg.det(covar[:-1, :-1]) != 0
                                                else np.zeros((6, 6)) for covar in tag_joint_covar_matrices])
            elif prescaling_opt == PrescalingOptEnum.DIAG_COV:
                tag_edge_prescaling = 1. / np.hstack((tag_position_variances, tag_orientation_variances[:, :-1]))
            elif prescaling_opt == PrescalingOptEnum.ONES:
                tag_edge_prescaling = np.ones((n_pose_ids, 6, 6))
            else:
                raise Exception("{} is not yet handled".format(str(prescaling_opt)))

        tag_id_by_tag_vertex_id = dict(zip(tag_vertex_id_by_tag_id.values(), tag_vertex_id_by_tag_id.keys()))
        if use_sba:
            tag_corner_ids_by_tag_vertex_id = dict(
                zip(tag_id_by_tag_vertex_id.keys(),
                    map(lambda tag_vertex_id_x: list(range(tag_vertex_id_x + 1, tag_vertex_id_x + 5)),
                        tag_id_by_tag_vertex_id.keys())))

        tag_vertex_id_and_index_by_frame_id = {}  # Enable lookup of tags by the frame they appear in
        for tag_index, (tag_id, tag_frame) in enumerate(np.hstack((tag_ids, pose_ids))):
            tag_vertex_id = tag_vertex_id_by_tag_id[tag_id]
            tag_vertex_id_and_index_by_frame_id[tag_frame] = tag_vertex_id_and_index_by_frame_id.get(tag_frame, [])
            tag_vertex_id_and_index_by_frame_id[tag_frame].append((tag_vertex_id, tag_index))

        # Possible filtering method for outliers in raw data?
        # last_tag = tag_vertex_id_and_index_by_frame_id[min(tag_vertex_id_and_index_by_frame_id.keys())][0][0]
        # tag_detections = []
        # for frame_id in sorted(tag_vertex_id_and_index_by_frame_id):
        #     if tag_vertex_id_and_index_by_frame_id[frame_id][0][0] == last_tag:
        #         tag_detections.append(np.asarray(
        #             tag_edge_measurements[tag_vertex_id_and_index_by_frame_id[frame_id][0][1]]))
        #     else:
        #         average_quaternion = np.mean(tag_detections, axis=0)
        #         deviation_quaternion = np.std(tag_detections, axis=0)
        #         last_tag = tag_vertex_id_and_index_by_frame_id[frame_id][0][0]
        #         tag_detections = []

        waypoint_names = [location_data['name'] for location_data in dct['location_data']]
        unique_waypoint_names = np.unique(waypoint_names)
        num_unique_waypoint_names = unique_waypoint_names.size

        waypoint_edge_measurements_matrix = np.zeros((0, 4, 4))
        if len(dct['location_data']) > 0:
            waypoint_edge_measurements_matrix = np.concatenate(
                [np.asarray(location_data['transform']).reshape((-1, 4, 4)) for location_data in dct['location_data']]
            )
        waypoint_edge_measurements = transform_matrix_to_vector(waypoint_edge_measurements_matrix)
        waypoint_frame_ids = [location_data['pose_id'] for location_data in dct['location_data']]

        if use_sba:
            waypoint_vertex_id_by_name = dict(
                zip(unique_waypoint_names,
                    range(unique_tag_ids.size * 5, unique_tag_ids.size * 5 + num_unique_waypoint_names)))
        else:
            waypoint_vertex_id_by_name = dict(
                zip(unique_waypoint_names, range(unique_tag_ids.size, unique_tag_ids.size + num_unique_waypoint_names)))

        waypoint_name_by_vertex_id = dict(zip(waypoint_vertex_id_by_name.values(), waypoint_vertex_id_by_name.keys()))
        waypoint_vertex_id_and_index_by_frame_id = {}  # Enable lookup of waypoints by the frame they appear in

        for waypoint_index, (waypoint_name, waypoint_frame) in enumerate(zip(waypoint_names, waypoint_frame_ids)):
            waypoint_vertex_id = waypoint_vertex_id_by_name[waypoint_name]
            waypoint_vertex_id_and_index_by_frame_id[waypoint_frame] = waypoint_vertex_id_and_index_by_frame_id.get(
                waypoint_name, [])
            waypoint_vertex_id_and_index_by_frame_id[waypoint_frame].append((waypoint_vertex_id, waypoint_index))

        num_tag_edges = edge_counter = 0
        vertices = {}
        edges = {}
        counted_tag_vertex_ids = set()
        counted_waypoint_vertex_ids = set()
        previous_vertex = None
        first_odom_processed = False
        if use_sba:
            vertex_counter = unique_tag_ids.size * 5 + num_unique_waypoint_names
            # TODO: debug; this appears to be counterproductive
            initialize_with_averages = False
            tag_transform_estimates = defaultdict(lambda: [])
        else:
            vertex_counter = unique_tag_ids.size + num_unique_waypoint_names
        for i, odom_frame in enumerate(frame_ids):
            current_odom_vertex_uid = vertex_counter
            vertices[current_odom_vertex_uid] = Vertex(
                mode=VertexType.ODOMETRY,
                estimate=odom_vertex_estimates[i],
                fixed=not first_odom_processed or VertexType.ODOMETRY in fixed_vertices,
                meta_data={'pose_id': odom_frame})
            first_odom_processed = True
            vertex_counter += 1

            # Connect odom to tag vertex
            for tag_vertex_id, tag_index in tag_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
                if use_sba:
                    current_tag_transform_estimate = \
                        SE3Quat(np.hstack((true_3d_tag_center, [0, 0, 0, 1]))) * \
                        SE3Quat(tag_edge_measurements[tag_index]).inverse() * \
                        SE3Quat(vertices[current_odom_vertex_uid].estimate)
                    # if(tag_vertex_id == 5):
                    #     print(current_tag_transform_estimate.to_homogeneous_matrix())
                    # keep track of estimates in case we want to average them to initialize the graph
                    tag_transform_estimates[tag_vertex_id].append(current_tag_transform_estimate)
                    if tag_vertex_id not in counted_tag_vertex_ids:
                        vertices[tag_vertex_id] = Vertex(
                            mode=VertexType.TAG,
                            estimate=current_tag_transform_estimate.to_vector(),
                            fixed=VertexType.TAG in fixed_vertices,
                            meta_data={'tag_id': tag_id_by_tag_vertex_id[tag_vertex_id]})

                        for idx, true_point_3d in enumerate(true_3d_tag_points):
                            vertices[tag_corner_ids_by_tag_vertex_id[tag_vertex_id][idx]] = Vertex(
                                mode=VertexType.TAGPOINT,
                                estimate=np.hstack((true_point_3d, [0, 0, 0, 1])),
                                fixed=True)
                        counted_tag_vertex_ids.add(tag_vertex_id)
                    # adjust the x-coordinates of the detections to account for differences in coordinate systems
                    # induced by the FLIP_Y_AND_Z_AXES
                    tag_corners[tag_index][::2] = 2 * camera_intrinsics_for_tag[tag_index][2] - \
                        tag_corners[tag_index][::2]

                    # Commented-out (unused):
                    # TODO: create proper subclasses
                    # for k, point in enumerate(true_3d_tag_points):
                    #     point_in_camera_frame = SE3Quat(tag_edge_measurements[tag_index]) * \
                    #                                     (point - np.array([0, 0, 1]))
                    #     cam = CameraParameters(camera_intrinsics_for_tag[tag_index][0],
                    #                            camera_intrinsics_for_tag[tag_index][2:], 0)
                    #     print("chi2", np.sum(np.square(tag_corners[tag_index][2*k : 2*k + 2] -
                    #                                    cam.cam_map(point_in_camera_frame))))

                    edges[edge_counter] = Edge(
                        startuid=current_odom_vertex_uid,
                        enduid=tag_vertex_id,
                        corner_ids=tag_corner_ids_by_tag_vertex_id[tag_vertex_id],
                        information=np.eye(2),
                        information_prescaling=None,
                        camera_intrinsics=camera_intrinsics_for_tag[tag_index],
                        measurement=tag_corners[tag_index]
                    )
                else:
                    if tag_vertex_id not in counted_tag_vertex_ids:
                        vertices[tag_vertex_id] = Vertex(
                            mode=VertexType.TAG,
                            estimate=transform_matrix_to_vector(pose_matrices[i].dot(
                                tag_edge_measurements_matrix[tag_index])),
                            fixed=VertexType.TAG in fixed_vertices,
                            meta_data={'tag_id': tag_id_by_tag_vertex_id[tag_vertex_id]})
                        counted_tag_vertex_ids.add(tag_vertex_id)
                    edges[edge_counter] = Edge(
                        startuid=current_odom_vertex_uid,
                        enduid=tag_vertex_id,
                        information=np.eye(6),
                        information_prescaling=tag_edge_prescaling[tag_index],
                        measurement=tag_edge_measurements[tag_index],
                        corner_ids=None,
                        camera_intrinsics=None)

                num_tag_edges += 1
                edge_counter += 1

            # Connect odom to waypoint vertex
            for waypoint_vertex_id, waypoint_index in waypoint_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
                if waypoint_vertex_id not in counted_waypoint_vertex_ids:
                    if use_sba:
                        estimate_arg = (SE3Quat(vertices[current_odom_vertex_uid].estimate).inverse() * SE3Quat(
                            waypoint_edge_measurements[waypoint_index])).to_vector()
                    else:
                        estimate_arg = transform_matrix_to_vector(pose_matrices[i].dot(
                            waypoint_edge_measurements_matrix[waypoint_index]))
                    vertices[waypoint_vertex_id] = Vertex(
                        mode=VertexType.WAYPOINT,
                        estimate=estimate_arg,
                        fixed=VertexType.WAYPOINT in fixed_vertices,
                        meta_data={'name': waypoint_name_by_vertex_id[waypoint_vertex_id]})
                    counted_waypoint_vertex_ids.add(waypoint_vertex_id)

                if use_sba:
                    measurement_arg = (SE3Quat(vertices[waypoint_vertex_id].estimate) * SE3Quat(
                        vertices[current_odom_vertex_uid].estimate).inverse()).to_vector()
                else:
                    measurement_arg = waypoint_edge_measurements[waypoint_index]
                edges[edge_counter] = Edge(
                    startuid=current_odom_vertex_uid,
                    enduid=waypoint_vertex_id,
                    corner_ids=None,
                    information=np.eye(6),
                    information_prescaling=None,
                    camera_intrinsics=None,
                    measurement=measurement_arg)
                edge_counter += 1

            if previous_vertex:
                if use_sba:
                    measurement_arg = (SE3Quat(vertices[current_odom_vertex_uid].estimate) * SE3Quat(
                        vertices[previous_vertex].estimate).inverse()).to_vector()
                else:
                    # TODO: might want to consider prescaling based on the magnitude of the change
                    measurement_arg = transform_matrix_to_vector(
                        np.linalg.inv(previous_pose_matrix).dot(pose_matrices[i]))

                edges[edge_counter] = Edge(
                    startuid=previous_vertex,
                    enduid=current_odom_vertex_uid,
                    corner_ids=None,
                    information=np.eye(6),
                    information_prescaling=None,
                    camera_intrinsics=None,
                    measurement=measurement_arg)
                edge_counter += 1

            # Make dummy node
            dummy_node_uid = vertex_counter
            vertices[dummy_node_uid] = Vertex(
                mode=VertexType.DUMMY,
                estimate=np.hstack((np.zeros(3, ), odom_vertex_estimates[i][3:])),
                fixed=True)
            vertex_counter += 1

            # Connect odometry to dummy node
            edges[edge_counter] = Edge(
                startuid=current_odom_vertex_uid,
                enduid=dummy_node_uid,
                information=np.eye(6),
                information_prescaling=None,
                measurement=np.array([0, 0, 0, 0, 0, 0, 1]),
                corner_ids=None,
                camera_intrinsics=None)
            edge_counter += 1
            previous_vertex = current_odom_vertex_uid

            if not use_sba:
                previous_pose_matrix = pose_matrices[i]

        if use_sba and initialize_with_averages:
            for vertex_id, transforms in tag_transform_estimates.items():
                vertices[vertex_id].estimate = se3_quat_average(transforms).to_vector()

        # TODO: Huber delta should probably scale with pixels rather than error
        resulting_graph = Graph(vertices, edges, gravity_axis='y', is_sparse_bundle_adjustment=use_sba,
                                use_huber=False, huber_delta=None, damping_status=True)
        return resulting_graph

    @staticmethod
    def transfer_vertex_estimates(graph_from: Graph, graph_to: Graph, filter_by: Optional[Set[VertexType]] = None) -> None:
        """Transfer vertex estimates from one graph to another.

        Args:
            graph_from: Graph to transfer vertex estimates from
            graph_to: Graph to transfer vertex estimates to
            filter_by: Only transfer vertex estimates when vertices are of these types. Default behavior when argument
             is None is to apply to no filter.

        Notes:
            Any vertices in graph_to that are not in graph_from are ignored.
        """
        if filter_by is None:
            filter_by = {t for t in VertexType}
        tag_vertices_dict = graph_from.get_optimizer_vertices_dict_by_types(types=filter_by)
        for uid, vertex in tag_vertices_dict.items():
            if uid in graph_to.vertices:
                graph_to.vertices[uid].estimate = vertex.estimate
