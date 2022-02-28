"""
Contains the Graph class which store a map in graph form and optimizes it.
"""

from __future__ import annotations

import copy
from collections import defaultdict
from typing import Set, Dict, Optional, Tuple, Union, List

import numpy as np
import pydantic
# For some reason, the EdgeSE3Gravity class is not being recognized. If the instructions in the README.md are followed,
# then importing this should not be an issue.
# noinspection PyUnresolvedReferences
from g2o import EdgeSE3Gravity
from g2o import SE3Quat, SparseOptimizer, EdgeProjectPSI2UV, EdgeSE3Expmap, OptimizationAlgorithmLevenberg, \
    CameraParameters, RobustKernelHuber, BlockSolverSE3, LinearSolverCholmodSE3, VertexSBAPointXYZ, VertexSE3Expmap, \
    EdgeSE3, VertexSE3
from scipy.optimize import OptimizeResult
from scipy.spatial.transform import Rotation as Rot

from . import PrescalingOptEnum, graph_opt_utils, ASSUMED_TAG_SIZE, VertexType
from .data_models import UGDataSet, OComputeInfParams, Weights
from .graph_vertex_edge_classes import Vertex, Edge
from .transform_utils import pose_to_se3quat, isometry_to_pose, transform_vector_to_matrix, \
    transform_matrix_to_vector, se3_quat_average, make_sba_tag_arrays, pose_to_isometry


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
        "gravity": {
            "sum": 0.0,
            "edges": 0
        },
    }

    def __init__(self, vertices: Dict[int, Vertex], edges: Dict[int, Edge], weights: Optional[Weights] = None,
                 is_sparse_bundle_adjustment: bool = False, use_huber: bool = False, huber_delta=None,
                 damping_status: bool = False):
        """The graph class

        The graph contains a dictionary of vertices and edges, the keys being UIDs such as ints. The start and end UIDs
        in each edge refer to the vertices in the `vertices` dictionary.

        TODO: add rest of the args here
        Args:
            vertices: A dictionary of vertices indexed by UIDs. The UID-vertices associations are referred to by the
             startuid and enduid fields of the :class: Edge  class.
            edges: A dictionary of edges indexed by UIDs.
        """

        self.edges: Dict[int, Edge] = copy.deepcopy(edges)
        self.vertices: Dict[int, Vertex] = copy.deepcopy(vertices)
        self.original_vertices = copy.deepcopy(vertices)
        self.huber_delta: bool = copy.deepcopy(huber_delta)
        self.is_sba: bool = is_sparse_bundle_adjustment
        self.damping_status: bool = damping_status
        self.use_huber: bool = use_huber

        self._weights = copy.deepcopy(weights) if weights is not None else Weights()

        self._verts_to_edges: Dict[int, Set[int]] = {}
        self.our_odom_edges_to_g2o_edges = {}
        self._generate_verts_to_edges_mapping()

        self.g2o_status = -1
        self.maximization_success_status = False
        self.errors = np.array([])
        self.observations = np.reshape([], [0, 18])
        self.maximization_success: bool = False
        self.maximization_results = OptimizeResult
        self.unoptimized_graph: Union[SparseOptimizer, None] = None
        self.optimized_graph: Union[SparseOptimizer, None] = None
        self.update_edge_information()

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
            verbose (bool): Boolean for whether to print the chi2 values

        Returns:
            A dict mapping 'odometry', 'tag', and 'gravity' to the chi2s corresponding to that edge type
        """
        chi2s = dict(Graph._chi2_dict_template)
        for edge in graph.edges():
            this_edge = self.edges[edge.id()]
            chi2 = graph_opt_utils.get_chi2_of_edge(edge)
            end_mode = self.vertices[this_edge.enduid].mode if this_edge.enduid is not None else None
            if end_mode == VertexType.ODOMETRY:
                chi2s["odometry"]["sum"] += chi2
                chi2s["odometry"]["edges"] += 1
            elif end_mode == VertexType.TAG or end_mode == VertexType.TAGPOINT:
                chi2s["tag"]["sum"] += chi2
                chi2s["tag"]["edges"] += 1
            elif end_mode is None:
                chi2s["gravity"]["sum"] += chi2
                chi2s["gravity"]["edges"] += 1
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
            end_vertex = self.vertices[edge.enduid] if edge.enduid is not None else None
            if end_vertex is None:  # Continue if the edge is a gravity edge
                continue
            start_vertex = self.vertices[edge.startuid]
            if start_vertex.mode == VertexType.ODOMETRY and end_vertex.mode == VertexType.ODOMETRY:
                odom_edge_uids.append(e)
            elif start_vertex.mode == VertexType.TAG or end_vertex.mode == VertexType.TAG:
                num_tags_visible += 1

        if len(odom_edge_uids) > 2:
            raise Exception("Odometry vertex appears to be incident to > two odometry vertices")

        adj_chi2 = 0.0
        for our_edge in odom_edge_uids:
            g2o_edge = self.our_odom_edges_to_g2o_edges[our_edge]
            adj_chi2 += graph_opt_utils.get_chi2_of_edge(g2o_edge, g2o_edge.vertices()[0])
        return adj_chi2, num_tags_visible

    def optimize_graph(self, verbose: bool = True) -> float:
        """Optimize the graph using g2o (optimization result is a SparseOptimizer object, which is stored in the
        optimized_graph attribute). The g2o_status attribute is set to the g2o success output.

        Args:
            verbose: Boolean for whether to print diagnostic messages about chi2 sums.

        Returns:
            Chi2 sum of optimized graph as returned by the call to `self.sum_optimizer_edges_chi2(self.optimized_graph)`
        """
        self.optimized_graph: SparseOptimizer = self.graph_to_optimizer()
        self.optimized_graph.initialize_optimization()
        run_status = self.optimized_graph.optimize(1024)
        self.g2o_status = run_status
        optimized_chi_sqr = graph_opt_utils.sum_optimizer_edges_chi2(self.optimized_graph, verbose=False)

        if verbose:
            print("unoptimized edges' chi2 sum:         " + \
                  str(graph_opt_utils.sum_optimizer_edges_chi2(self.unoptimized_graph, verbose=False)))

            print("unoptimized gravity edges' chi2 sum: " + \
                  str(graph_opt_utils.sum_optimizer_edges_chi2(self.unoptimized_graph, verbose=False,
                                                               edge_type_filter={EdgeSE3Gravity})))
            print("optimized edges' chi2 sum:           " + \
                  str(graph_opt_utils.sum_optimizer_edges_chi2(self.optimized_graph, verbose=False)))

            print("optimized gravity edges' chi2 sum:   " + \
                  str(graph_opt_utils.sum_optimizer_edges_chi2(self.optimized_graph, verbose=False,
                                                               edge_type_filter={EdgeSE3Gravity})))
        return optimized_chi_sqr

    def graph_to_optimizer(self) -> SparseOptimizer:
        """Convert a :class: graph to a :class: SparseOptimizer.  Only the edges and vertices fields need to be
        filled out.

        Vertices' ids in the resulting SparseOptimizer match their UIDs in the `self.vertices` attribute.

        Returns:
            A :class: SparseOptimizer that can be optimized via its optimize class method.
        """
        optimizer: SparseOptimizer = SparseOptimizer()
        optimizer.set_algorithm(OptimizationAlgorithmLevenberg(BlockSolverSE3(LinearSolverCholmodSE3())))
        cpp_bool_ret_val_check = True
        self.our_odom_edges_to_g2o_edges.clear()

        # Add all vertices
        for i, vertex_i in self.vertices.items():
            if vertex_i.mode == VertexType.TAGPOINT:
                vertex = VertexSBAPointXYZ()
                vertex.set_estimate(vertex_i.estimate[:3])
            else:
                if self.is_sba:
                    vertex = VertexSE3Expmap()
                    vertex.set_estimate(pose_to_se3quat(vertex_i.estimate))
                else:
                    vertex = VertexSE3()
                    vertex.set_estimate(pose_to_isometry(vertex_i.estimate))
            vertex.set_id(i)
            vertex.set_fixed(vertex_i.fixed)
            cpp_bool_ret_val_check &= optimizer.add_vertex(vertex)

        # Add all edges
        cam_idx = 0
        for i, edge_i in self.edges.items():
            if edge_i.corner_ids is not None:
                # Note: we only use the focal length in the x direction since: (a) that's all that g2o supports and
                # (b) it is always the same in ARKit (at least currently)
                cam = CameraParameters(edge_i.camera_intrinsics[0], edge_i.camera_intrinsics[2:], 0)
                cam.set_id(cam_idx)
                optimizer.add_parameter(cam)
                for corner_idx, corner_id in enumerate(edge_i.corner_ids):
                    edge = EdgeProjectPSI2UV()
                    edge.resize(3)
                    edge.set_vertex(0, optimizer.vertex(corner_id))
                    edge.set_vertex(1, optimizer.vertex(edge_i.startuid))
                    edge.set_vertex(2, optimizer.vertex(edge_i.enduid))
                    edge.set_information(edge_i.information)
                    edge.set_measurement(edge_i.measurement[corner_idx * 2:corner_idx * 2 + 2])
                    edge.set_parameter_id(0, cam_idx)
                    if self.use_huber:
                        edge.set_robust_kernel(RobustKernelHuber(self.huber_delta))
                    optimizer.add_edge(edge)
                cam_idx += 1
            elif edge_i.enduid is None:  # If is none, then this edge is a gravity edge
                edge = EdgeSE3Gravity()
                if self.is_sba:
                    edge.set_odometry_is_se3_expmap()
                edge.set_vertex(0, optimizer.vertex(edge_i.startuid))
                # There is intentionally no end vertex
                edge.set_measurement(edge_i.measurement)
                edge.set_information(edge_i.information)
                cpp_bool_ret_val_check &= optimizer.add_edge(edge)
            else:
                if self.is_sba:
                    edge = EdgeSE3Expmap()
                    edge.set_measurement(pose_to_se3quat(edge_i.measurement))
                else:
                    edge = EdgeSE3()
                    edge.set_measurement(pose_to_isometry(edge_i.measurement))
                edge.set_vertex(0, optimizer.vertex(edge_i.startuid))
                edge.set_vertex(1, optimizer.vertex(edge_i.enduid))
                edge.set_information(edge_i.information)
                cpp_bool_ret_val_check &= optimizer.add_edge(edge)

                if edge_i.start_end[0].mode == VertexType.ODOMETRY and \
                        edge_i.start_end[1].mode == VertexType.ODOMETRY:
                    self.our_odom_edges_to_g2o_edges[i] = edge

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
            vertex_uid (int): UID of vertex to delete which must be of a `VertexType.TAG` type.

        Raises:
            ValueError if the specified vertex to delete is not of a `VertexType.TAG` type.
        """
        if self.vertices[vertex_uid].mode != VertexType.TAG:
            raise ValueError("Specified vertex for deletion is not a tag vertex")

        # Delete connected edge(s)
        connected_edges = self._verts_to_edges[vertex_uid]
        for edge_uid in connected_edges:
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

    def update_edge_information(self, compute_inf_params: Optional[OComputeInfParams] = None) -> None:
        """Invokes the compute_information method on each edge in the graph with the corresponding weights vector as
        the weights_vec argument.

        Prescaling is also applied here if the edge contains prescaling information.

        Args:
            compute_inf_params: Passed down to the `Edge.compute_information` method.

        Raises:
            Exception if an edge is encountered whose start mode is not an odometry node
            Exception if an edge has an unhandled end node type
        """
        for uid in self.edges:
            edge: Edge = self.edges[uid]
            end_mode = self.vertices[edge.enduid].mode if edge.enduid is not None else None
            weights_to_use = self._weights.get_weights_from_end_vertex_mode(
                end_vertex_mode=VertexType.TAGPOINT if edge.corner_ids is not None else end_mode)
            edge.compute_information(weights_vec=weights_to_use, compute_inf_params=compute_inf_params)

            if edge.information_prescaling is not None and len(edge.information_prescaling.shape) != 0:
                prescaling_matrix = self.edges[uid].information_prescaling
                if prescaling_matrix.ndim == 1:
                    prescaling_matrix = np.diag(prescaling_matrix)
                self.edges[uid].information = np.matmul(prescaling_matrix, edge.information)

    def update_vertices_estimates(self) -> None:
        """Update the vertices' estimate attributes with the optimized graph values' estimates.
        """
        for uid in self.optimized_graph.vertices():
            if self.is_sba:
                if type(self.optimized_graph.vertex(uid).estimate()) == np.ndarray:
                    self.vertices[uid].estimate = self.optimized_graph.vertex(uid).estimate()
                else:
                    self.vertices[uid].estimate = self.optimized_graph.vertex(uid).estimate().to_vector()
            else:
                self.vertices[uid].estimate = isometry_to_pose(self.optimized_graph.vertices()[uid].estimate())

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

    def set_weights(self, weights: Weights, scale_by_edge_amount: bool = True) -> None:
        """Sets the weights for the graph representation within this instance (i.e., does not apply the weights to the
        optimizer object; this must be done through the update_edge_information instance method of the Graph class).

        Notes:
            The `weights` argument is deep-copied before being set to the _weights attribute.

        Args:
            weights:
            scale_by_edge_amount: If true, then the odom:tag ratio is scaled by the ratio of tag edges to odometry edges
        """
        self._weights = copy.deepcopy(weights)
        if not scale_by_edge_amount:
            self._weights.scale_tag_and_odom_weights()
            return

        num_odom_edges = 0
        num_tag_edges = 0
        for edge_id, edge in self.edges.items():
            if edge.get_end_vertex_type(self.vertices) == VertexType.ODOMETRY:
                num_odom_edges += 1
            elif edge.get_end_vertex_type(self.vertices) in (VertexType.TAG, VertexType.TAGPOINT):
                num_tag_edges += 1

        # Compute the ratio and normalize
        self._weights.odom_tag_ratio *= num_tag_edges / num_odom_edges
        self._weights.scale_tag_and_odom_weights()

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
        """Returns a Graph instance that is a subgraph created from the specified range of odometry vertices.

        Args:
            start_vertex_uid: First vertex in range of vertices from which to create a subgraph
            end_vertex_uid: Last vertex in range of vertices from which to create a subgraph

        Returns:
            A Graph object constructed from all odometry vertices within the prescribed range and any other vertices
            incident to the odometry vertices. Gravity edges are included (despite their not having been assigned an end
            vertex), as are any tag vertices (even if they are not connected to the included odometry vertices).
        """
        start_found = False
        edges: Dict[int, Edge] = {}
        vertices: Dict[int, Vertex] = {}
        for i, edgeuid in enumerate(self.get_ordered_odometry_edges()[0]):
            edge = self.edges[edgeuid]
            if edge.startuid == start_vertex_uid:
                start_found = True
            if start_found:
                vertices[edge.startuid] = self.vertices[edge.startuid]
                if edge.enduid is not None:
                    vertices[edge.enduid] = self.vertices[edge.enduid]
                edges[edgeuid] = edge
            if edge.enduid == end_vertex_uid:
                break

        # Ensure all tag vertices are included, regardless of connectivity with the odometry vertices
        for (vert_id, vert) in self.vertices.items():
            if vert.mode == VertexType.TAGPOINT:
                if vert_id not in vertices:
                    vertices[vert_id] = vert

        ret_graph = Graph(vertices, edges, weights=self._weights,
                          is_sparse_bundle_adjustment=self.is_sba, use_huber=self.use_huber,
                          huber_delta=self.huber_delta, damping_status=self.damping_status)
        return ret_graph

    def get_tag_verts(self) -> List[int]:
        """
        Returns:
            A list of the tag vertices' UIDs
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

    @staticmethod
    def as_graph(data_set: Union[Dict, UGDataSet], fixed_vertices: Union[VertexType, Tuple[VertexType]] = (),
                 prescaling_opt: PrescalingOptEnum = PrescalingOptEnum.USE_SBA) -> Graph:
        """Convert a dictionary decoded from JSON into a Graph object.

        Args:
            data_set: Unprocessed map data set. If a dict, it is decoded into a `UGDataSet` instance.
            fixed_vertices (tuple): Determines which vertex types to set to fixed. Dummy and Tagpoints are always fixed
             regardless of their presence in the tuple.
            prescaling_opt (PrescalingOptEnum): Selects which logical branches to use. If it is equal to
            `PrescalingOptEnum.USE_SBA`, then sparse bundle adjustment is used; otherwise, the outcome only differs
             between the remaining enum values by how the tag edge prescaling matrix is selected. Read the
             PrescalingOptEnum class documentation for more information.

        Returns:
            A graph derived from the input dictionary.

        Raises:
            ValueError: If prescaling_opt is an enum_value that is not handled.
            ValueError: If `data_set` is a dictionary and could not be decoded into a `UGDataSet` instance.
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
        if isinstance(data_set, dict):
            try:
                data_set = UGDataSet(**data_set)
            except pydantic.ValidationError as ve:
                raise ValueError(f"Could not parse the provided data set into a {UGDataSet.__name__} instance. "
                                 f"Diagnostic from pydantic validation:\n{ve.json(indent=2)}")

        # Pull out this equality from the enum (this equality is checked many times)
        use_sba = prescaling_opt == PrescalingOptEnum.USE_SBA

        # Used only if `use_sba` is false:
        tag_position_variances = None
        tag_orientation_variances = None
        tag_edge_prescaling = None
        previous_pose_matrix = None

        # Used only if `use_sba` is true:
        camera_intrinsics_for_tag: Union[np.ndarray, None] = None
        tag_corners = None
        true_3d_tag_center: Union[None, np.ndarray] = None
        true_3d_tag_points: Union[None, np.ndarray] = None
        tag_transform_estimates = None
        initialize_with_averages = None

        # Ensure that the fixed_vertices is always a tuple
        if isinstance(fixed_vertices, VertexType):
            fixed_vertices = (fixed_vertices,)

        if use_sba:
            true_3d_tag_points, true_3d_tag_center = make_sba_tag_arrays(ASSUMED_TAG_SIZE)

        frame_ids_to_timestamps = data_set.frame_ids_to_timestamps
        pose_matrices = data_set.pose_matrices
        odom_vertex_estimates = transform_matrix_to_vector(pose_matrices, invert=use_sba)

        # Commented out: a potential filter to apply to the tag detections (simply uncommenting would not do
        # anything; further refactoring would be required)
        # good_tag_detections = list(filter(lambda l: len(l) > 0,
        #                              [[tag_data for tag_data in tags_from_frame
        #                           if np.linalg.norm(np.asarray([tag_data['tag_pose'][i] for i in (3, 7, 11)])) < 1
        #                              and tag_data['tag_pose'][10] < 0.7] for tags_from_frame in dct['tag_data']]))

        tag_ids = data_set.tag_ids
        pose_ids = data_set.pose_ids
        if use_sba:
            camera_intrinsics_for_tag = data_set.camera_intrinsics_for_tag
            tag_corners = data_set.tag_corners
        else:
            tag_position_variances = data_set.tag_position_variances
            tag_orientation_variances = data_set.tag_orientation_variances

        tag_edge_measurements_matrix = data_set.tag_edge_measurements_matrix
        tag_edge_measurements = transform_matrix_to_vector(tag_edge_measurements_matrix)
        n_pose_ids = pose_ids.shape[0]

        if not use_sba:
            if prescaling_opt == PrescalingOptEnum.FULL_COV:
                tag_joint_covar_matrices = data_set.tag_joint_covar_matrices
                # TODO: for some reason we have missing measurements (all zeros). Throw those out
                tag_edge_prescaling = np.array([np.linalg.inv(covar[:-1, :-1]) if np.linalg.det(covar[:-1, :-1]) != 0
                                                else np.zeros((6, 6)) for covar in tag_joint_covar_matrices])
            elif prescaling_opt == PrescalingOptEnum.DIAG_COV:
                tag_edge_prescaling = 1. / np.hstack((tag_position_variances, tag_orientation_variances[:, :-1]))
            elif prescaling_opt == PrescalingOptEnum.ONES:
                tag_edge_prescaling = np.array([np.eye(6, 6)] * n_pose_ids)
            else:
                raise ValueError("{} is not yet handled".format(str(prescaling_opt)))

        unique_tag_ids = np.unique(tag_ids)
        tag_vertex_id_by_tag_id: Dict[int, int]
        if use_sba:
            tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(0, unique_tag_ids.size * 5, 5)))
        else:
            tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(unique_tag_ids.size)))
        tag_id_by_tag_vertex_id = dict(zip(tag_vertex_id_by_tag_id.values(), tag_vertex_id_by_tag_id.keys()))

        tag_corner_ids_by_tag_vertex_id: Dict[int, List[int]]
        if use_sba:
            tag_corner_ids_by_tag_vertex_id = dict(
                zip(tag_id_by_tag_vertex_id.keys(),
                    map(lambda tag_vertex_id_x: list(range(tag_vertex_id_x + 1, tag_vertex_id_x + 5)),
                        tag_id_by_tag_vertex_id.keys())))

        # Enable lookup of tags by the frame they appear in
        tag_vertex_id_and_index_by_frame_id: Dict[int, List[Tuple[int, int]]] = {}
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

        waypoint_names = data_set.waypoint_names
        unique_waypoint_names = np.unique(waypoint_names)
        num_unique_waypoint_names = unique_waypoint_names.size
        waypoint_edge_measurements_matrix = data_set.waypoint_edge_measurements_matrix
        waypoint_edge_measurements = transform_matrix_to_vector(waypoint_edge_measurements_matrix)
        waypoint_frame_ids = data_set.waypoint_frame_ids

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
                waypoint_index, [])
            waypoint_vertex_id_and_index_by_frame_id[waypoint_frame].append((waypoint_vertex_id, waypoint_index))

        num_tag_edges = edge_counter = 0
        vertices = {}
        edges = {}
        counted_tag_vertex_ids = set()
        counted_waypoint_vertex_ids = set()
        previous_vertex_uid = None
        first_odom_processed = False
        if use_sba:
            vertex_counter = unique_tag_ids.size * 5 + num_unique_waypoint_names
            # TODO: debug; this appears to be counterproductive
            initialize_with_averages = False
            tag_transform_estimates = defaultdict(lambda: [])
        else:
            vertex_counter = unique_tag_ids.size + num_unique_waypoint_names
        for i, odom_frame in enumerate(frame_ids_to_timestamps.keys()):
            current_odom_vertex_uid = vertex_counter
            vertices[current_odom_vertex_uid] = Vertex(
                mode=VertexType.ODOMETRY,
                estimate=odom_vertex_estimates[i],
                fixed=not first_odom_processed or VertexType.ODOMETRY in fixed_vertices,
                meta_data={'pose_id': odom_frame, 'timestamp': frame_ids_to_timestamps[odom_frame]})
            first_odom_processed = True
            vertex_counter += 1

            # Connect odom to tag vertex
            for tag_vertex_id, tag_index in tag_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
                if use_sba:
                    current_tag_transform_estimate = \
                        SE3Quat(np.hstack((true_3d_tag_center, [0, 0, 0, 1]))) * \
                        SE3Quat(tag_edge_measurements[tag_index]).inverse() * \
                        SE3Quat(vertices[current_odom_vertex_uid].estimate)

                    # keep track of estimates in case we want to average them to initialize the graph
                    tag_transform_estimates[tag_vertex_id].append(current_tag_transform_estimate)
                    if tag_vertex_id not in counted_tag_vertex_ids:
                        vertices[tag_vertex_id] = Vertex(
                            mode=VertexType.TAG,
                            estimate=current_tag_transform_estimate.to_vector(),
                            fixed=VertexType.TAG in fixed_vertices,
                            meta_data={'tag_id': tag_id_by_tag_vertex_id[tag_vertex_id]})

                        for idx, true_point_3d in enumerate(true_3d_tag_points):
                            # noinspection PyUnboundLocalVariable
                            vertices[tag_corner_ids_by_tag_vertex_id[tag_vertex_id][idx]] = Vertex(
                                mode=VertexType.TAGPOINT,
                                estimate=np.hstack((true_point_3d, [0, 0, 0, 1])),
                                fixed=True)
                        counted_tag_vertex_ids.add(tag_vertex_id)

                    # adjust the x-coordinates of the detections to account for differences in coordinate systems
                    # induced by the FLIP_Y_AND_Z_AXES
                    tag_corners[tag_index][::2] = 2 * camera_intrinsics_for_tag[tag_index][2] - \
                        tag_corners[tag_index][::2]

                    # Archive:
                    # for k, point in enumerate(true_3d_tag_points):
                    #     point_in_camera_frame = SE3Quat(tag_edge_measurements[tag_index]) * \
                    #                                     (point - np.array([0, 0, 1]))
                    #     cam = CameraParameters(camera_intrinsics_for_tag[tag_index][0],
                    #                            camera_intrinsics_for_tag[tag_index][2:], 0)
                    #     print("chi2", np.sum(np.square(tag_corners[tag_index][2*k : 2*k + 2] -
                    #                                    cam.cam_map(point_in_camera_frame))))

                    edges[edge_counter] = Edge(startuid=current_odom_vertex_uid, enduid=tag_vertex_id,
                                               corner_ids=tag_corner_ids_by_tag_vertex_id[tag_vertex_id],
                                               information_prescaling=None,
                                               camera_intrinsics=camera_intrinsics_for_tag[tag_index],
                                               measurement=tag_corners[tag_index],
                                               start_end=(vertices[current_odom_vertex_uid],
                                                          vertices[tag_vertex_id]))
                else:
                    if tag_vertex_id not in counted_tag_vertex_ids:
                        vertices[tag_vertex_id] = Vertex(
                            mode=VertexType.TAG,
                            estimate=transform_matrix_to_vector(pose_matrices[i].dot(
                                tag_edge_measurements_matrix[tag_index])),
                            fixed=VertexType.TAG in fixed_vertices,
                            meta_data={'tag_id': tag_id_by_tag_vertex_id[tag_vertex_id]})
                        counted_tag_vertex_ids.add(tag_vertex_id)
                    edges[edge_counter] = Edge(startuid=current_odom_vertex_uid, enduid=tag_vertex_id, corner_ids=None,
                                               information_prescaling=tag_edge_prescaling[tag_index],
                                               camera_intrinsics=None, measurement=tag_edge_measurements[tag_index],
                                               start_end=(vertices[current_odom_vertex_uid], vertices[tag_vertex_id]))

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
                edges[edge_counter] = Edge(startuid=current_odom_vertex_uid, enduid=waypoint_vertex_id, corner_ids=None,
                                           information_prescaling=None, camera_intrinsics=None,
                                           measurement=measurement_arg, start_end=(vertices[current_odom_vertex_uid],
                                                                                   vertices[waypoint_vertex_id]))
                edge_counter += 1

            # Connect odometry nodes
            if previous_vertex_uid:
                if use_sba:
                    measurement_arg = (SE3Quat(vertices[current_odom_vertex_uid].estimate) * SE3Quat(
                        vertices[previous_vertex_uid].estimate).inverse()).to_vector()
                else:
                    # TODO: might want to consider prescaling based on the magnitude of the change
                    measurement_arg = transform_matrix_to_vector(
                        np.linalg.inv(previous_pose_matrix).dot(pose_matrices[i]))

                edges[edge_counter] = Edge(startuid=previous_vertex_uid, enduid=current_odom_vertex_uid,
                                           corner_ids=None, information_prescaling=None, camera_intrinsics=None,
                                           measurement=measurement_arg, start_end=(vertices[previous_vertex_uid],
                                                                                   vertices[current_odom_vertex_uid]))
                edge_counter += 1

            # Connect gravity edge to odometry vertex
            # (use the second column of the inverted rotation matrix)
            gravity_edge_measurement_vector = np.concatenate((np.array([0.0, 1.0, 0.0]),
                                                              pose_matrices[i, 1, :-1]))
            edges[edge_counter] = Edge(
                startuid=current_odom_vertex_uid, enduid=None, information_prescaling=None,
                measurement=gravity_edge_measurement_vector, start_end=(vertices[current_odom_vertex_uid], None),
                camera_intrinsics=None, corner_ids=None)
            edge_counter += 1

            # Store uid so that it can be easily accessed for the next odometry-to-odometry edge addition
            previous_vertex_uid = current_odom_vertex_uid

            if not use_sba:
                previous_pose_matrix = pose_matrices[i]

        if use_sba and initialize_with_averages:
            for vertex_id, transforms in tag_transform_estimates.items():
                vertices[vertex_id].estimate = se3_quat_average(transforms).to_vector()

        # TODO: Huber delta should probably scale with pixels rather than error
        resulting_graph = Graph(vertices, edges, is_sparse_bundle_adjustment=use_sba,
                                use_huber=False, huber_delta=None, damping_status=False)
        return resulting_graph

    @staticmethod
    def transfer_vertex_estimates(graph_from: Graph, graph_to: Graph,
                                  filter_by: Optional[Set[VertexType]] = None) -> None:
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
