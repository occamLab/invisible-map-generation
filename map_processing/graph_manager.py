"""
Contains the GraphManager class. For the command line utility that makes use of it, see graph_manager_user.py. The
graph_optimization_analysis.ipynb notebook also makes use of this class.
"""

from __future__ import annotations

from typing import *

import numpy as np
from g2o import SE3Quat, SparseOptimizer
from geneticalgorithm import geneticalgorithm as ga

import map_processing
from map_processing import graph_opt_utils, PrescalingOptEnum
from . import graph_opt_utils, graph_opt_plot_utils, transform_utils, OCCAM_ROOM_TAGS_DICT
from .cache_manager import CacheManagerSingleton, MapInfo
from .graph import Graph
from .graph_vertex_edge_classes import Vertex, VertexType
from enum import Enum


class GraphManager:
    """Provides routines for the graph optimization capabilities provided by the Graph class.

    Organized as a class instead of a namespace of functions because the class and instance attributes configure the
    optimization routines.

    Class Attributes:
        _comparison_graph1_subgraph_weights: A list that contains a subset of the keys in
         _weights_dict; the keys identify the different weights vectors applied to the first subgraph when the
         compare_weights method is invoked.
        _weights_dict (Dict[str, np.ndarray]): Maps descriptive names of weight vectors to the corresponding weight
         vector, Higher values in the vector indicate greater noise (note: the uncertainty estimates of translation
         seem to be pretty over optimistic, hence the large correction here) for the orientation

    Attributes:
        _pso: TODO: documentation
        _selected_weights: TODO: documentation
        _firebase_manager: TODO: documentation
    """

    class WeightSpecifier(Enum):
        SENSIBLE_DEFAULT_WEIGHTS = 0
        TRUST_ODOM = 1
        TRUST_TAGS = 2
        GENETIC_RESULTS = 3
        BEST_SWEEP = 4
        COMPARISON_BASELINE = 5

    # Importance is set to e^{-weight}
    ordered_weights_dict_keys: List[str] = [
        "sensible_default_weights",
        "trust_odom",
        "trust_tags",
        "genetic_results",
        "best_sweep",
        "comparison_baseline"
    ]
    _default_dummy_weights = np.array([-1, 1e2, -1])
    _weights_dict: Dict[str, Dict[str, np.ndarray]] = {
        "sensible_default_weights": map_processing.graph_opt_utils.normalize_weights({
            'odometry': np.array([-6., -6., -6., -6., -6., -6.]),
            'tag_sba': np.array([18, 18]),
            'tag': np.array([18, 18, 0, 0, 0, 0]),
            'dummy': np.array([-1, 1e2, -1]),
        }),
        "trust_odom": map_processing.graph_opt_utils.normalize_weights({
            'odometry': np.array([-3., -3., -3., -3., -3., -3.]),
            'tag_sba': np.array([10.6, 10.6]),
            'tag': np.array([10.6, 10.6, 10.6, 10.6, 10.6, 10.6]),
            'dummy': _default_dummy_weights,
        }),
        "trust_tags": map_processing.graph_opt_utils.normalize_weights({
            'odometry': np.array([10, 10, 10, 10, 10, 10]),
            'tag_sba': np.array([-10.6, -10.6]),
            'tag': np.array([-10.6, -10.6, -10.6, -10.6, -10.6, -10.6]),
            'dummy': _default_dummy_weights,
        }),
        # Only used for SBA - no non-SBA tag weights
        "genetic_results": map_processing.graph_opt_utils.normalize_weights({
            'odometry': np.array([9.25, -7.96, -1.27, 7.71, -1.7, -0.08]),
            'tag_sba': np.array([9.91, 8.88]),
            'dummy': _default_dummy_weights,
        }),
        "best_sweep":
            map_processing.graph_opt_utils.weight_dict_from_array(np.exp(np.array([8.5, 10]))),
        "comparison_baseline": map_processing.graph_opt_utils.normalize_weights({
            'odometry': np.ones(6),
            'tag_sba': np.ones(2),
            'tag': np.ones(6),
            'dummy': _default_dummy_weights,
        })
    }
    _comparison_graph1_subgraph_weights: List[str] = [
        "sensible_default_weights",
        "trust_odom",
        "trust_tags",
        "genetic_results",
        "best_sweep"
    ]

    def __init__(self, weights_specifier: int, firebase_manager: CacheManagerSingleton, pso: int = 0):
        """Initializes GraphManager instance (only populates instance attributes)

        Args:
             weights_specifier: Used as the key to access the corresponding value in
              GraphManager._weights_dict (integer is mapped to the key with the GraphManager.ordered_weights_dict_keys
              list).
             firebase_manager: The firebase manager to use for reading from/to the cache.
             pso: Integer corresponding to the enum value in the PrescalingOptEnum enum which selects the
              type of prescaling weights used in non-SBA optimizations
        """

        self._pso = map_processing.PrescalingOptEnum(pso)
        self._selected_weights: str = GraphManager.ordered_weights_dict_keys[weights_specifier]
        self._firebase_manager = firebase_manager

    def process_maps(self, pattern: str, visualize: bool = True, upload: bool = False, compare: bool = False,
                     new_pso: Union[None, int] = None, new_weights_specifier: Union[None, int] = None,
                     fixed_vertices: Union[VertexType, Tuple[VertexType]] = (), obs_chi2_filter: float = -1) -> None:
        """Invokes optimization and plotting routines for any cached graphs matching the specified pattern.

        The _resolve_cache_dir method is first called, then the glob package is used to find matching files.
        Matching maps' json strings are loaded, parsed, and provided to the _process_map method. If an exception is
        raised in the process of loading a map or processing it, it is caught and its details are printed to the
        command line.

        Additionally, save the optimized json in <cache directory>/GraphManager._processed_upload_to.

        Args:
            pattern: Pattern to find matching cached graphs (which are stored as .json files. The cache
            pattern: Pattern to find matching cached graphs (which are stored as .json files. The cache
             directory (specified by the _cache_path attribute) is searched recursively
            visualize: Value passed as the visualize argument to the invocation of the _process_map method.
            upload: Value passed as the upload argument to the invocation of the _process_map method.
            compare: If true, run the routine for comparing graph optimization (invokes the compare_weights
             method).
            new_pso: If not None, then it overrides what was specified by the constructor's pso argument (and changes
             the corresponding _pso instance attribute).
            new_weights_specifier: If not none, then it overrides what was specified by the constructor's
             weights_specifier argument (and changes the corresponding _selected_weights instance attribute).
            fixed_vertices: Parameter to pass to the Graph.as_graph class method (see more there)
            obs_chi2_filter: Parameter to pass to the _optimize_graph method (see more there)
        """
        if new_pso is not None:
            self._pso = map_processing.PrescalingOptEnum(new_pso)
        if new_weights_specifier is not None:
            self._selected_weights: str = GraphManager.ordered_weights_dict_keys[new_weights_specifier]

        if len(pattern) == 0:
            print("Empty pattern provided; no maps will be processed")
            return

        matching_maps = self._firebase_manager.find_maps(pattern, search_only_unprocessed=False)
        if len(matching_maps) == 0:
            print("No matches for {} in recursive search of {}".format(pattern, self._firebase_manager.cache_path))
            return

        for map_info in matching_maps:
            if compare:
                if upload:
                    print("Warning: Ignoring True upload argument because comparing graphs")
                self.compare_weights(map_info, visualize)
                return

            graph_plot_title = None
            chi2_plot_title = None
            if visualize:
                graph_plot_title = "Optimization results for map: {}".format(map_info.map_name)
                chi2_plot_title = "Odom. node incident edges chi2 values for map: {}".format(map_info.map_name)

            processed_map_json = self.optimize_map_and_get_json(
                map_info,
                fixed_vertices=fixed_vertices,
                visualize=visualize,
                graph_plot_title=graph_plot_title,
                chi2_plot_title=chi2_plot_title,
                obs_chi2_filter=obs_chi2_filter
            )

            print("Processed map: {}".format(map_info.map_name))
            if upload:
                self._firebase_manager.upload(map_info, processed_map_json)
                print("Uploaded processed map: {}".format(map_info.map_name))

            self._firebase_manager.cache_map(self._firebase_manager.PROCESSED_UPLOAD_TO, map_info,
                                             processed_map_json)

    def compare_weights(self, map_info: MapInfo, visualize: bool = True, obs_chi2_filter: float = -1) -> None:
        """Invocation results in the weights comparison routine.

        Iterate through the different weight vectors (using the iter_weights variable) and, for each, do the
        following:
        1. Acquire two sub-graphs: one from the first half of the ordered odometry nodes (called g1sg) and one from the
           other half (called g2sg); note that g2sg is created from the Graph.as_graph class method with the
           fix_tag_vertices as True, whereas g1sg is created with fix_tag_vertices as False.
        2. Optimize the g1sg with the iter_weights, then transfer the estimated locations of its tag vertices to the
           g2sg. The assumption is that a majority - if not all - tag vertices are present in both sub-graphs; the
           number of instances where this is not true is tallied, and warning messages are printed accordingly.
        3. g2sg is then optimized with the self.selected_weights attribute selecting its weights (as opposed to
           g1sg which is optimized using the weights selected by iter_weights)
        The results of the comparison are then printed.

        Args:
            map_info: Map to use for weights comparison
            visualize: Used as the visualize argument for the _process_map method invocation.
            obs_chi2_filter: Passed to the _optimize_graph function (read more there)
        """
        results = "\n### Results ###\n\n"
        g1sg, g2sg = self.create_graphs_for_chi2_comparison(map_info.map_dct)

        missing_vertex_count = 0
        for graph1_sg_vert in g1sg.get_tag_verts():
            if not g2sg.vertices.__contains__(graph1_sg_vert):
                missing_vertex_count += 1
        if missing_vertex_count > 0:
            print("Warning: {} {} present in first subgraph that are not present in the second subgraph ("
                  "{} ignored)".format(missing_vertex_count, "vertices" if missing_vertex_count > 1 else
                                       "vertex", "these were" if missing_vertex_count > 1 else "this was"))

        deleted_vertex_count = 0
        for graph2_sg_vert in g2sg.get_tag_verts():
            if not g1sg.vertices.__contains__(graph2_sg_vert):
                g2sg.delete_tag_vertex(graph2_sg_vert)
                deleted_vertex_count += 1
        if deleted_vertex_count > 0:
            print("Warning: {} {} present in second subgraph that are not present in the first subgraph ("
                  "{} deleted from the second subgraph)"
                  .format(deleted_vertex_count, "vertices" if deleted_vertex_count > 1 else "vertex",
                          "these were" if deleted_vertex_count > 1 else "this was"))

        # After iterating through the different weights, the results of the comparison are printed.
        for iter_weights in GraphManager._comparison_graph1_subgraph_weights:
            print("\n-- Processing sub-graph without tags fixed, using weights set: {} --".format(iter_weights))
            if visualize:
                g1sg_plot_title = "Optimization results for 1st sub-graph from map: {} (weights = {})".format(
                    map_info.map_name, iter_weights)
                g1sg_chi2_plot_title = "Odom. node incident edges' chi2 values for 1st sub-graph from map: {} (" \
                                       "weights = {})".format(map_info.map_name, iter_weights)
            else:
                g1sg_plot_title = None
                g1sg_chi2_plot_title = None

            g1sg_tag_locs, g1sg_odom_locs, g1sg_waypoint_locs, g1sg_chi_sqr, g1sg_odom_adj_chi2, \
                g1sg_visible_tags_count = self._optimize_graph(
                    g1sg,
                    tune_weights=False,
                    visualize=visualize,
                    weights_key=iter_weights,
                    graph_plot_title=g1sg_plot_title,
                    chi2_plot_title=g1sg_chi2_plot_title,
                    obs_chi2_filter=obs_chi2_filter
                )
            processed_map_json_1 = map_processing.graph_opt_utils.make_processed_map_JSON(tag_locations=g1sg_tag_locs,
                                                                                          odom_locations=g1sg_odom_locs,
                                                                                          waypoint_locations=g1sg_waypoint_locs,
                                                                                          adj_chi2_arr=g1sg_odom_adj_chi2,
                                                                                          visible_tags_count=g1sg_visible_tags_count)
            del g1sg_tag_locs, g1sg_odom_locs, g1sg_waypoint_locs

            self._firebase_manager.cache_map(self._firebase_manager.PROCESSED_UPLOAD_TO, map_info,
                                             processed_map_json_1,
                                             "-comparison-subgraph-1-with_weights-set{}".format(iter_weights))
            del processed_map_json_1

            print("\n-- Processing sub-graph with tags fixed using weights set: {} --".format(self._selected_weights))

            # Get optimized tag vertices from g1sg and transfer their estimated positions to g2sg
            for graph1_sg_vert in g1sg.get_tag_verts():
                if g2sg.vertices.__contains__(graph1_sg_vert):
                    g2sg.vertices[graph1_sg_vert].estimate = g1sg.vertices[graph1_sg_vert].estimate

            if visualize:
                g2sg_plot_title = "Optimization results for 2nd sub-graph from map: {} (weights = {})".format(
                    map_info.map_name, self._selected_weights)
                g2sg_chi2_plot_title = "Odom. node incident edges chi2 values for 2nd sub-graph from  map: {} (" \
                                       "weights = {}))".format(map_info.map_name, self._selected_weights)
            else:
                g2sg_plot_title = None
                g2sg_chi2_plot_title = None

            g2sg_tag_locs, g2sg_odom_locs, g2sg_waypoint_locs, g2sg_chi_sqr, g2sg_odom_adj_chi2, \
                g2sg_visible_tags_count = self._optimize_graph(
                    g2sg,
                    tune_weights=False,
                    visualize=visualize,
                    weights_key=None,
                    graph_plot_title=g2sg_plot_title,
                    chi2_plot_title=g2sg_chi2_plot_title,
                    obs_chi2_filter=obs_chi2_filter
                )
            processed_map_json_2 = map_processing.graph_opt_utils.make_processed_map_JSON(tag_locations=g2sg_tag_locs,
                                                                                          odom_locations=g2sg_odom_locs,
                                                                                          waypoint_locations=g2sg_waypoint_locs,
                                                                                          adj_chi2_arr=g2sg_odom_adj_chi2,
                                                                                          visible_tags_count=g2sg_visible_tags_count)
            del g2sg_tag_locs, g2sg_odom_locs, g2sg_waypoint_locs

            self._firebase_manager.cache_map(self._firebase_manager.PROCESSED_UPLOAD_TO, map_info,
                                             processed_map_json_2,
                                             "-comparison-subgraph-2-with_weights-set{}".format(self._selected_weights))
            del processed_map_json_2

            results += "No fixed tags with weights set {}: chi2 = {}\n" \
                       "Subsequent optimization, fixed tags with weights set {}: chi2 = {}\n" \
                       "Abs(delta chi2): {}\n\n".format(iter_weights, g1sg_chi_sqr, self._selected_weights,
                                                        g2sg_chi_sqr, abs(g1sg_chi_sqr - g2sg_chi_sqr))
        print(results)

    def optimize_weights(self, map_json_path: str, verbose: bool = True) -> np.ndarray:
        """
        Determines the best weights to optimize a graph with

        Args:
            map_json_path: the path to the json containing the unprocessed map information
            verbose (bool): whether to provide output for the chi2 calculation

        Returns:
            A list of the best weights
        """
        map_dct = self._firebase_manager.map_info_from_path(map_json_path).map_dct
        graph = Graph.as_graph(map_dct)

        # Use a genetic algorithm
        model = ga(
            function=lambda x: self.get_ground_truth_from_graph(x, graph, OCCAM_ROOM_TAGS_DICT, verbose),
            dimension=8,
            variable_type='real',
            variable_boundaries=np.array([[-10, 10]] * 8),
            algorithm_parameters={
                'max_num_iteration': 2000,
                'population_size': 50,
                'mutation_probability': 0.1,
                'elit_ratio': 0.01,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type': 'uniform',
                'max_iteration_without_improv': None
            }
        )
        model.run()
        return model.report

    def sweep_weights(self, map_json_path: str, dimensions: int = 2, sweep: np.ndarray = np.arange(-10, 10, 0.2),
                      verbose: bool = False, visualize: bool = True) -> np.ndarray:
        """
        Sweeps a set of weights, returning the resulting chi2 values from each

        Args:
            map_json_path (str): the path to the json containing the map data to optimize on
            dimensions (int): the number of dimensions to sweep across
            sweep: TODO: documentation
            verbose (bool): whether to print out the chi2 values
            visualize (bool): whether to display the visualization plot. If not two_d, this will be ignored

        Returns:
            An ndarray, where each axis is a weight and each value is the resulting chi2. Note that the indexes will
                start at 0 with a step size of 1 regardless of actual bounds and step size
        """
        map_dct = self._firebase_manager.map_info_from_path(map_json_path).map_dct
        graph = Graph.as_graph(map_dct)
        metrics = self._sweep_weights(graph, sweep, dimensions, None, verbose=verbose)

        if dimensions == 2 and visualize:
            map_processing.graph_opt_plot_utils.plot_metrics(sweep, metrics, log_sweep=True, log_metric=True)
        if verbose:
            best_metric = metrics.min(initial=None)
            best_weights = [sweep[i[0]] for i in np.where(metrics == best_metric)]
            print(f'\nBEST METRIC: {best_weights}: {best_metric}')
        return metrics

    def get_optimized_graph_info(
            self, graph: Graph, 
            weights: Union[int, float, str, Dict[str, np.ndarray], np.ndarray, None] = None,
            verbose: bool = False,
            vertex_types=None
    ) -> Tuple[float, Dict[int, Vertex]]:
        """Finds the total chi2 and vertex locations of the optimized graph. Does not modify the graph.

        Returns:
             chi2: The sum of all edges' chi2 values.
             vertices: The optimized graph's vertices
        """
        if vertex_types is None:
            vertex_types = [VertexType.TAG]

        # Load in new weights and update graph
        optimizer = self.get_optimizer(graph, weights)

        # Find info
        chi2 = graph_opt_utils.sum_optimized_edges_chi2(optimizer, verbose=verbose)
        vertices = {uid: Vertex(graph.vertices[uid].mode, optimizer.vertex(uid).estimate().vector(),
                                graph.vertices[uid].fixed, graph.vertices[uid].meta_data)
                    for uid in optimizer.vertices() if vertex_types is None or graph.vertices[uid].mode in vertex_types}
        return chi2, vertices

    def get_chi2_from_subgraphs(
            self, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray]],
            subgraphs: Tuple[Graph, Graph],
            comparison_weights: Union[int, str, Dict[str, np.ndarray]] = "comparison_baseline",
            verbose: bool = False
    ) -> float:
        """TODO: documentation
        """
        self._weights_dict['variable'] = self._weights_to_dict(weights)
        _, vertices = self.get_optimized_graph_info(subgraphs[0], weights='variable', verbose=verbose)
        for uid, vertex in vertices.items():
            if subgraphs[1].vertices.__contains__(uid):
                subgraphs[1].vertices[uid].estimate = vertex.estimate
        return self.get_optimized_graph_info(
            subgraphs[1],
            weights=self.ordered_weights_dict_keys[comparison_weights] if
            isinstance(comparison_weights, int) else comparison_weights,
            verbose=verbose
        )[0]

    def get_chi2_by_edge_from_subgraphs(
            self, 
            weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray]],
            subgraphs: Union[Tuple[Graph, Graph], Dict],
            comparison_weights: Union[int, str, Dict[str, np.ndarray]] = "comparison_baseline", 
            verbose: bool = False
    ) -> Dict[str, Dict[str, float]]:
        """TODO: documentation
        """
        if isinstance(subgraphs, Dict):
            subgraphs = self.create_graphs_for_chi2_comparison(subgraphs)

        self._weights_dict['variable'] = self._weights_to_dict(weights)
        _, vertices = self.get_optimized_graph_info(subgraphs[0], weights='variable', verbose=verbose)
        for uid, vertex in vertices.items():
            if subgraphs[1].vertices.__contains__(uid):
                subgraphs[1].vertices[uid].estimate = vertex.estimate

        return subgraphs[1].get_chi2_by_edge_type(self.get_optimizer(subgraphs[1], comparison_weights), verbose=verbose)

    def get_ground_truth_from_graph(
            self, 
            weights: Union[str, Dict[str, np.ndarray], np.ndarray], 
            graph: Graph,
            ground_truth_tags: np.ndarray, 
            verbose: bool = False
    ) -> float:
        """TODO: documentation
        """
        if isinstance(weights, str):
            weight_name = weights
        else:
            weight_name = 'variable'
            self._weights_dict[weight_name] = weights if isinstance(weights, dict) else \
                map_processing.graph_opt_utils.weight_dict_from_array(weights)

        _, vertices = self.get_optimized_graph_info(graph, weights=weight_name)
        return GraphManager.get_ground_truth_from_optimized_tags(
            optimized_tags=vertices,
            ground_truth_tags=ground_truth_tags,
            verbose=verbose
        )

    @staticmethod
    def get_ground_truth_from_optimized_tags(
            optimized_tags: Dict[int, Vertex],
            ground_truth_tags: np.ndarray,
            verbose: bool = False,
            is_sba: bool = False
    ) -> float:
        """TODO: documentation
        """
        optimized_tag_verts = np.zeros((len(optimized_tags), 7))
        for vertex in optimized_tags.values():
            estimate = vertex.estimate

            # TODO: Double-check whether this check for SBA is being correctly handled
            if is_sba:
                optimized_tag_verts[vertex.meta_data["tag_id"]] = \
                    (SE3Quat([0, 0, -1, 0, 0, 0, 1]) * SE3Quat(estimate)).inverse().to_vector()
            else:
                optimized_tag_verts[vertex.meta_data["tag_id"]] = estimate

        # TODO: find intersection of tag ids and convert to numpy arrays after. Display warnings for tags not in
        #  intersection.

        metric = graph_opt_utils.ground_truth_metric(
            optimized_tag_verts=optimized_tag_verts,
            ground_truth_tags=ground_truth_tags,
            verbose=verbose
        )

        if verbose:
            print(metric)
        return metric

    def create_graphs_for_chi2_comparison(self, graph: Dict) -> Tuple[Graph, Graph]:
        """
        Creates then splits a graph in half, as required for weight comparison

        Specifically, this will create the graph based off the information in dct with the given prescaling option. It 
        will then exactly halve this graph's vertices into two graphs. The first will allows the tag vertices to vary, 
        while the second does not.

        Args:
            graph (Dict): A dictionary containing the unprocessed data to create the graph

        Returns:
            A tuple of 2 graphs, an even split of graph, as described above
        """
        graph1 = Graph.as_graph(graph, prescaling_opt=self._pso)
        graph2 = Graph.as_graph(graph, fixed_vertices=VertexType.TAG, prescaling_opt=self._pso)
        dummy_nodes = [0, 0]
        for vertex in graph1.vertices.values():
            if vertex.mode == VertexType.DUMMY:
                dummy_nodes[0] += 1
        for vertex in graph2.vertices.values():
            if vertex.mode == VertexType.DUMMY:
                dummy_nodes[1] += 1
        print(f'Dummy nodes: {dummy_nodes}')
        ordered_odom_edges = graph1.get_ordered_odometry_edges()[0]
        start_uid = graph1.edges[ordered_odom_edges[0]].startuid
        middle_uid_lower = graph1.edges[ordered_odom_edges[len(ordered_odom_edges) // 2]].startuid
        middle_uid_upper = graph1.edges[ordered_odom_edges[len(ordered_odom_edges) // 2]].enduid
        end_uid = graph1.edges[ordered_odom_edges[-1]].enduid

        print(f"start: {start_uid} mid_lower: {middle_uid_lower} mid_upper: {middle_uid_upper} end: {end_uid} total: "
              f"{len(graph1.vertices)}")

        g1sg = graph1.get_subgraph(start_vertex_uid=start_uid, end_vertex_uid=middle_uid_lower)
        g2sg = graph2.get_subgraph(start_vertex_uid=middle_uid_upper, end_vertex_uid=end_uid)

        return g1sg, g2sg

    def get_optimizer(self, graph: Graph, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray], None]) \
            -> SparseOptimizer:
        """
        Returns:
            The optimized g20.SparseOptimizer for the given graph with the given weights, or the graph's default weights
             if no weights are given.
        """
        if weights is not None:
            graph.set_weights(self._weights_to_dict(weights))
            graph.update_edge_information()

        optimizer = graph.graph_to_optimizer()
        optimizer.initialize_optimization()
        optimizer.optimize(1024)
        return optimizer

    def optimize_map_and_get_json(
            self,
            map_info: MapInfo,
            fixed_vertices: Union[VertexType, Tuple[VertexType]] = (),
            visualize: bool = False,
            graph_plot_title: str = None,
            chi2_plot_title: str = None,
            obs_chi2_filter: float = -1.0
    ) -> str:
        """Wrapper for graph optimization that takes in a MapInfo object and returns a json of the optimized map.
        """
        graph = Graph.as_graph(map_info.map_dct, fixed_vertices=fixed_vertices, prescaling_opt=self._pso)
        tag_locations, odom_locations, waypoint_locations, opt_chi2, adj_chi2, visible_tags_count = \
            self._optimize_graph(
                graph,
                tune_weights=False,
                visualize=visualize,
                weights_key=None,
                graph_plot_title=graph_plot_title,
                chi2_plot_title=chi2_plot_title,
                obs_chi2_filter=obs_chi2_filter
            )
        return map_processing.graph_opt_utils.make_processed_map_JSON(tag_locations=tag_locations,
                                                                      odom_locations=odom_locations,
                                                                      waypoint_locations=waypoint_locations,
                                                                      adj_chi2_arr=adj_chi2,
                                                                      visible_tags_count=visible_tags_count)

    # -- Private Methods --

    def _weights_to_dict(self, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray], None]):
        """
        Converts each representation of weights to a weight dictionary
        """
        if isinstance(weights, int):
            return self._weights_dict[self.ordered_weights_dict_keys[weights]]
        elif isinstance(weights, float):
            return map_processing.graph_opt_utils.weights_from_ratio(weights)
        elif isinstance(weights, str):
            return self._weights_dict[weights]
        elif isinstance(weights, np.ndarray):
            return map_processing.graph_opt_utils.weight_dict_from_array(weights)
        elif isinstance(weights, dict):
            return weights
        else:
            return self._selected_weights

    def _sweep_weights(self, graph: Graph, sweep: np.ndarray, dimensions: int,
                       metric_info: Union[np.ndarray, Tuple[Graph, Graph], None] = None, verbose: bool = False,
                       _cur_weights: np.ndarray = np.asarray([])) -> np.ndarray:
        """
        Sweeps the weights with the current chi2 algorithm evaluated on the given map

        Args:
            graph (Graph): the graph for weight comparison
            metric_info: Information for the metric calculation.
                ndarray: an array of SE3Quats representing the actual tag poses
                (Graph, Graph): for subgraph comparison
                None: use the chi2 of the optimized graph
            sweep (ndarray): a 1D array containing the values to sweep over
            dimensions (int): the number of dimensions to sweep over (1, 2 or 12)
            verbose (bool): whether to print the chi2 values
            _cur_weights (ndarray): the weights that are already set (do not set manually!)
        """
        # TODO: replace recursion with loop

        if metric_info is None:
            metric_to_use = lambda w, g, mi: self.get_optimized_graph_info(g, w)[0]
        elif isinstance(metric_info, tuple):
            metric_to_use = lambda w, g, mi: self.get_chi2_from_subgraphs(w, mi)
        elif isinstance(metric_info, np.ndarray):
            metric_to_use = lambda w, g, mi: self.get_ground_truth_from_graph(w, g, mi)
        else:
            raise Exception("metric_info is not a valid type")

        if dimensions == 1:
            metrics = np.asarray([])
            for weight in sweep:
                full_weights = np.append(_cur_weights, weight)
                try:
                    metric = metric_to_use(full_weights, graph, metric_info)
                except ValueError:
                    metric = -1
                if verbose:
                    print(f'{full_weights.tolist()}: {metric}')
                metrics = np.append(metrics, metric)
            return metrics
        else:
            metrics = np.asarray([])
            first_run = True
            for weight in sweep:
                if first_run:
                    metrics = self._sweep_weights(graph, sweep, dimensions - 1, metric_info, verbose,
                                                  np.append(_cur_weights, weight)).reshape(1, -1)
                    first_run = False
                else:
                    metrics = np.concatenate((metrics, self._sweep_weights(graph, sweep, dimensions - 1, metric_info,
                                                                           verbose, np.append(_cur_weights, weight))
                                              .reshape(1, -1)))
            return metrics

    def _optimize_graph(
            self, 
            graph: Graph, 
            tune_weights: bool = False, 
            visualize: bool = False, 
            weights_key: Union[None, str] = None, 
            obs_chi2_filter: float = -1, 
            graph_plot_title: Union[str, None] = None, 
            chi2_plot_title: Union[str, None] = None
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[List[Dict], np.ndarray], float, np.ndarray, np.ndarray]:
        """Optimizes the input graph.

        Process:
        1) Prepare the graph: set weights (optionally through expectation maximization), update edge information, and
        generate the sparse optimizer.
        2) Optimize the graph.
        3) [optional] Filter out tag observation edges with high chi2 values (see observation_chi2_filter parameter).
        4) [optional] Plot the optimization results

        Args:
            graph: A Graph instance to optimize.
            tune_weights: A boolean for whether expectation_maximization_once is called on the graph instance.
            visualize: A boolean for whether the visualize static method of this class is called.
            weights_key: Specifies the weight vector to set the weights attribute of the graph
             instance to from one of the weight vectors in GraphManager._weights_dict. If weights_key is None,
             then the weight vector corresponding to `self._selected_weights` is selected; otherwise, the weights_key
             selects the weight vector from the dictionary.
            obs_chi2_filter: Removes from the graph (stored in the `graph` instance attribute) observation edges above
             this many standard deviations from the mean observation edge chi2 value in the optimized graph. The graph
             optimization is then re-run with the modified graph. A negative value performs no filtering.
            graph_plot_title: Plot title argument to pass to the visualization routine for the graph visualizations.
            chi2_plot_title: Plot title argument to pass to the visualization routine for the chi2 plot.

        Returns:
            A tuple containing in the following order:
            - The numpy array of tag vertices from the optimized graph
            - The numpy array of odometry vertices from the optimized graph
            - The numpy array of waypoint vertices from the optimized graph
            - The total chi2 value of the optimized graph as returned by the optimize_graph method of the graph
              instance.
            - A numpy array where each element corresponds to the chi2 value for each odometry node; each chi2
              value is calculated as the sum of chi2 values of the (up to) two incident edges to the odometry node
              that connects it to (up to) two other odometry nodes.
            - A numpy array where each element corresponds to the number of visible tag vertices from the corresponding
              odometry vertices.
        """
        graph.set_weights(GraphManager._weights_dict[weights_key if weights_key is not None else 
                          self._selected_weights])

        if tune_weights:
            graph.expectation_maximization_once()
        else:
            # The expectation_maximization_once invocation calls generate_unoptimized_graph, so avoid repeating this if
            # tune_weights is true
            graph.generate_unoptimized_graph()

        # Load these weights into the graph
        graph.update_edge_information()

        # Acquire original_tag_verts for return value (not used elsewhere)
        starting_map = map_processing.graph_opt_utils.optimizer_to_map(graph.vertices, graph.unoptimized_graph,
                                                                       is_sparse_bundle_adjustment=self._pso == 0)
        original_tag_verts = transform_utils.locations_from_transforms(starting_map["tags"]) \
            if self._pso == map_processing.PrescalingOptEnum.USE_SBA else starting_map["tags"]
        del starting_map

        opt_chi2 = graph.optimize_graph()

        if obs_chi2_filter > 0:
            graph.filter_out_high_chi2_observation_edges(obs_chi2_filter)
            graph.optimize_graph()

        # Change vertex estimates based off the optimized graph
        graph.update_vertices_estimates()

        prior_map = map_processing.graph_opt_utils.optimizer_to_map_chi2(graph, graph.unoptimized_graph)
        resulting_map = map_processing.graph_opt_utils.optimizer_to_map_chi2(graph, graph.optimized_graph,
                                                                             is_sparse_bundle_adjustment=self._pso == 0)

        odom_chi2_adj_vec: np.ndarray = resulting_map["locationsAdjChi2"]
        visible_tags_count_vec: np.ndarray = resulting_map["visibleTagsCount"]

        locations = transform_utils.locations_from_transforms(resulting_map["locations"]) \
            if self._pso == map_processing.PrescalingOptEnum.USE_SBA else resulting_map["locations"]
        tag_verts = transform_utils.locations_from_transforms(resulting_map["tags"]) \
            if self._pso == map_processing.PrescalingOptEnum.USE_SBA else resulting_map["tags"]
        waypoint_verts = tuple(resulting_map["waypoints"])

        if visualize:
            prior_locations = transform_utils.locations_from_transforms(prior_map["locations"]) \
                if self._pso == map_processing.PrescalingOptEnum.USE_SBA else prior_map["locations"]
            tagpoint_positions = resulting_map["tagpoints"]
            graph_opt_plot_utils.plot_optimization_result(
                locations=locations,
                prior_locations=prior_locations,
                tag_verts=tag_verts,
                tagpoint_positions=tagpoint_positions,
                waypoint_verts=waypoint_verts,
                original_tag_verts=original_tag_verts,
                ground_truth_tags=None,
                plot_title=graph_plot_title,
                is_sba=self._pso == 0
            )
            graph_opt_plot_utils.plot_adj_chi2(resulting_map, chi2_plot_title)

        return tag_verts, locations, tuple(waypoint_verts), opt_chi2, odom_chi2_adj_vec, visible_tags_count_vec
