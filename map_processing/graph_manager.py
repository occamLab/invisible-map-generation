"""
Contains the GraphManager class. For the command line utility that makes use of it, see graph_manager_user.py. The
graph_optimization_analysis.ipynb notebook also makes use of this class.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional, Dict, List, Union, Tuple

import numpy as np
from geneticalgorithm import geneticalgorithm as ga

import map_processing
from map_processing import PrescalingOptEnum, VertexType
from . import graph_opt_utils, graph_opt_plot_utils
from .cache_manager import CacheManagerSingleton, MapInfo
from .data_models import OComputeInfParams, Weights, OConfig, GTDataSet, OResult, OSGPairResult
from .graph import Graph


class GraphManager:
    """Provides routines for the graph optimization capabilities provided by the `Graph` class.

    Stores optimization configuration parameters and mostly serves as a wrapper around core optimization capabilities
    afforded by the `Graph` class.

    Class Attributes:
        _comparison_graph1_subgraph_weights: A list that contains a subset of the keys in
         weights_dict; the keys identify the different weights vectors applied to the first subgraph when the
         compare_weights method is invoked.
        weights_dict (Dict[str, np.ndarray]): Maps descriptive names of weight vectors to the corresponding weight
         vector, Higher values in the vector indicate greater noise (note: the uncertainty estimates of translation
         seem to be pretty over optimistic, hence the large correction here) for the orientation

    Args:
        weights_specifier: Used as the key to access the corresponding value in `GraphManager.weights_dict` dictionary.
         Sets the selected_weights attribute with the value in the dictionary.
        cms: The firebase manager to use for reading from/to the cache.
        pso: Sets the default prescaling argument used for optimization.
        compute_inf_params: Passed down to the `Edge.compute_information` method to specify the edge
         information computation parameters.
        scale_by_edge_amount: Passed on to the `scale_by_edge_amount` argument of the `Graph.set_weights` method. If
             true, then the odom:tag ratio is scaled by the ratio of tag edges to odometry edges

    Attributes:
        pso: Default prescaling argument used for optimization.
        selected_weights: Weight object used for optimization by default.
        _cms: The firebase manager to use for reading from/to the cache.
        compute_inf_params: Passed down to the `Edge.compute_information` method to specify the edge
         information computation parameters.
        scale_by_edge_amount: Passed on to the `scale_by_edge_amount` argument of the `Graph.set_weights` method. If
             true, then the odom:tag ratio is scaled by the ratio of tag edges to odometry edges
    """

    class WeightSpecifier(Enum):
        SENSIBLE_DEFAULT_WEIGHTS = 0
        TRUST_ODOM = 1
        TRUST_TAGS = 2
        GENETIC_RESULTS = 3
        BEST_SWEEP = 4
        IDENTITY = 5
        VARIABLE = 6
        TRUST_GRAVITY = 7

    weights_dict: Dict[WeightSpecifier, Weights] = {
        WeightSpecifier.SENSIBLE_DEFAULT_WEIGHTS: Weights(odometry=np.exp(-np.array([-6., -6., -6., -6., -6., -6.])),
                                                          tag=np.exp(-np.array([18, 18, 0, 0, 0, 0])),
                                                          tag_sba=np.exp(-np.array([18, 18])), gravity=np.ones(3)),
        WeightSpecifier.TRUST_ODOM: Weights(odometry=np.exp(-np.array([-3., -3., -3., -3., -3., -3.])),
                                            tag=np.exp(-np.array([10.6, 10.6, 10.6, 10.6, 10.6, 10.6])),
                                            tag_sba=np.exp(-np.array([10.6, 10.6])), gravity=np.ones(3)),
        WeightSpecifier.TRUST_TAGS: Weights(odometry=np.exp(-np.array([10, 10, 10, 10, 10, 10])),
                                            tag=np.exp(-np.array([-10.6, -10.6, -10.6, -10.6, -10.6, -10.6])),
                                            tag_sba=np.exp(-np.array([-10.6, -10.6])), gravity=np.ones(3)),
        # Only used for SBA - no non-SBA tag weights
        WeightSpecifier.GENETIC_RESULTS: Weights(odometry=np.exp(-np.array([9.25, -7.96, -1.27, 7.71, -1.7, -0.08])),
                                                 tag_sba=np.exp(-np.array([9.91, 8.88])), gravity=np.ones(3)),
        WeightSpecifier.BEST_SWEEP: Weights.legacy_from_array(np.exp(np.array([8.5, 10]))),
        WeightSpecifier.IDENTITY: Weights(),
        WeightSpecifier.TRUST_GRAVITY: Weights(gravity=1 * np.ones(3))
    }
    _comparison_graph1_subgraph_weights: List[WeightSpecifier] = [
        WeightSpecifier.SENSIBLE_DEFAULT_WEIGHTS,
        WeightSpecifier.TRUST_ODOM,
        WeightSpecifier.TRUST_TAGS,
        WeightSpecifier.GENETIC_RESULTS,
        WeightSpecifier.BEST_SWEEP
    ]

    # -- Instance Methods: core functionality --

    def __init__(self, weights_specifier: WeightSpecifier, cms: CacheManagerSingleton,
                 pso: Union[int, PrescalingOptEnum] = 0,
                 compute_inf_params: Optional[OComputeInfParams] = None,
                 scale_by_edge_amount: bool = False):
        self.pso: PrescalingOptEnum = PrescalingOptEnum(pso) if isinstance(pso, int) else pso
        self.selected_weights: GraphManager.WeightSpecifier = weights_specifier
        self._cms = cms
        self.compute_inf_params: Optional[OComputeInfParams] = compute_inf_params
        self.scale_by_edge_amount = scale_by_edge_amount

    def holistic_optimize(
            self, map_info: MapInfo, visualize: bool = True, upload: bool = False,
            fixed_vertices: Union[VertexType, Tuple[VertexType]] = (), obs_chi2_filter: float = -1,
            compute_inf_params: Optional[OComputeInfParams] = None, gt_data: Optional[GTDataSet] = None,
            verbose: bool = False) -> OResult:
        """Optimizes graph, caches the result, and if specified by the arguments: upload the processed graph, visualize
        the graph optimization, and/or compute the ground truth metric.

        Args:
            map_info: Graph to process.
            visualize: Value passed as the `visualize` argument to the invocation of the _process_map method.
            upload: Value passed as the upload argument to the invocation of the _process_map method.
            fixed_vertices: Parameter to pass to the Graph.as_graph class method (see more there)
            obs_chi2_filter: Parameter to pass to the optimize_graph method (see more there)
            compute_inf_params: Passed down to the `Edge.compute_information` method to specify the edge
             information computation parameters.
            gt_data: If provided, used in the downstream optimization visualization and in ground truth metric
             computation.
            verbose: Toggles print statements within this function and passed as the verbose argument to called
             functions where applicable.

        Returns:
            An OResult object.
        """
        graph = Graph.as_graph(map_info.map_dct, fixed_vertices=fixed_vertices, prescaling_opt=self.pso)

        graph_plot_title = None
        chi2_plot_title = None
        if visualize:
            graph_plot_title = "Optimization results for map: {}".format(map_info.map_name)
            chi2_plot_title = "Odom. node incident edges chi2 values for map: {}".format(map_info.map_name)
        optimization_config = OConfig(
            is_sba=self.pso == PrescalingOptEnum.USE_SBA,
            weights=GraphManager.weights_dict[self.selected_weights], obs_chi2_filter=obs_chi2_filter,
            graph_plot_title=graph_plot_title, chi2_plot_title=chi2_plot_title, compute_inf_params=compute_inf_params,
            scale_by_edge_amount=self.scale_by_edge_amount)

        opt_result = GraphManager.optimize_graph(
            graph=graph, visualize=visualize, optimization_config=optimization_config, gt_data=gt_data)
        processed_map_json = map_processing.graph_opt_utils.make_processed_map_json(opt_result.map_opt)

        if verbose:
            print("Processed map: {}".format(map_info.map_name))

        if upload:
            self._cms.upload(map_info, processed_map_json)
            if verbose:
                print("Uploaded processed map: {}".format(map_info.map_name))

        self._cms.cache_map(self._cms.PROCESSED_UPLOAD_TO, map_info, processed_map_json)

        if gt_data is not None:
            opt_result.gt_metric_pre = self.ground_truth_metric_with_tag_id_intersection(
                optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_result.map_pre.tags),
                ground_truth_tags=gt_data.as_dict_of_se3_arrays, verbose=verbose)
            opt_result.gt_metric_opt = self.ground_truth_metric_with_tag_id_intersection(
                optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_result.map_opt.tags),
                ground_truth_tags=gt_data.as_dict_of_se3_arrays, verbose=verbose)
        return opt_result

    def holistic_optimize_batch_from_cache(
            self, pattern: str, visualize: bool = True, upload: bool = False, compare: bool = False,
            fixed_vertices: Union[VertexType, Tuple[VertexType]] = (), obs_chi2_filter: float = -1,
            search_only_unprocessed: bool = True, compute_inf_params: Optional[OComputeInfParams] = None) -> None:
        """Invokes `holistic_optimize` for any cached graphs matching the specified pattern.

        See more at the documentation for process_map.

        Args:
            pattern: Pattern to find matching cached graphs (which are stored as .json files). The cache
             directory (specified by the _cache_path attribute) via the find_maps instance method of the
             CacheManagerSingleton class.
            visualize: Value passed as the `visualize` argument to the invocation of the _process_map method.
            upload: Value passed as the upload argument to the invocation of the _process_map method.
            compare: If true, run the routine for comparing graph optimization (invokes the compare_weights
             method).
            fixed_vertices: Parameter to pass to the Graph.as_graph class method (see more there). Only applies if
             the compare argument is false.
            obs_chi2_filter: Parameter to pass to the optimize_graph method (see more there)
            search_only_unprocessed: Passed on as the argument to the find_maps method of the CacheManagerSingleton
             instance.
            compute_inf_params: Passed down to the `Edge.compute_information` method to specify the edge
             information computation parameters.
        """
        if len(pattern) == 0:
            print("Empty pattern provided; no maps will be processed")
            return

        matching_maps = self._cms.find_maps(pattern, search_only_unprocessed=search_only_unprocessed)
        if len(matching_maps) == 0:
            print("No matches for {} in recursive search of {}".format(pattern, self._cms.cache_path))
            return

        for map_info in matching_maps:
            if compare:
                if upload:
                    print("Warning: Ignoring True upload argument because comparing graphs")
                self.compare_weights(map_info, visualize)
                return
            else:
                self.holistic_optimize(
                    map_info=map_info, visualize=visualize, upload=upload, fixed_vertices=fixed_vertices,
                    obs_chi2_filter=obs_chi2_filter, compute_inf_params=compute_inf_params)

    def compare_weights(self, map_info: MapInfo, visualize: bool = True, obs_chi2_filter: float = -1) -> None:
        """Invocation results in the weight vectors comparison routine.

        TODO: Do more with results than simply print them
        TODO: Make use of the subgraph_pair_optimize method

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
            visualize: Used as the `visualize` argument for the _process_map method invocation.
            obs_chi2_filter: Passed to the optimize_graph function (read more there)
        """
        results = "\n### Results ###\n\n"
        g1sg, g2sg = self.create_graphs_for_chi2_comparison(map_info.map_dct)

        missing_vertex_count = 0
        for graph1_sg_vert in g1sg.get_tag_verts():
            if not g2sg.vertices.__contains__(graph1_sg_vert):
                missing_vertex_count += 1
        if missing_vertex_count > 0:
            print("Warning: {} {} present in first subgraph that are not present in the second subgraph ("
                  "{} ignored)".format(missing_vertex_count, "vertices" if missing_vertex_count > 1 else "vertex",
                                       "these were" if missing_vertex_count > 1 else "this was"))

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

            optimization_config_1 = OConfig(
                is_sba=self.pso == PrescalingOptEnum.USE_SBA,
                weights=GraphManager.weights_dict[iter_weights], obs_chi2_filter=obs_chi2_filter,
                graph_plot_title=g1sg_plot_title, chi2_plot_title=g1sg_chi2_plot_title,
                scale_by_edge_amount=self.scale_by_edge_amount
            )
            g1sg_opt_result = GraphManager.optimize_graph(
                graph=g1sg, optimization_config=optimization_config_1, visualize=visualize)
            processed_map_json_1 = map_processing.graph_opt_utils.make_processed_map_json(g1sg_opt_result.map_opt)

            self._cms.cache_map(self._cms.PROCESSED_UPLOAD_TO, map_info,
                                processed_map_json_1, "-comparison-subgraph-1-with_weights-set{}".format(iter_weights))
            del processed_map_json_1

            print("\n-- Processing sub-graph with tags fixed using weights set: {} --".format(self.selected_weights))

            # Get optimized tag vertices from g1sg and transfer their estimated positions to g2sg
            Graph.transfer_vertex_estimates(g1sg, g2sg, filter_by={VertexType.TAG, })

            if visualize:
                g2sg_plot_title = "Optimization results for 2nd sub-graph from map: {} (weights = {})".format(
                    map_info.map_name, self.selected_weights)
                g2sg_chi2_plot_title = "Odom. node incident edges chi2 values for 2nd sub-graph from  map: {} (" \
                                       "weights = {}))".format(map_info.map_name, self.selected_weights)
            else:
                g2sg_plot_title = None
                g2sg_chi2_plot_title = None

            optimization_config_2 = OConfig(
                is_sba=self.pso == PrescalingOptEnum.USE_SBA, obs_chi2_filter=obs_chi2_filter,
                graph_plot_title=g2sg_plot_title, chi2_plot_title=g2sg_chi2_plot_title,
                scale_by_edge_amount=self.scale_by_edge_amount)
            g2sg_opt_result = GraphManager.optimize_graph(
                graph=g2sg, visualize=visualize, optimization_config=optimization_config_2)
            processed_map_json_2 = map_processing.graph_opt_utils.make_processed_map_json(g2sg_opt_result.map_opt)

            self._cms.cache_map(
                self._cms.PROCESSED_UPLOAD_TO, map_info, processed_map_json_2,
                "-comparison-subgraph-2-with_weights-set{}".format(self.selected_weights)
            )
            del processed_map_json_2

            results += \
                "No fixed tags with weights set {}: chi2 = {}\nSubsequent optimization, fixed tags with weights set " \
                "{}: chi2 = {}\nAbs(delta chi2): {}\n\n".format(
                    iter_weights, g1sg_opt_result.chi2s.chi2_all_after, self.selected_weights,
                    g2sg_opt_result.chi2s.chi2_all_after,
                    abs(g1sg_opt_result.chi2s.chi2_all_after - g2sg_opt_result.chi2s.chi2_all_after))
        print(results)

    # noinspection PyUnreachableCode,PyUnusedLocal
    def optimize_weights(self, map_json_path: str, verbose: bool = True) -> np.ndarray:
        """
        Determines the best weights to optimize a graph with

        Args:
            map_json_path: the path to the json containing the unprocessed map information
            verbose (bool): whether to provide output for the chi2 calculation

        Returns:
            A list of the best weights
        """
        raise NotImplementedError("This function has not been updated to work with the new way that ground truth data"
                                  "is being handled")
        map_dct = self._cms.map_info_from_path(map_json_path).map_dct
        graph = Graph.as_graph(map_dct)

        # Use a genetic algorithm
        model = ga(
            function=lambda x: 0.0,  # TODO: replace this placeholder with invocation of the ground truth metric
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

    def create_graphs_for_chi2_comparison(self, graph: Dict) -> Tuple[Graph, Graph]:
        """
        Creates then splits a graph in half, as required for weight comparison

        Specifically, this will create the graph based off the information in dct with the given prescaling option. It 
        will then exactly halve this graph's vertices into two graphs. The first will allow the tag vertices to vary,
        while the second does not.

        Args:
            graph (Dict): A dictionary containing the unprocessed data to create the graph

        Returns:
            A tuple of 2 graphs, an even split of graph, as described above
        """
        graph1 = Graph.as_graph(graph, prescaling_opt=self.pso)
        graph2 = Graph.as_graph(graph, fixed_vertices=VertexType.TAG, prescaling_opt=self.pso)

        ordered_odom_edges = graph1.get_ordered_odometry_edges()[0]
        start_uid = graph1.edges[ordered_odom_edges[0]].startuid
        middle_uid_lower = graph1.edges[ordered_odom_edges[len(ordered_odom_edges) // 2]].startuid
        middle_uid_upper = graph1.edges[ordered_odom_edges[len(ordered_odom_edges) // 2]].enduid
        end_uid = graph1.edges[ordered_odom_edges[-1]].enduid

        # print(f"start: {start_uid} mid_lower: {middle_uid_lower} mid_upper: {middle_uid_upper} end: {end_uid} total: "
        #       f"{len(graph1.vertices)}")

        g1sg = graph1.get_subgraph(start_vertex_uid=start_uid, end_vertex_uid=middle_uid_lower)
        g2sg = graph2.get_subgraph(start_vertex_uid=middle_uid_upper, end_vertex_uid=end_uid)
        return g1sg, g2sg

    @staticmethod
    def optimize_graph(graph: Graph, optimization_config: OConfig, visualize: bool = False,
                       gt_data: Optional[GTDataSet] = None, verbose: bool = False) -> OResult:
        """Optimizes the input graph.

        Notes:
            Process used:
            1. Prepare the graph: set weights (optionally through expectation maximization), update edge information,
             and generate the sparse optimizer.
            2. Optimize the graph.
            3. [optional] Filter out tag observation edges with high chi2 values (see observation_chi2_filter
             parameter).
            4. [optional] Plot the optimization results

        Args:
            graph: A Graph instance to optimize.
            visualize: A boolean for whether the `visualize` static method of this class is called.
            optimization_config: Configures the optimization.
            gt_data: If provided, only used for the downstream optimization visualization.
            verbose: Passed to Graph.optimize_graph as the verbose argument.

        Returns:
            A tuple containing in the following order: (1) The total chi2 value of the optimized graph as returned by
             the optimize_graph method of the graph instance. (2) The dictionary returned by
             `map_processing.graph_opt_utils.optimizer_to_map_chi2` when called on the optimized graph. (3) The
             dictionary returned by `map_processing.graph_opt_utils.optimizer_to_map_chi2` when called on the
             graph before optimization.
        """
        is_sba = optimization_config.is_sba
        graph.set_weights(weights=optimization_config.weights,
                          scale_by_edge_amount=optimization_config.scale_by_edge_amount)
        graph.update_edge_information(compute_inf_params=optimization_config.compute_inf_params)

        graph.generate_unoptimized_graph()
        before_opt_map = map_processing.graph_opt_utils.optimizer_to_map_chi2(graph, graph.unoptimized_graph,
                                                                              is_sba=is_sba)
        chi2_values = graph.optimize_graph(verbose=verbose)
        if optimization_config.obs_chi2_filter > 0:
            graph.filter_out_high_chi2_observation_edges(optimization_config.obs_chi2_filter)
            graph.optimize_graph()

        # Change vertex estimates based off the optimized graph
        graph.update_vertices_estimates_from_optimized_graph()
        opt_result_map = map_processing.graph_opt_utils.optimizer_to_map_chi2(graph, graph.optimized_graph,
                                                                          is_sba=is_sba)

        if visualize:
            graph_opt_plot_utils.plot_optimization_result(
                opt_odometry=opt_result_map.locations,
                orig_odometry=before_opt_map.locations,
                opt_tag_verts=opt_result_map.tags,
                opt_tag_corners=opt_result_map.tagpoints,
                opt_waypoint_verts=(opt_result_map.waypoints_metadata, opt_result_map.waypoints_arr),
                orig_tag_verts=before_opt_map.tags,
                ground_truth_tags=gt_data.sorted_poses_as_se3quat_list if gt_data is not None else None,
                plot_title=optimization_config.graph_plot_title,
            )
            graph_opt_plot_utils.plot_adj_chi2(opt_result_map, optimization_config.chi2_plot_title)
        return OResult(
            oconfig=optimization_config,
            map_pre=before_opt_map,
            map_opt=opt_result_map,
            chi2s=chi2_values
        )

    def optimize_from_weights(self, graph: Graph, weights: Optional[Weights] = None, verbose: bool = False) -> OResult:
        """Wrapper to optimize_graph that self-constructs an optimization configuration object based on the members of
        this instance.
        """
        optimization_config = OConfig(
            is_sba=self.pso == PrescalingOptEnum.USE_SBA,
            scale_by_edge_amount=self.scale_by_edge_amount,
            weights=weights
        )
        return GraphManager.optimize_graph(graph=graph, optimization_config=optimization_config)

    def subgraph_pair_optimize(
            self, subgraph_0_weights: Weights, subgraphs: Union[Tuple[Graph, Graph], Dict],
            subgraph_1_weights: Weights = WeightSpecifier.IDENTITY, verbose: bool = False) -> OSGPairResult:
        """Perform the subgraph pair optimization routine and return the difference between the first subgraph's chi2
        metric value and the second subgraph's.
        
        Notes:
            Tag vertex estimates are transferred between the two graphs' optimizations.
        
        Args:
             subgraph_0_weights: Weights to optimize the first subgraph with
             subgraphs: If a tuple of graphs, then the assumption is that they have been prepared using the 
              create_graphs_for_chi2_comparison instance method; if a dictionary, then create_graphs_for_chi2_comparison
              is invoked with the dictionary as its argument to construct the two subgraphs. Read more of that method's
              documentation to understand this process.
             subgraph_1_weights: Weights to optimize the second subgraph with.
             verbose: Boolean for whether diagnostics are printed.
         
         Returns:
             Difference of the subgraph's chi2 metrics.
        """
        if isinstance(subgraphs, Dict):
            subgraphs = self.create_graphs_for_chi2_comparison(subgraphs)
        o1_config = OConfig(
            is_sba=self.pso == PrescalingOptEnum.USE_SBA,
            scale_by_edge_amount=self.scale_by_edge_amount,
            weights=subgraph_0_weights)
        opt1_result = GraphManager.optimize_graph(graph=subgraphs[0], optimization_config=o1_config, verbose=verbose)

        o2_config = OConfig(
            is_sba=self.pso == PrescalingOptEnum.USE_SBA,
            scale_by_edge_amount=self.scale_by_edge_amount,
            weights=subgraph_1_weights)
        Graph.transfer_vertex_estimates(subgraphs[0], subgraphs[1], filter_by={VertexType.TAG, })
        opt2_result = GraphManager.optimize_graph(graph=subgraphs[1], optimization_config=o2_config, verbose=verbose)

        return OSGPairResult(
            sg1_result=opt1_result,
            sg2_result=opt2_result)

    def optimize_and_get_ground_truth_error_metric(
            self, weights: Optional[Weights], graph: Graph, ground_truth_tags: Dict[int, np.ndarray],
            verbose: bool = False, visualize: bool = False, obs_chi2_filter: float = -1,
            graph_plot_title: Optional[str] = None, chi2_plot_title: Optional[str] = None) -> OResult:
        """Light wrapper for the optimize_graph instance method and ground_truth_metric_with_tag_id_intersection method.
        """
        optimization_config = OConfig(
            is_sba=self.pso == PrescalingOptEnum.USE_SBA, scale_by_edge_amount=self.scale_by_edge_amount,
            weights=weights, obs_chi2_filter=obs_chi2_filter, graph_plot_title=graph_plot_title,
            chi2_plot_title=chi2_plot_title,
        )
        opt_result = GraphManager.optimize_graph(graph=graph, visualize=visualize,
                                                 optimization_config=optimization_config)
        opt_result.gt_metric_opt = GraphManager.ground_truth_metric_with_tag_id_intersection(
            optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_result.map_opt.tags),
            ground_truth_tags=ground_truth_tags,
            verbose=verbose)
        return opt_result

    # -- Static Methods --

    @staticmethod
    def tag_pose_array_with_metadata_to_map(tag_array_with_metadata: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Args:
            tag_array_with_metadata: nx8 array of n poses (as 7-element vectors) where the 8th element in each row is
             interpreted as the tag id.

        Returns:
            A dictionary mapping tag ids to their poses.
        """
        ret: Dict[int, np.ndarray] = {}
        for i in range(tag_array_with_metadata.shape[0]):
            ret[tag_array_with_metadata[i, -1]] = tag_array_with_metadata[i, :-1]
        return ret

    @staticmethod
    def ground_truth_metric_with_tag_id_intersection(
            optimized_tags: Dict[int, np.ndarray], ground_truth_tags: Dict[int, np.ndarray], verbose: bool = False
    ) -> float:
        """Use the intersection of the two provided tag dictionaries as input to the graph_opt_utils.ground_truth_metric
        function. Includes handling of the SBA case in which the optimized tags' estimates need to be translated and
        then inverted.

        Args:
            optimized_tags: Dictionary mapping tag IDs to their pose-containing Vertex objects.
            ground_truth_tags: Dictionary mapping tag IDs to their poses (as length-7 vectors).
            verbose: Boolean for whether to print diagnostic messages.

        Returns:
            Value returned by the graph_opt_utils.ground_truth_metric function (see more there).
        """
        tag_id_intersection = set(optimized_tags.keys()).intersection(set(ground_truth_tags.keys()))
        optimized_tags_poses_intersection = np.zeros((len(tag_id_intersection), 7))
        gt_tags_poses_intersection = np.zeros((len(tag_id_intersection), 7))
        for i, tag_id in enumerate(sorted(tag_id_intersection)):
            optimized_vertex_estimate = optimized_tags[tag_id]
            optimized_tags_poses_intersection[i] = optimized_vertex_estimate
            gt_tags_poses_intersection[i] = ground_truth_tags[tag_id]

        metric = graph_opt_utils.ground_truth_metric(
            optimized_tag_verts=optimized_tags_poses_intersection,
            ground_truth_tags=gt_tags_poses_intersection,
            verbose=verbose
        )

        if verbose:
            print(metric)
        return metric
