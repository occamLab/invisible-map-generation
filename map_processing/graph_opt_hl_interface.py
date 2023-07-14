"""
Provides a high-level interface for graph optimization capabilities provided by the `Graph` class and functions from
other modules in this package.

Notes:
    For the command line utility that makes use of these features, see
    `run_scripts/optimize_graphs_and_manage_cache.py`.
"""

from enum import Enum
from typing import Optional, Dict, Union, Tuple, Set

import numpy as np

# from geneticalgorithm import geneticalgorithm as ga

from . import PrescalingOptEnum, VertexType, graph_opt_utils, graph_opt_plot_utils
from .cache_manager import CacheManagerSingleton, MapInfo
from .data_models import Weights, OConfig, GTDataSet, OResult, OSGPairResult
from .graph import Graph


class WeightSpecifier(Enum):
    SENSIBLE_DEFAULT_WEIGHTS = 0
    TRUST_ODOM = 1
    TRUST_TAGS = 2
    GENETIC_RESULTS = 3
    BEST_SWEEP = 4
    IDENTITY = 5
    VARIABLE = 6
    TRUST_GRAVITY = 7


# Testing dicts for different weight defaults
WEIGHTS_DICT: Dict[WeightSpecifier, Weights] = {  # TODO: revisit these
    WeightSpecifier.SENSIBLE_DEFAULT_WEIGHTS: Weights(
        orig_odometry=np.array([-6.0, -6.0, -6.0, -6.0, -6.0, -6.0]),
        orig_tag=np.array([18, 18, 0, 0, 0, 0]),
        orig_tag_sba=np.array([18, 18]),
    ),
    WeightSpecifier.TRUST_ODOM: Weights(
        orig_odometry=np.array([1.0, 1.0, 1.0, 0.01, 0.01, 0.01]),
        orig_tag=np.array([10.6, 10.6, 10.6, 10.6, 10.6, 10.6]),
        orig_tag_sba=np.array([10**7, 10**7]),
    ),
    WeightSpecifier.TRUST_TAGS: Weights(
        orig_odometry=np.array([10, 10, 10, 10, 10, 10]),
        orig_tag=np.array([-10.6, -10.6, -10.6, -10.6, -10.6, -10.6]),
        orig_tag_sba=np.array([-10.6, -10.6]),
    ),
    WeightSpecifier.GENETIC_RESULTS: Weights(  # Only used for SBA - no non-SBA tag weights
        orig_odometry=np.exp(-np.array([9.25, -7.96, -1.27, 7.71, -1.7, -0.08])),
        orig_tag_sba=np.exp(-np.array([9.91, 8.88])),
        orig_gravity=np.ones(3),
    ),
    WeightSpecifier.BEST_SWEEP: Weights.legacy_from_array(np.exp(np.array([8.5, 10]))),
    WeightSpecifier.IDENTITY: Weights(),
    WeightSpecifier.TRUST_GRAVITY: Weights(orig_gravity=1 * np.ones(3)),
}


def holistic_optimize(
    map_info: MapInfo,
    pso: PrescalingOptEnum,
    oconfig: OConfig,
    fixed_vertices: Optional[Union[VertexType, Set[VertexType]]] = None,
    cms: Optional[CacheManagerSingleton] = None,
    gt_data: Optional[GTDataSet] = None,
    verbose: bool = False,
    visualize: bool = True,
    compare: bool = False,
    upload: bool = False,
    abs_anchor_pos: bool = False,
    generate_plot_titles: bool = True,
) -> Union[OResult, OSGPairResult]:
    """Optimizes graph, caches the result, and if specified by the arguments: upload the processed graph, visualize
    the graph optimization, and/or compute the ground truth metric.

    Args:
        map_info: Graph to process.
        pso: determines if you're using a sparse optimizer or not, if no, takes in a covariance matrix
        oconfig: the "optimization configuration" that dictates the mechanics and cosmetic qualities of the optimization graph. CONTAINS THE WEIGHTS
        fixed_vertices: Parameter to pass to the Graph.as_graph class method (see more there)
        cms: Handles downloading and uploading to/from Firebase.
        gt_data: If provided, used in the downstream optimization visualization and in ground truth metric
         computation.
        verbose: Toggles print statements within this function and passed as the verbose argument to called
        visualize: Value passed as the `visualize` argument to the invocation of the _process_map method.
         functions where applicable.
        compare: Invokes the subgraph graph comparison routine (see notes section for more information).
        upload: Value passed as the upload argument to the invocation of the _process_map method.
        generate_plot_titles: Generates a plot title from a template.

    Notes:
        The subgraph comparison routine is as follows:

        Iterate through the different weight vectors (using the iter_weights variable) and, for each, do the
        following:
        1. Acquire two sub-graphs: one from the first half of the ordered odometry nodes (called g1sg) and one from
           the other half (called g2sg); note that g2sg is created from the Graph.as_graph class method with the
           fix_tag_vertices as True, whereas g1sg is created with fix_tag_vertices as False.
        2. Optimize the g1sg with the iter_weights, then transfer the estimated locations of its tag vertices to the
           g2sg. The assumption is that a majority - if not all - tag vertices are present in both sub-graphs; the
           number of instances where this is not true is tallied, and warning messages are printed accordingly.
        3. g2sg is then optimized with the self.selected_weights attribute selecting its weights (as opposed to
           g1sg which is optimized using the weights selected by iter_weights)

    Returns:
        An OResult object if compare is false, and an OSGPairResult result if compare is true.

    Raises:
        ValueError - If `upload` and `compare` are both True.
        ValueError - If `upload` is True and `cms` is None.
        ValueError - If `cms` has not had its credentials set, but `upload` is True.
    """
    if upload and compare:
        ValueError("Invalid arguments: `upload` and `compare` are both True.")
    elif upload and cms is None:
        ValueError("Invalid arguments: `upload` cannot be True while `cms` is None.")
    elif upload and not cms.were_credentials_set:
        ValueError(
            "Invalid arguments: `upload` cannot be True while `cms` has not been provided credentials."
        )

    gt_data_as_dict_of_se3_arrays = None
    if gt_data is not None:
        gt_data_as_dict_of_se3_arrays = gt_data.as_dict_of_se3_arrays

    graph = Graph.as_graph(
        map_info.map_dct, fixed_vertices=fixed_vertices, prescaling_opt=pso, abs_anchor_pos=abs_anchor_pos
    )
    if generate_plot_titles:
        oconfig.graph_plot_title = "Optimization results for map: {}".format(
            map_info.map_name
        )
        oconfig.chi2_plot_title = (
            "Odom. node incident edges chi2 values for map: {}".format(
                map_info.map_name
            )
        )

    if compare:
        g1sg, g2sg = create_subgraphs_for_subgraph_chi2_comparison(
            map_info.map_dct, pso=pso
        )
        osg_pair_result = subgraph_pair_optimize(
            subgraphs=(g1sg, g2sg), oconfig_1=oconfig, oconfig_2=oconfig, pso=pso
        )

        # Find metrics for the two OResults to be compared
        for oresult in [osg_pair_result.sg1_oresult, osg_pair_result.sg2_oresult]:
            if gt_data_as_dict_of_se3_arrays is not None:
                (
                    gt_metric_pre,
                    max_diff_pre,
                    max_diff_idx_pre,
                    _,
                    _,
                    _,
                ) = ground_truth_metric_with_tag_id_intersection(
                    optimized_tags=tag_pose_array_with_metadata_to_map(
                        oresult.map_pre.tags
                    ),
                    ground_truth_tags=gt_data_as_dict_of_se3_arrays,
                )
                oresult.gt_metric_pre = gt_metric_pre
                oresult.max_pre = max_diff_pre
                oresult.max_idx_pre = max_diff_idx_pre
                (
                    gt_metric,
                    max_diff,
                    max_diff_idx,
                    _,
                    _,
                    gt_per_anchor,
                ) = ground_truth_metric_with_tag_id_intersection(
                    optimized_tags=tag_pose_array_with_metadata_to_map(
                        oresult.map_opt.tags
                    ),
                    ground_truth_tags=gt_data_as_dict_of_se3_arrays,
                )
                oresult.gt_metric_opt = gt_metric
                oresult.max_opt = max_diff
                oresult.max_idx_opt = max_diff_idx
                oresult.gt_per_anchor_tag_opt = gt_per_anchor

        return osg_pair_result

    opt_result = optimize_graph(
        graph=graph, oconfig=oconfig, visualize=visualize, gt_data=gt_data
    )
    processed_map_json = graph_opt_utils.make_processed_map_json(
        opt_result.map_opt, calculate_intersections=upload
    )

    if verbose:
        print(f"Optimized {map_info.map_name}.\nResulting chi2 metrics:")
        print(opt_result.fitness_metrics.repr_as_list())

    if gt_data_as_dict_of_se3_arrays is not None:
        # Find metrics translational
        intersection = ground_truth_metric_with_tag_id_intersection(
            optimized_tags=tag_pose_array_with_metadata_to_map(opt_result.map_pre.tags),
            ground_truth_tags=gt_data_as_dict_of_se3_arrays,
        )
        (
            gt_metric_pre,
            max_diff_pre,
            max_diff_idx_pre,
            _,
            _,
            _,
        ) = intersection
        opt_result.gt_metric_pre = gt_metric_pre
        opt_result.max_pre = max_diff_pre
        opt_result.max_idx_pre = max_diff_idx_pre

        (
            gt_metric,
            max_diff,
            max_diff_idx,
            _,
            _,
            gt_per_anchor,
        ) = ground_truth_metric_with_tag_id_intersection(
            optimized_tags=tag_pose_array_with_metadata_to_map(opt_result.map_opt.tags),
            ground_truth_tags=gt_data_as_dict_of_se3_arrays,
        )
        opt_result.gt_metric_opt = gt_metric
        opt_result.max_opt = max_diff
        opt_result.max_idx_opt = max_diff_idx
        opt_result.gt_per_anchor_tag_opt = gt_per_anchor

        # Print results
        if verbose:
            print(f"Pre-optimization metric: {opt_result.gt_metric_pre:.3f}")
            print(
                f"Ground truth metric: {opt_result.gt_metric_opt:.3f} ("
                f"delta of {opt_result.gt_metric_opt - opt_result.gt_metric_pre:.3f} from pre-optimization)"
            )
            print(
                f"Maximum difference metric (pre-optimized): {opt_result.max_pre:.3f} (tag id: {opt_result.max_idx_pre})"
            )
            print(
                f"Maximum difference metric (optimized): {opt_result.max_opt:.3f} (tag id: {opt_result.max_idx_opt})"
            )

    CacheManagerSingleton.cache_map(
        CacheManagerSingleton.PROCESSED_UPLOAD_TO, map_info, processed_map_json
    )
    if upload:
        cms.upload(map_info, processed_map_json, verbose=verbose)
    return opt_result


# noinspection PyUnreachableCode,PyUnusedLocal
def optimize_weights(map_json_path: str, verbose: bool = True) -> np.ndarray:
    """
    Determines the best weights to optimize a graph with

    Args:
        map_json_path: the path to the json containing the unprocessed map information
        verbose (bool): whether to provide output for the chi2 calculation

    Returns:
        A list of the best weights
    """
    raise NotImplementedError(
        "This function has not been updated to work with the new way that ground truth data"
        "is being handled"
    )
    # map_dct = self._cms.map_info_from_path(map_json_path).map_dct
    # Graph.as_graph(map_dct)

    # # Use a genetic algorithm
    # model = ga(
    #     function=lambda x: 0.0,  # TODO: replace this placeholder with invocation of the ground truth metric
    #     dimension=8,
    #     variable_type="real",
    #     variable_boundaries=np.array([[-10, 10]] * 8),
    #     algorithm_parameters={
    #         "max_num_iteration": 2000,
    #         "population_size": 50,
    #         "mutation_probability": 0.1,
    #         "elit_ratio": 0.01,
    #         "crossover_probability": 0.5,
    #         "parents_portion": 0.3,
    #         "crossover_type": "uniform",
    #         "max_iteration_without_improv": None,
    #     },
    # )
    # model.run()
    # return model.report


def create_subgraphs_for_subgraph_chi2_comparison(
    graph: Dict, pso: PrescalingOptEnum
) -> Tuple[Graph, Graph]:
    """Creates then splits a graph in half, as required for weight comparison

    Specifically, this will create the graph based off the information in dct with the given prescaling option. It
    will then exactly halve this graph's vertices into two graphs. The first will allow the tag vertices to vary,
    while the second does not.

    Args:
        graph: A dictionary containing the unprocessed data that can be parsed by the `Graph.as_graph` method.
        pso: Prescaling option to pass to the `Graph.as_graph` method.

    Returns:
        A tuple of 2 graphs, an even split of graph, as described above.
    """
    graph1 = Graph.as_graph(graph, prescaling_opt=pso)
    graph2 = Graph.as_graph(
        graph, fixed_vertices={VertexType.TAG, VertexType.TAGPOINT}, prescaling_opt=pso
    )

    ordered_odom_edges = graph1.get_ordered_odometry_edges()[0]
    start_uid = graph1.edges[ordered_odom_edges[0]].startuid
    middle_uid_lower = graph1.edges[
        ordered_odom_edges[len(ordered_odom_edges) // 2]
    ].startuid
    middle_uid_upper = graph1.edges[
        ordered_odom_edges[len(ordered_odom_edges) // 2]
    ].enduid
    end_uid = graph1.edges[ordered_odom_edges[-1]].enduid

    g1sg = graph1.get_subgraph(start_odom_uid=start_uid, end_odom_uid=middle_uid_lower)
    g2sg = graph2.get_subgraph(start_odom_uid=middle_uid_upper, end_odom_uid=end_uid)

    # Delete any tag vertices from the 2nd graph that are not in the first graph
    for graph2_sg_vert in g2sg.get_tag_verts():
        if graph2_sg_vert not in g1sg.vertices:
            g2sg.delete_tag_vertex(graph2_sg_vert)
    return g1sg, g2sg


def optimize_graph(
    graph: Graph,
    oconfig: OConfig,
    visualize: bool = False,
    gt_data: Optional[GTDataSet] = None,
    anchor_tag_id: float = None,
) -> OResult:
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
        oconfig: Configures the optimization.
        gt_data: If provided, only used for the downstream optimization visualization.
        anchor_tag_id: Tag to anchor off of

    Returns:
        A tuple containing in the following order: (1) The total chi2 value of the optimized graph as returned by
         the optimize_graph method of the graph instance. (2) The dictionary returned by
         `map_processing.graph_opt_utils.optimizer_to_map_chi2` when called on the optimized graph. (3) The
         dictionary returned by `map_processing.graph_opt_utils.optimizer_to_map_chi2` when called on the
         graph before optimization.
    """
    is_sba = oconfig.is_sba
    graph.set_weights(
        weights=oconfig.weights, scale_by_edge_amount=oconfig.scale_by_edge_amount
    )
    graph.update_edge_information(compute_inf_params=oconfig.compute_inf_params)

    graph.generate_unoptimized_graph()
    before_opt_map = graph_opt_utils.optimizer_to_map_chi2(
        graph, graph.unoptimized_graph, is_sba=is_sba
    )
    fitness_metrics = graph.optimize_graph()
    if oconfig.obs_chi2_filter > 0:
        graph.filter_out_high_chi2_observation_edges(oconfig.obs_chi2_filter)
        graph.optimize_graph()

    chi2_by_cloud = graph.determine_chi2_cloud_edges()

    # Change vertex estimates based off the optimized graph
    graph.update_vertices_estimates_from_optimized_graph()
    opt_result_map = graph_opt_utils.optimizer_to_map_chi2(
        graph, graph.optimized_graph, is_sba=False
    )
    if visualize:
        graph_opt_plot_utils.plot_optimization_result(
            opt_odometry=opt_result_map.locations,
            orig_odometry=before_opt_map.locations,
            opt_tag_verts=opt_result_map.tags,
            opt_tag_corners=opt_result_map.tagpoints,
            orig_cloud_anchor=before_opt_map.cloud_anchors,
            opt_cloud_anchor=opt_result_map.cloud_anchors,
            opt_waypoint_verts=(
                opt_result_map.waypoints_metadata,
                opt_result_map.waypoints_arr,
            ),
            orig_tag_verts=before_opt_map.tags,
            ground_truth_tags=gt_data if gt_data is not None else None,
            plot_title=oconfig.graph_plot_title,
            anchor_tag_id=anchor_tag_id,
        )
        graph_opt_plot_utils.plot_adj_chi2(opt_result_map, oconfig.chi2_plot_title)
        graph_opt_plot_utils.plot_box_whisker_chis(chi2_by_cloud)

    return OResult(
        oconfig=oconfig,
        map_pre=before_opt_map,
        map_opt=opt_result_map,
        fitness_metrics=fitness_metrics,
    )


def optimize_and_get_ground_truth_error_metric(
    oconfig: OConfig,
    graph: Graph,
    ground_truth_tags: Dict[int, np.ndarray],
    visualize: bool = False,
) -> OResult:
    """Light wrapper for the optimize_graph instance method and ground_truth_metric_with_tag_id_intersection method."""
    opt_result = optimize_graph(graph=graph, oconfig=oconfig, visualize=visualize)
    (
        gt_metric,
        max_diff,
        max_diff_idx,
        _,
        _,
        gt_per_anchor,
    ) = ground_truth_metric_with_tag_id_intersection(
        optimized_tags=tag_pose_array_with_metadata_to_map(opt_result.map_opt.tags),
        ground_truth_tags=ground_truth_tags,
    )
    opt_result.gt_metric_opt = gt_metric
    opt_result.max_opt = max_diff
    opt_result.max_idx_opt = max_diff_idx
    opt_result.gt_per_anchor_tag_opt = gt_per_anchor

    return opt_result


def tag_pose_array_with_metadata_to_map(
    tag_array_with_metadata: np.ndarray,
) -> Dict[int, np.ndarray]:
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


def ground_truth_metric_with_tag_id_intersection(
    optimized_tags: Dict[int, np.ndarray], ground_truth_tags: Dict[int, np.ndarray]
) -> Tuple[float, float, int, float, int, Dict[int, float]]:
    """Use the intersection of the two provided tag dictionaries as input to the `graph_opt_utils.ground_truth_metric`
    function. Includes handling of the SBA case in which the optimized tags' estimates need to be translated and
    then inverted.

    Args:
        optimized_tags: Dictionary mapping tag IDs to their pose-containing Vertex objects.
        ground_truth_tags: Dictionary mapping tag IDs to their poses (as length-7 vectors).

    Returns:
        Value returned by the graph_opt_utils.ground_truth_metric function (see more there).
    """
    tag_id_intersection = set(optimized_tags.keys()).intersection(
        set(ground_truth_tags.keys())
    )
    optimized_tags_poses_intersection = np.zeros((len(tag_id_intersection), 7))
    gt_tags_poses_intersection = np.zeros((len(tag_id_intersection), 7))
    tag_ids = []
    for i, tag_id in enumerate(sorted(tag_id_intersection)):
        optimized_vertex_estimate = optimized_tags[tag_id]
        optimized_tags_poses_intersection[i] = optimized_vertex_estimate
        gt_tags_poses_intersection[i] = ground_truth_tags[tag_id]
        tag_ids.append(tag_id)

    (
        metric,
        max_diff,
        max_diff_idx,
        min_diff,
        min_diff_idx,
        gt_per_anchor_tag,
    ) = graph_opt_utils.ground_truth_metric(
        tag_ids,
        optimized_tag_verts=optimized_tags_poses_intersection,
        ground_truth_tags=gt_tags_poses_intersection,
    )

    return metric, max_diff, max_diff_idx, min_diff, min_diff_idx, gt_per_anchor_tag


def subgraph_pair_optimize(
    subgraphs: Union[Tuple[Graph, Graph], Dict],
    oconfig_1: OConfig,
    oconfig_2: OConfig,
    pso: PrescalingOptEnum,
) -> OSGPairResult:
    """Perform the subgraph pair optimization routine and return the difference between the first subgraph's chi2
    metric value and the second subgraph's.

    Notes:
        Tag vertex estimates are transferred between the two graphs' optimizations.

    Args:
         subgraphs: If a tuple of graphs, then the assumption is that they have been prepared using the
          create_graphs_for_chi2_comparison instance method; if a dictionary, then create_graphs_for_chi2_comparison
          is invoked with the dictionary as its argument to construct the two subgraphs. Read more of that method's
          documentation to understand this process.
         oconfig_1: Configures the optimization for the first subgraph
         oconfig_2: Configures the optimization for the second subgraph
         pso: TODO

     Returns:
         Difference of the subgraph's chi2 metrics.
    """
    if isinstance(subgraphs, Dict):
        subgraphs = create_subgraphs_for_subgraph_chi2_comparison(subgraphs, pso=pso)
    opt1_result = optimize_graph(graph=subgraphs[0], oconfig=oconfig_1)
    Graph.transfer_vertex_estimates(
        subgraphs[0], subgraphs[1], filter_by={VertexType.TAG, VertexType.TAGPOINT}
    )
    opt2_result = optimize_graph(graph=subgraphs[1], oconfig=oconfig_2)
    return OSGPairResult(sg1_oresult=opt1_result, sg2_oresult=opt2_result)
