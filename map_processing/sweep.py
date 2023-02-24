"""Utilities for parameter sweeping
"""

import datetime
import json
import multiprocessing as mp
import pdb
import tqdm
import os
import pdb
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Iterable, Any, Union, Optional, Set

import numpy as np
from matplotlib import pyplot as plt

from map_processing import TIME_FORMAT
from map_processing import graph_opt_utils
from map_processing.cache_manager import MapInfo, CacheManagerSingleton
from map_processing.data_models import OConfig, OResult, OSweepResults, UGDataSet, GTDataSet, OG2oOptimizer
from map_processing.graph import Graph
from map_processing.graph_opt_hl_interface import optimize_graph, ground_truth_metric_with_tag_id_intersection, \
    tag_pose_array_with_metadata_to_map
from map_processing.graph_vertex_edge_classes import VertexType
from map_processing.graph_opt_utils import rotation_metric
from map_processing.validate_shift_metric import calculate_shift_metric
from . import PrescalingOptEnum, graph_opt_plot_utils


def run_param_sweep(mi: MapInfo, ground_truth_data: dict, base_oconfig: OConfig,
                 sweep_config: Union[Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]],
                                     Dict[OConfig.OConfigEnum, np.ndarray]],
                 ordered_sweep_config_keys: List[OConfig.OConfigEnum], fixed_vertices: Optional[Set[VertexType]] = None,
                 verbose: bool = False, num_processes: int = 1) -> Tuple[float, int, OResult]:
    graph_to_opt = Graph.as_graph(mi.map_dct, fixed_vertices=fixed_vertices, prescaling_opt=PrescalingOptEnum.USE_SBA\
        if base_oconfig.is_sba else PrescalingOptEnum.FULL_COV)
    sweep_arrs: Dict[OConfig.OConfigEnum, np.ndarray] = {}

    # Expand sweep_config if it contains callables and arguments to those callables
    for key, value in sweep_config.items():
        if isinstance(value, np.ndarray):
            sweep_arrs[key] = value
        elif isinstance(value, float):
            sweep_arrs[key] = np.array([value])
        else:  # Assume that value[0] is a callable and value[1] contains its arguments
            sweep_arrs[key] = value[0](*value[1])
    if verbose:
        print("Generating list of optimization sweeping parameters...")

    products, oconfigs = OConfig.oconfig_generator(
        param_multiplicands=sweep_arrs, param_order=ordered_sweep_config_keys, base_oconfig=base_oconfig)
    if len(set([oconfig.__hash__() for oconfig in oconfigs])) != len(oconfigs):
        raise Exception("Non-unique set of optimization configurations generated")

    # Create these mappings so that the ordering of the arguments to the cartesian product in
    # `OConfig.oconfig_sweep_generator` is arbitrary with respect to the ordering of ORDERED_SWEEP_CONFIG_KEYS
    sweep_param_to_result_idx_mappings: Dict[str, Dict[float, int]] = {}
    for key in ordered_sweep_config_keys:
        sweep_param_to_result_idx_mappings[key] = {sweep_arg: sweep_idx for sweep_idx, sweep_arg in
                                                   enumerate(sweep_arrs[key])}
    sweep_args = []
    for i, oconfig in (enumerate(oconfigs)):
        sweep_args.append((graph_to_opt, oconfig, ground_truth_data, (i, len(oconfigs)), verbose, mi.map_dct))
    if verbose:
        print(f"{len(sweep_args)} parameters generated for sweeping")

    # Run the parameter sweep
    num_processes = min(num_processes, len(sweep_args))
    if num_processes == 1:  # Skip multiprocessing if only one process is specified
        if verbose:
            print("Starting single-process optimization parameter sweep...")
        for i, sweep_arg in enumerate(sweep_args):
            results_tuples = [_sweep_target(sweep_arg)]
            print(i)
    else:
        if verbose:
            print(f"Starting multi-process optimization parameter sweep (with {num_processes} processes)...")
        with mp.Pool(processes=num_processes) as pool:
            total_task_num = len(sweep_args)
            results_tuples = []
            for result_tuple in tqdm.tqdm(pool.imap_unordered(_sweep_target, sweep_args), total=total_task_num):
                results_tuples.append(result_tuple)

    # OResults
    results_oresults = [result[2] for result in results_tuples]

    # Ground truth results
    results_arr_dims = [len(sweep_arrs[key]) for key in ordered_sweep_config_keys]
    results_arr = np.ones(results_arr_dims) * -1

    for result, result_idx, _ in results_tuples:
        result_arr_idx = []
        for key_idx, key in enumerate(ordered_sweep_config_keys):
            result_arr_idx.append(sweep_param_to_result_idx_mappings[key][products[result_idx][key_idx]])
        results_arr[tuple(result_arr_idx)] = result
    if np.any(results_arr < 0):
        raise Exception("Array of sweep results was not completely populated")

    return OSweepResults(
        gt_results_arr_shape=list(results_arr.shape),
        sweep_config={item[0]: list(item[1]) for item in sweep_arrs.items()},
        sweep_config_keys_order=ordered_sweep_config_keys, base_oconfig=base_oconfig, map_name=mi.map_name,
        generated_params=UGDataSet.parse_obj(mi.map_dct).generated_from, oresults_list=results_oresults,
        sweep_args=sweep_args)


def sweep_params(mi: MapInfo, ground_truth_data: dict, base_oconfig: OConfig,
                 sweep_config: Union[Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]],
                                     Dict[OConfig.OConfigEnum, np.ndarray]],
                 ordered_sweep_config_keys: List[OConfig.OConfigEnum], fixed_vertices: Optional[Set[VertexType]] = None,
                 verbose: bool = False, generate_plot: bool = False, show_plot: bool = False, num_processes: int = 1,
                 cache_results: bool = False, no_sba_baseline: bool = False, upload_best: bool = False, cms: CacheManagerSingleton = None,
                 simple_metrics = False, visualize_best_map = True) -> OSweepResults:
    """
    TODO: Documentation and add SBA weighting to the sweeping
    """
    if no_sba_baseline:
        non_sba_base_oconfig = deepcopy(base_oconfig)
        non_sba_base_oconfig.is_sba = False
        print("Running SBA Sweep")
        sba_osweep_results = run_param_sweep(mi=mi, ground_truth_data=ground_truth_data, base_oconfig=base_oconfig,
                                             sweep_config=sweep_config,
                                             ordered_sweep_config_keys=ordered_sweep_config_keys,
                                             fixed_vertices=fixed_vertices, verbose=verbose,
                                             num_processes=num_processes)
        sba_osweep_results.populate_alpha_result_list
        print("Running No SBA Sweep")
        non_sba_osweep_results = run_param_sweep(mi=mi, ground_truth_data=ground_truth_data,
                                                 base_oconfig=non_sba_base_oconfig, sweep_config=sweep_config,
                                                 ordered_sweep_config_keys=ordered_sweep_config_keys,
                                                 fixed_vertices=fixed_vertices, verbose=verbose,
                                                 num_processes=num_processes)
        non_sba_osweep_results.populate_alpha_result_list

        # Compare minimum gt metric for best parameter across sba and no sba
        min_sba_gt = sba_osweep_results.min_gt_result
        min_non_sba_gt = non_sba_osweep_results.min_gt_result

        # Compare minimum alpha metric for best parameter across sba and non sba
        min_sba_alpha = sba_osweep_results.min_alpha_result
        min_non_sba_alpha = non_sba_osweep_results.min_alpha_result

        # Currently best is based on ground truth metric
        if min_sba_gt < min_non_sba_gt:
            if verbose:
                print("SBA performed better than No SBA for ground truth")
            sweep_results = sba_osweep_results
        else:
            if verbose:
                print("No SBA performed better than SBA for ground truth")
            sweep_results = non_sba_osweep_results

        # Represent results
        print(f"Pre-Optimization GT: {sweep_results.pre_opt_gt}")
        print(f"Best SBA GT: {min_sba_gt} (delta: {min_sba_gt-sweep_results.pre_opt_gt})")
        print(f"Best No SBA GT: {min_non_sba_gt} (delta: {min_non_sba_gt-sweep_results.pre_opt_gt})")

        print(f"Best SBA Alpha: {min_sba_alpha}")
        print(f"Best No SBA Alpha: {min_non_sba_alpha}")

    else:
        sweep_results = run_param_sweep(mi=mi, ground_truth_data=ground_truth_data, base_oconfig=base_oconfig,
                                        sweep_config=sweep_config, ordered_sweep_config_keys=ordered_sweep_config_keys,
                                        fixed_vertices=fixed_vertices, verbose=verbose, num_processes=num_processes)
        sweep_results.populate_alpha_result_list

        # Find min metrics from all the parameters
        min_gt = sweep_results.min_gt_result
        min_alpha = sweep_results.min_alpha_result
        # Represent results
        print(f"Pre-Optimization GT: {sweep_results.pre_opt_gt}")
        print(f"Best GT: {min_gt} (delta: {min_gt-sweep_results.pre_opt_gt}")
        print(f"Best Alpha: {min_alpha}")

    # Get best parameter based on ground truth
    min_value_idx = sweep_results.min_gt_result_idx # Index in the list of parameters that provides the min gt_result
    min_oresult = sweep_results.min_oresult # OResult at that index
    pre_optimized_tags = min_oresult.map_pre.tags # Pre-optimized tags for the best config
    optimized_tags = min_oresult.map_opt.tags # Optimized tags for the best config

    # Get best parameter based on alpha metric
    min_value_idx_alpha = sweep_results.min_alpha_result_idx
    min_oresult_alpha = sweep_results.min_oresult_alpha
    pre_optimized_tags_alpha = min_oresult_alpha.map_pre.tags
    optimized_tags_alpha = min_oresult_alpha.map_opt.tags

    # Currently metrics are based on best gt and alpha
    rot_metric, max_rot_diff, max_rot_diff_tag_id, max_rot_diff_idx = rotation_metric(pre_optimized_tags, optimized_tags)
    rot_metric_alpha, max_rot_diff_alpha, max_rot_diff_tag_id_alpha, max_rot_diff_idx_alpha = \
        rotation_metric(pre_optimized_tags_alpha, optimized_tags_alpha)

    # Get max ground truth from above dict for the best parameter config
    max_gt = min_oresult.find_max_gt
    max_gt_tag = min_oresult.find_max_gt_tag
    max_rot_tag = optimized_tags[max_rot_diff_idx][7]

    # Print results
    if verbose:
        # Max difference is the maximum distance between a tag and when it is optimized for the best parameter
        # print(f"Maximum difference metric (pre-optimized): {min_oresult.max_pre:.3f} (tag id: {min_oresult.max_idx_pre})")
        # print(f"Maximum difference metric (optimized): {min_oresult.max_opt:.3f} (tag id: {min_oresult.max_idx_opt})")

        # print("Parameters (GT):\n" + json.dumps(sweep_results.args_producing_min, indent=2))
        # print("Parameters (Alpha):\n" + json.dumps(sweep_results.args_producing_min_alpha, indent=2))

        # Print fitness metrics
        print(f"Pre-Optimization GT: {sweep_results.pre_opt_gt}")
        print(f"For map based on min alpha, GT: {min_oresult_alpha.gt_metric_opt} "
              f"(delta = {min_oresult_alpha.gt_metric_opt - min_oresult_alpha.gt_metric_pre})")
        print(f"For map based on min gt, GT: {min_gt} "
              f"(delta = {min_gt - min_oresult.gt_metric_pre})")
        print(f"For map based on min shift, GT: {sweep_results.min_shift_gt} "
              f"(delta = {sweep_results.min_shift_gt - min_oresult.gt_metric_pre})")
        print(f"Min Shift Metric: {sweep_results.min_shift_metric}")
        print(f"\n \nFitness metrics (GT): \n"
              f"{min_oresult.fitness_metrics.repr_as_list()}")
        print(f"\nFitness metrics (Alpha): \n"
              f"{min_oresult_alpha.fitness_metrics.repr_as_list()}")
        print(f"Maximum ground truth metric: {max_gt} (tag id: {max_gt_tag})")
        print(f"Ground Truth per Tag: \n {min_oresult.gt_per_anchor_tag_opt}")

        # Print rotation metrics
        print(f"\n \nRotation metric (GT): {rot_metric}")
        print(f"Maximum rotation (GT): {max_rot_diff} (tag id: {max_rot_diff_tag_id})")
        print(f"Rotation metric (Alpha): {rot_metric_alpha}")
        print(f"Maximum rotation (Alpha): {max_rot_diff_alpha} (tag id: {max_rot_diff_tag_id_alpha})")

    if simple_metrics:
        print("Parameters (GT):\n" + json.dumps(sweep_results.args_producing_min, indent=2))
        print(f"\n \nFitness metrics (GT): \n"
              f"{min_oresult.fitness_metrics.repr_as_list()}")
    # Cache file from sweep
    results_cache_file_name_no_ext = f"{datetime.datetime.now().strftime(TIME_FORMAT)}_{mi.map_name}_sweep"
    
    processed_map_json = graph_opt_utils.make_processed_map_json(min_oresult.map_opt,
                                                                 calculate_intersections=upload_best)
    
    if visualize_best_map:
        opt_map: OG2oOptimizer = sweep_results.min_oresult.map_opt
        pre_map: OG2oOptimizer = sweep_results.min_oresult.map_pre
        graph_opt_plot_utils.plot_optimization_result(
            opt_odometry=opt_map.locations,
            orig_odometry=pre_map.locations,
            opt_tag_verts=opt_map.tags,
            opt_tag_corners=opt_map.tagpoints,
            opt_waypoint_verts=(opt_map.waypoints_metadata, opt_map.waypoints_arr),
            orig_tag_verts=pre_map.tags,
            ground_truth_tags=GTDataSet.gt_data_set_from_dict_of_arrays(ground_truth_data) if ground_truth_data else None,
            plot_title=base_oconfig.graph_plot_title,
        )
        graph_opt_plot_utils.plot_adj_chi2(opt_map, base_oconfig.chi2_plot_title)
    if cache_results:
        CacheManagerSingleton.cache_sweep_results(deepcopy(sweep_results), results_cache_file_name_no_ext)
        CacheManagerSingleton.cache_map(CacheManagerSingleton.SWEEP_PROCESSED_UPLOAD_TO, mi, processed_map_json)
    if generate_plot:
        # Visualize the worst anchor point from the best OResult (gt)
        # optimize_graph(graph=deepcopy(sweep_results.sweep_args[min_value_idx][0]), oconfig=sweep_results.weep_args[min_value_idx][1],
        #                visualize=True, gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(ground_truth_data) \
        #         if ground_truth_data is not None else None, max_gt_tag=max_gt_tag)
    
    #     # Visualize the best anchor point from the best OResult (GT)
        # optimize_graph(graph=deepcopy(sweep_results.sweep_args[min_value_idx][0]),
        #                oconfig=sweep_results.sweep_args[min_value_idx][1],
        #                visualize=True, gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(ground_truth_data) \
        #         if ground_truth_data is not None else None, max_gt_tag=max_rot_tag)
    #
    #     # Visualize the best anchor point from the best OResult (Alpha)
    #     optimize_graph(graph=deepcopy(sweep_results.sweep_args[min_value_idx_alpha][0]),
    #                    oconfig=sweep_results.sweep_args[min_value_idx_alpha][1],
    #                    visualize=True, gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(ground_truth_data) \
    #             if ground_truth_data is not None else None)
    #
        fig = sweep_results.visualize_results_heatmap()
        if show_plot:
            plt.show()
        if cache_results:
            fig.savefig(os.path.join(CacheManagerSingleton.SWEEP_RESULTS_PATH, results_cache_file_name_no_ext + ".png"),
                        dpi=500)

    if upload_best:
        cms.upload(mi, processed_map_json, verbose=verbose)

    results_dict = {
        mi.map_name: {
            "Min_GT_Param_Index": float(min_value_idx),
            "Min_Alpha_Param_Index": float(min_value_idx_alpha),
            "GT_Param": json.dumps(sweep_results.args_producing_min),
            "Alpha_Param": json.dumps(sweep_results.args_producing_min_alpha),
            "GT_GT": float(min_oresult.gt_metric_opt),
            "GT_GT_Delta": float(min_oresult.gt_metric_opt - min_oresult.gt_metric_pre),
            "Alpha_GT": float(min_oresult_alpha.gt_metric_opt),
            "Alpha_GT_Delta": float(min_oresult_alpha.gt_metric_opt - min_oresult_alpha.gt_metric_pre),
            "GT_Fitness": min_oresult.fitness_metrics.repr_as_list(),
            "Alpha_Fitness": min_oresult_alpha.fitness_metrics.repr_as_list(),
            "GT_Rotation": list(rot_metric),
            "Alpha_Rotation": list(rot_metric_alpha),
            "GT_Max_Rotation": list(max_rot_diff),
            "GT_Max_Rotation_Tag_ID": float(max_rot_diff_tag_id),
            "Alpha_Max_Rotation": list(max_rot_diff_alpha),
            "Alpha_Max_Rotation_Tag_ID": float(max_rot_diff_tag_id_alpha)
            }
        }
    # with open("results_of_sweep.json", "r+") as f:
    #     try:
    #         json_obj = json.load(f)
    #     except json.decoder.JSONDecodeError:
    #         print("HAD TO EXCEPT")
    #         pass

    # json_obj.update(results_dict)
    # with open("results_of_sweep.json", "w") as f:
    #     json.dump(json_obj, f, indent=2)

    return sweep_results


def _sweep_target(sweep_args_tuple: Tuple[Graph, OConfig, Dict[int, np.ndarray], Tuple[int, int], bool, Dict]) \
        -> Tuple[float, int]:
    """Target callable used in the sweep_params function.
    *****NOTE: This function is what individually optimizes each of the parameters provided through the sweep.

    Args:
        sweep_args_tuple: In order, contains: (1) The graph object to optimize (which is deep-copied before being passed
            sweep_args_tuple[3][0]: Int representing the index of the sweep parameter
            oresult: OResult representing the result of GraphManager.optimize_graph
    """
    # Same workflow as holistic_optimize from graph_opt_hl_interface
    oresult = optimize_graph(graph=deepcopy(sweep_args_tuple[0]), oconfig=sweep_args_tuple[1], visualize=False)
    gt_result, max_diff, max_diff_idx, min_diff, min_diff_idx, gt_per_anchor_tag = ground_truth_metric_with_tag_id_intersection(
        optimized_tags=tag_pose_array_with_metadata_to_map(oresult.map_opt.tags),
        ground_truth_tags=sweep_args_tuple[2])
    oresult.gt_per_anchor_tag_opt = gt_per_anchor_tag
    gt_result_pre, max_diff_pre, max_diff_idx_pre, min_diff_pre, min_diff_idx_pre, gt_per_anchor_tag_pre = ground_truth_metric_with_tag_id_intersection(
        optimized_tags=tag_pose_array_with_metadata_to_map(oresult.map_pre.tags),
        ground_truth_tags=sweep_args_tuple[2])

    # Add metrics to corresponding OResult
    oresult.gt_metric_pre = gt_result_pre
    oresult.gt_metric_opt = gt_result
    oresult.max_pre = max_diff_pre
    oresult.max_opt = max_diff
    oresult.min_pre = min_diff_pre
    oresult.min_opt = min_diff
    oresult.max_idx_pre = max_diff_idx_pre
    oresult.max_idx_opt = max_diff_idx
    oresult.min_idx_pre = min_diff_idx_pre
    oresult.min_idx_opt = min_diff_idx
    oresult.shift_metric = calculate_shift_metric(oresult.map_opt.tags, sweep_args_tuple[5])

    # if sweep_args_tuple[4]:
    #     print(f"Completed sweep (parameter idx={sweep_args_tuple[3][0] + 1})")

    return gt_result, sweep_args_tuple[3][0], oresult
