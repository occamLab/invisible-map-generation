"""
Utilities for parameter sweeping
"""

import datetime
import json
import multiprocessing as mp
import tqdm
import os
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Iterable, Any, Union, Optional, Set

import numpy as np
from matplotlib import pyplot as plt

from map_processing import TIME_FORMAT
from map_processing import graph_opt_utils
from map_processing.cache_manager import MapInfo, CacheManagerSingleton
from map_processing.data_models import (
    OConfig,
    OResult,
    OSweepResults,
    UGDataSet,
    GTDataSet,
    OG2oOptimizer,
)
from map_processing.graph import Graph
from map_processing.graph_opt_hl_interface import (
    optimize_graph,
    ground_truth_metric_with_tag_id_intersection,
    tag_pose_array_with_metadata_to_map,
)
from map_processing.graph_vertex_edge_classes import VertexType
from map_processing.validate_shift_metric import calculate_shift_metric
from . import PrescalingOptEnum, graph_opt_plot_utils


def run_param_sweep(
    mi: MapInfo,
    ground_truth_data: dict,
    base_oconfig: OConfig,
    sweep_config: Union[
        Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]],
        Dict[OConfig.OConfigEnum, np.ndarray],
    ],
    ordered_sweep_config_keys: List[OConfig.OConfigEnum],
    fixed_vertices: Optional[Set[VertexType]] = None,
    verbose: bool = False,
    num_processes: int = 1,
) -> Tuple[float, int, OResult]:
    graph_to_opt = Graph.as_graph(
        mi.map_dct,
        fixed_vertices=fixed_vertices,
        prescaling_opt=PrescalingOptEnum.USE_SBA
        if base_oconfig.is_sba
        else PrescalingOptEnum.FULL_COV,
    )
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
        param_multiplicands=sweep_arrs,
        param_order=ordered_sweep_config_keys,
        base_oconfig=base_oconfig,
    )
    if len(set([oconfig.__hash__() for oconfig in oconfigs])) != len(oconfigs):
        raise Exception("Non-unique set of optimization configurations generated")

    # Create these mappings so that the ordering of the arguments to the cartesian product in
    # `OConfig.oconfig_sweep_generator` is arbitrary with respect to the ordering of ORDERED_SWEEP_CONFIG_KEYS
    sweep_param_to_result_idx_mappings: Dict[str, Dict[float, int]] = {}
    for key in ordered_sweep_config_keys:
        sweep_param_to_result_idx_mappings[key] = {
            sweep_arg: sweep_idx for sweep_idx, sweep_arg in enumerate(sweep_arrs[key])
        }
    sweep_args = []
    for i, oconfig in enumerate(oconfigs):
        sweep_args.append(
            (
                graph_to_opt,
                oconfig,
                ground_truth_data,
                (i, len(oconfigs)),
                verbose,
                mi.map_dct,
            )
        )
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
            print(
                f"Starting multi-process optimization parameter sweep (with {num_processes} processes)..."
            )
        with mp.Pool(processes=num_processes) as pool:
            total_task_num = len(sweep_args)
            results_tuples = []
            for result_tuple in tqdm.tqdm(
                pool.imap_unordered(_sweep_target, sweep_args), total=total_task_num
            ):
                results_tuples.append(result_tuple)

    # OResults
    results_oresults = [result[2] for result in results_tuples]

    # Ground truth results
    results_arr_dims = [len(sweep_arrs[key]) for key in ordered_sweep_config_keys]
    results_arr = np.ones(results_arr_dims) * -1

    for result, result_idx, _ in results_tuples:
        result_arr_idx = []
        for key_idx, key in enumerate(ordered_sweep_config_keys):
            result_arr_idx.append(
                sweep_param_to_result_idx_mappings[key][products[result_idx][key_idx]]
            )
        results_arr[tuple(result_arr_idx)] = result
    if np.any(results_arr < 0):
        raise Exception("Array of sweep results was not completely populated")

    return OSweepResults(
        gt_results_arr_shape=list(results_arr.shape),
        sweep_config={item[0]: list(item[1]) for item in sweep_arrs.items()},
        sweep_config_keys_order=ordered_sweep_config_keys,
        base_oconfig=base_oconfig,
        map_name=mi.map_name,
        generated_params=UGDataSet.parse_obj(mi.map_dct).generated_from,
        oresults_list=results_oresults,
        sweep_args=sweep_args,
    )


def sweep_params(
    mi: MapInfo,
    ground_truth_data: dict,
    base_oconfig: OConfig,
    sweep_config: Union[
        Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]],
        Dict[OConfig.OConfigEnum, np.ndarray],
    ],
    ordered_sweep_config_keys: List[OConfig.OConfigEnum],
    fixed_vertices: Optional[Set[VertexType]] = None,
    verbose: bool = False,
    generate_plot: bool = False,
    show_plots: bool = False,
    num_processes: int = 1,
    cache_results: bool = False,
    upload_best: bool = False,
    cms: CacheManagerSingleton = None,
) -> OSweepResults:
    """
    TODO: Documentation and add SBA weighting to the sweeping
    """
    sweep_results: OSweepResults = run_param_sweep(
        mi=mi,
        ground_truth_data=ground_truth_data,
        base_oconfig=base_oconfig,
        sweep_config=sweep_config,
        ordered_sweep_config_keys=ordered_sweep_config_keys,
        fixed_vertices=fixed_vertices,
        verbose=verbose,
        num_processes=num_processes,
    )

    # Find min metrics from all the parameters
    min_gt = sweep_results.min_gt_result
    min_alpha = sweep_results.min_alpha_result
    min_shift = sweep_results.min_shift_result

    # Get best parameter based on ground truth
    min_oresult = sweep_results.min_oresult  # OResult at that index

    # Get best parameter based on alpha metric
    min_oresult_alpha = sweep_results.min_oresult_alpha

    # Get max ground truth from above dict for the best parameter config
    max_gt = min_oresult.find_max_gt
    max_gt_tag = min_oresult.find_max_gt_tag

    # Print results
    if verbose:
        # Print fitness metrics
        print(f"Pre-Optimization GT: {sweep_results.pre_opt_gt}")
        print(
            f"For map based on min alpha, GT: {min_oresult_alpha.gt_metric_opt} "
            f"(delta = {min_oresult_alpha.gt_metric_opt - min_oresult_alpha.gt_metric_pre})"
        )
        print(
            f"For map based on min gt, GT: {min_gt} "
            f"(delta = {min_gt - min_oresult.gt_metric_pre})"
        )
        print(
            f"For map based on min shift, GT: {sweep_results.min_shift_gt} "
            f"(delta = {sweep_results.min_shift_gt - min_oresult.gt_metric_pre})"
        )
        print(f"Min Shift Metric: {sweep_results.min_shift_metric}")
        print(
            f"\n \nFitness metrics (GT): \n"
            f"{min_oresult.fitness_metrics.repr_as_list()}"
        )
        print(
            f"\nFitness metrics (Alpha): \n"
            f"{min_oresult_alpha.fitness_metrics.repr_as_list()}"
        )
        print(f"Maximum ground truth metric: {max_gt} (tag id: {max_gt_tag})")
        print(f"Ground Truth per Tag: \n {min_oresult.gt_per_anchor_tag_opt}")

        print("Optimal Hyperparameters:")
        print(
            "Parameters (GT):\n"
            + json.dumps(sweep_results.args_producing_min, indent=2)
        )
        print(
            "Parameters (Alpha):\n"
            + json.dumps(sweep_results.args_producing_min_alpha, indent=2)
        )
        print(
            "Parameters (Shift):\n"
            + json.dumps(sweep_results.args_producing_min_shift, indent=2)
        )
    else:
        # Display minimal results
        print(f"Pre-Optimization GT: {sweep_results.pre_opt_gt}")
        print(f"Best GT: {min_gt} (delta: {min_gt-sweep_results.pre_opt_gt}")
        print(f"Best Alpha: {min_alpha} (delta: {min_alpha-sweep_results.pre_opt_gt})")
        print(f"Best Shift: {min_shift} (delta: {min_shift-sweep_results.pre_opt_gt})")

    # Cache file from sweep
    results_cache_file_name_no_ext = (
        f"{datetime.datetime.now().strftime(TIME_FORMAT)}_{mi.map_name}_sweep"
    )

    processed_map_json = graph_opt_utils.make_processed_map_json(
        min_oresult.map_opt, calculate_intersections=upload_best
    )

    if cache_results:
        CacheManagerSingleton.cache_sweep_results(
            deepcopy(sweep_results), results_cache_file_name_no_ext
        )
        CacheManagerSingleton.cache_map(
            CacheManagerSingleton.SWEEP_PROCESSED_UPLOAD_TO, mi, processed_map_json
        )
    if generate_plot:
        fig = sweep_results.visualize_results_heatmap()
        if show_plots:
            plt.show()

            # Visualize best sweep result map
            opt_map: OG2oOptimizer = sweep_results.min_oresult.map_opt
            pre_map: OG2oOptimizer = sweep_results.min_oresult.map_pre
            graph_opt_plot_utils.plot_optimization_result(
                opt_odometry=opt_map.locations,
                orig_odometry=pre_map.locations,
                opt_tag_verts=opt_map.tags,
                opt_tag_corners=opt_map.tagpoints,
                opt_waypoint_verts=(opt_map.waypoints_metadata, opt_map.waypoints_arr),
                orig_tag_verts=pre_map.tags,
                ground_truth_tags=GTDataSet.gt_data_set_from_dict_of_arrays(
                    ground_truth_data
                )
                if ground_truth_data
                else None,
                plot_title=base_oconfig.graph_plot_title,
            )
            graph_opt_plot_utils.plot_adj_chi2(opt_map, base_oconfig.chi2_plot_title)

        if cache_results:
            fig.savefig(
                os.path.join(
                    CacheManagerSingleton.SWEEP_RESULTS_PATH,
                    results_cache_file_name_no_ext + ".png",
                ),
                dpi=500,
            )

    if upload_best:
        cms.upload(mi, processed_map_json, verbose=verbose)

    return sweep_results


def _sweep_target(
    sweep_args_tuple: Tuple[
        Graph, OConfig, Dict[int, np.ndarray], Tuple[int, int], bool, Dict
    ]
) -> Tuple[float, int]:
    """Target callable used in the sweep_params function.
    *****NOTE: This function is what individually optimizes each of the parameters provided through the sweep.

    Args:
        sweep_args_tuple: In order, contains: (1) The graph object to optimize (which is deep-copied before being passed
            sweep_args_tuple[3][0]: Int representing the index of the sweep parameter
            oresult: OResult representing the result of GraphManager.optimize_graph
    """
    # Same workflow as holistic_optimize from graph_opt_hl_interface
    oresult = optimize_graph(
        graph=deepcopy(sweep_args_tuple[0]),
        oconfig=sweep_args_tuple[1],
        visualize=False,
    )
    (
        gt_result,
        max_diff,
        max_diff_idx,
        min_diff,
        min_diff_idx,
        gt_per_anchor_tag,
    ) = ground_truth_metric_with_tag_id_intersection(
        optimized_tags=tag_pose_array_with_metadata_to_map(oresult.map_opt.tags),
        ground_truth_tags=sweep_args_tuple[2],
    )
    oresult.gt_per_anchor_tag_opt = gt_per_anchor_tag
    (
        gt_result_pre,
        max_diff_pre,
        max_diff_idx_pre,
        min_diff_pre,
        min_diff_idx_pre,
        _,
    ) = ground_truth_metric_with_tag_id_intersection(
        optimized_tags=tag_pose_array_with_metadata_to_map(oresult.map_pre.tags),
        ground_truth_tags=sweep_args_tuple[2],
    )

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
    oresult.shift_metric = calculate_shift_metric(
        oresult.map_opt.tags, sweep_args_tuple[5]
    )

    # if sweep_args_tuple[4]:
    #     print(f"Completed sweep (parameter idx={sweep_args_tuple[3][0] + 1})")

    return gt_result, sweep_args_tuple[3][0], oresult
