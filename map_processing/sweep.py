import datetime
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Iterable, Any, Union

import numpy as np
from matplotlib import pyplot as plt

from map_processing import TIME_FORMAT
from map_processing.cache_manager import MapInfo
from map_processing.data_models import OConfig, OSweepResults, UGDataSet
from map_processing.graph import Graph
from map_processing.graph_manager import GraphManager

REPOSITORY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(REPOSITORY_ROOT)


def sweep_params(mi: MapInfo, ground_truth_data: dict, base_oconfig: OConfig,
                 sweep_config: Union[Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]],
                                     Dict[OConfig.OConfigEnum, np.ndarray]],
                 ordered_sweep_config_keys: List[OConfig.OConfigEnum], verbose: bool = False,
                 generate_plot: bool = False, show_plot: bool = False, num_processes: int = 1) -> OSweepResults:
    """TODO: Documentation and add SBA weighting to the sweeping
    """
    graph_to_opt = Graph.as_graph(mi.map_dct)

    sweep_arrs: Dict[OConfig.OConfigEnum, np.ndarray] = {}
    # Expand sweep_config if it contains callables and arguments to those callables
    for key, value in sweep_config.items():
        if isinstance(value, np.ndarray):
            sweep_arrs[key] = value
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
    for i, oconfig in enumerate(oconfigs):
        sweep_args.append((graph_to_opt, oconfig, ground_truth_data, (i, len(oconfigs))))
    if verbose:
        print(f"{len(sweep_args)} parameters generated for sweeping")

    # Run the parameter sweep
    num_processes = min(num_processes, len(sweep_args))
    if num_processes == 1:  # Skip multiprocessing if only one process is specified
        if verbose:
            print("Starting single-process optimization parameter sweep...")
        results_tuples = [_sweep_target(sweep_arg) for sweep_arg in sweep_args]
    else:
        if verbose:
            print(f"Starting multi-process optimization parameter sweep (with {num_processes} processes)...")
        with mp.Pool(processes=num_processes) as pool:
            results_tuples = pool.map(_sweep_target, sweep_args)

    results: List[float] = []
    results_indices: List[int] = []
    for result_tuple in results_tuples:
        results.append(result_tuple[0])
        results_indices.append(result_tuple[1])

    results_arr_dims = [len(sweep_arrs[key]) for key in ordered_sweep_config_keys]
    results_arr = np.ones(results_arr_dims) * -1
    for result, result_idx in results_tuples:
        result_arr_idx = []
        for key_idx, key in enumerate(ordered_sweep_config_keys):
            result_arr_idx.append(sweep_param_to_result_idx_mappings[key][products[result_idx][key_idx]])
        results_arr[tuple(result_arr_idx)] = result
    if np.any(results_arr < 0):
        raise Exception("Array of sweep results was not completely populated")

    sweep_results = OSweepResults(
        gt_results_list=list(results_arr.flatten(order="C")), gt_results_arr_shape=list(results_arr.shape),
        sweep_config={item[0]: list(item[1]) for item in sweep_arrs.items()},
        sweep_config_keys_order=ordered_sweep_config_keys, base_oconfig=base_oconfig, map_name=mi.map_name,
        generated_params=UGDataSet.parse_obj(mi.map_dct).generated_from)
    if verbose:
        print(f"\nMinimum ground truth value: {sweep_results.min_gt_result:.3f} with parameters:\n" +
              json.dumps(sweep_results.args_producing_min, indent=2))

    results_target_folder = os.path.join(REPOSITORY_ROOT, "saved_sweeps", mi.map_name)
    if not os.path.exists(results_target_folder):
        os.mkdir(results_target_folder)
    results_cache_file_name_no_ext = f"{datetime.datetime.now().strftime(TIME_FORMAT)}_{mi.map_name}_sweep"
    results_cache_file_path_no_ext = os.path.join(results_target_folder, results_cache_file_name_no_ext)

    if generate_plot:
        fig = sweep_results.visualize_results_heatmap()
        if show_plot:
            plt.show()
        fig.savefig(results_cache_file_path_no_ext + ".png", dpi=500)

    with open(results_cache_file_path_no_ext + ".json", "w") as f:
        s = sweep_results.json(indent=2)
        f.write(s)
    return sweep_results


def _sweep_target(sweep_args_tuple: Tuple[Graph, OConfig, Dict[int, np.ndarray], Tuple[int, int]]) -> Tuple[float, int]:
    """
    Args:
        sweep_args_tuple: In order, contains: (1) The graph object to optimize (which is deep-copied before being passed
         as the argument), (2) the optimization configuration, and (3) the ground truth tags dictionary.

    Returns:
        Return value from GraphManager.optimize_graph
    """
    oresult = GraphManager.optimize_graph(graph=deepcopy(sweep_args_tuple[0]), optimization_config=sweep_args_tuple[1],
                                          visualize=False)
    gt_result = GraphManager.ground_truth_metric_with_tag_id_intersection(
        optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(oresult.map_opt.tags),
        ground_truth_tags=sweep_args_tuple[2])
    print(f"Completed sweep (parameter idx={sweep_args_tuple[3][0] + 1})")
    return gt_result, sweep_args_tuple[3][0]
