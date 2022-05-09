import datetime
import json
import multiprocessing as mp
import os
import sys
from copy import deepcopy
from typing import Dict, List, Tuple, Callable, Iterable, Any

import numpy as np
from matplotlib import pyplot as plt

from map_processing.cache_manager import MapInfo
from map_processing.data_models import OConfig, OSweepResults, UGDataSet
from map_processing.graph import Graph
from map_processing.graph_manager import GraphManager


REPOSITORY_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(REPOSITORY_ROOT)


NOW_FORMAT = "%y-%m-%d-%H-%M-%S"
NUM_SWEEP_PROCESSES: int = 12
IS_SBA = True

# TODO: revisit the use of np.exp(.) around the lin_ and ang_vel_var arrays
SWEEP_CONFIG: Dict[OConfig.SweepParamsEnum, Tuple[Callable, Iterable[Any]]] = {
    OConfig.SweepParamsEnum.ODOM_TAG_RATIO_ARR: (np.linspace, [0.01, 3, 5]),
    OConfig.SweepParamsEnum.LIN_VEL_VAR_ARR: (np.linspace, [0.01, 3, 20]),
    OConfig.SweepParamsEnum.ANG_VEL_VAR_ARR: (np.linspace, [0.01, 3, 20]),
    OConfig.SweepParamsEnum.GRAV_MAG_ARR: (np.linspace, [0.01, 3, 5]),
}

ORDERED_SWEEP_CONFIG_KEYS: List[OConfig.SweepParamsEnum] = [
    OConfig.SweepParamsEnum.ODOM_TAG_RATIO_ARR,
    OConfig.SweepParamsEnum.LIN_VEL_VAR_ARR,
    OConfig.SweepParamsEnum.ANG_VEL_VAR_ARR,
    OConfig.SweepParamsEnum.GRAV_MAG_ARR,
]


def sweep_params(mi: MapInfo, ground_truth_data: dict, scale_by_edge_amount: bool) -> None:
    """TODO: Documentation and add SBA weighting to the sweeping
    """
    graph_to_opt = Graph.as_graph(mi.map_dct)
    base_oconfig = OConfig(is_sba=IS_SBA, scale_by_edge_amount=scale_by_edge_amount)

    sweep_arrs: Dict[OConfig.SweepParamsEnum, np.ndarray] = {}
    for key, value in SWEEP_CONFIG.items():
        sweep_arrs[key] = value[0](*value[1])

    print("Generating list of optimization sweeping parameters...")
    products, oconfigs = OConfig.oconfig_sweep_generator(
        sweep_arrs=sweep_arrs, sweep_config_key_order=ORDERED_SWEEP_CONFIG_KEYS, base_oconfig=base_oconfig)

    if len(set([oconfig.__hash__() for oconfig in oconfigs])) != len(oconfigs):
        raise Exception("Non-unique set of optimization configurations generated")

    # Create these mappings so that the ordering of the arguments to the cartesian product in
    # `OConfig.oconfig_sweep_generator` is arbitrary with respect to the ordering of ORDERED_SWEEP_CONFIG_KEYS
    sweep_param_to_result_idx_mappings: Dict[str, Dict[float, int]] = {}
    for key in ORDERED_SWEEP_CONFIG_KEYS:
        sweep_param_to_result_idx_mappings[key] = {sweep_arg: sweep_idx for sweep_idx, sweep_arg in
                                                   enumerate(sweep_arrs[key])}
    sweep_args = []
    for i, oconfig in enumerate(oconfigs):
        sweep_args.append((graph_to_opt, oconfig, ground_truth_data, (i, len(oconfigs))))
    print(f"{len(sweep_args)} parameters generated for sweeping")

    # Run the parameter sweep
    if NUM_SWEEP_PROCESSES == 1:  # Skip multiprocessing if only one process is specified
        print("Starting optimization parameter sweep with ...")
        results_tuples = [_sweep_target(sweep_arg) for sweep_arg in sweep_args]
    else:
        print("Starting multi-processed optimization parameter sweep...")
        with mp.Pool(processes=NUM_SWEEP_PROCESSES) as pool:
            results_tuples = pool.map(_sweep_target, sweep_args)

    results: List[float] = []
    results_indices: List[int] = []
    for result_tuple in results_tuples:
        results.append(result_tuple[0])
        results_indices.append(result_tuple[1])

    results_arr_dims = [len(sweep_arrs[key]) for key in ORDERED_SWEEP_CONFIG_KEYS]
    results_arr = np.ones(results_arr_dims) * -1
    for result, result_idx in results_tuples:
        result_arr_idx = []
        for key_idx, key in enumerate(ORDERED_SWEEP_CONFIG_KEYS):
            result_arr_idx.append(sweep_param_to_result_idx_mappings[key][products[result_idx][key_idx]])
        results_arr[tuple(result_arr_idx)] = result
    if np.any(results_arr < 0):
        raise Exception("Array of sweep results was not completely populated")

    sweep_results = OSweepResults(
        gt_results_list=list(results_arr.flatten(order="C")), gt_results_arr_shape=list(results_arr.shape),
        sweep_config={item[0]: list(item[1]) for item in sweep_arrs.items()},
        sweep_config_keys_order=ORDERED_SWEEP_CONFIG_KEYS, base_oconfig=base_oconfig, map_name=mi.map_name,
        generated_params=UGDataSet.parse_obj(mi.map_dct).generated_from)
    print(f"\nMinimum ground truth value: {sweep_results.min_gt_result:.3f} with parameters:\n" +
          json.dumps(sweep_results.args_producing_min, indent=2))
    fig = sweep_results.visualize_results_heatmap()
    plt.show()

    results_target_folder = os.path.join(REPOSITORY_ROOT, "saved_sweeps", mi.map_name)
    if not os.path.exists(results_target_folder):
        os.mkdir(results_target_folder)
    results_cache_file_name_no_ext = f"{datetime.datetime.now().strftime(NOW_FORMAT)}_{mi.map_name}_sweep"
    results_cache_file_path_no_ext = os.path.join(results_target_folder, results_cache_file_name_no_ext)

    fig.savefig(results_cache_file_path_no_ext + ".png", dpi=500)
    with open(results_cache_file_path_no_ext + ".json", "w") as f:
        s = sweep_results.json(indent=2)
        f.write(s)


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
