"""
Script that makes use of the GraphManager class.

Print the usage instructions:
>> python3 graph_manager_user.py -h

Example usage that listens to the unprocessed maps' database reference:
>> python3 graph_manager_user.py -f

Example usage that optimizes and plots all graphs matching the pattern specified by the -p flag:
>> python3 graph_manager_user.py -p "unprocessed_maps/**/*Living Room*"

Notes:
- This script was adapted from the script test_firebase_sba as of commit 74891577511869f7cd3c4743c1e69fb5145f81e0
- The maps that are *processed* and cached are of a different format than the unprocessed graphs and cannot be-loaded
  for further processing.
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from typing import Tuple, Dict, List, Callable, Iterable, Any
import argparse
from firebase_admin import credentials
import map_processing
from map_processing import PrescalingOptEnum
from map_processing.graph_manager import GraphManager
from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.graph import Graph
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
import json
import multiprocessing as mp

from map_processing.data_models import OComputeInfParams, OConfig, GTDataSet, SweepResults

NOW_FORMAT = "%y-%m-%d-%H-%M-%S"

NUM_SWEEP_PROCESSES: int = 12
IS_SBA = True
ORDERED_SWEEP_CONFIG_KEYS: List[str] = [
    "odom_tag_ratio_arr",
    "lin_vel_var_arr",
    "ang_vel_var_arr",
    "grav_mag_arr",
]

# TODO: revisit the use of np.exp(.) around the lin_ and ang_vel_var arrays
SWEEP_CONFIG: Dict[str, Tuple[Callable, Iterable[Any]]] = {
    "odom_tag_ratio_arr": (np.linspace,  [0.01, 1000, 2]),
    "lin_vel_var_arr":    (np.linspace,  [0.01, 1000, 2]),
    "ang_vel_var_arr":    (np.linspace,  [0.01, 1000, 2]),
    "grav_mag_arr":       (np.linspace,  [0.01, 1000, 2]),
}


def make_parser() -> argparse.ArgumentParser:
    """Makes an argument p object for this program

    Returns:
        Argument parser
    """
    p = argparse.ArgumentParser(description="Graph optimization utility for optimizing, plotting, and database "
                                            "upload/download", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-p",
        type=str,
        help="Pattern to match to graph names; matching graph names in cache are optimized and plotted (e.g., "
             "'-g *Living_Room*' will plot any cached map with 'Living_Room' in its name); if no pattern is specified, "
             "then all cached maps are plotted and optimized (default pattern is '*'). The cache directory is searched "
             "recursively, and '**/' is automatically prepended to the pattern. If the -u flag is not given, then "
             "the root of the search is the unprocessed_maps/ sub-directory of the cache; if it is given, then the "
             "root of the search is the cache folder itself."
    )
    p.add_argument(
        "-u",
        action="store_true",
        help="Specifies the recursive search (with the pattern given by the -p argument) to be rooted in the cache "
             "folder's root (default is to be rooted in the unprocessed_maps/ sub-directory of the cache)."
    )
    p.add_argument(
        "--pso",
        type=int,
        required=False,
        help="Specifies the prescaling option used in the Graph.as_graph class method (according to the "
             "PrescalingOptEnum enum). Viable options are: "
             " 0-Sparse bundle adjustment, "
             " 1-Tag prescaling uses the full covariance matrix,"
             " 2-Tag prescaling uses only the covariance matrix diagonal,"
             " 3-Tag prescaling is a matrix of ones.",
        default=0,
        choices={0, 1, 2, 3}
    )

    weights_options = [f"{weight_option.value}-'{str(weight_option)[len(GraphManager.WeightSpecifier.__name__) + 1:]}'"
                       for weight_option in GraphManager.weights_dict.keys()]
    p.add_argument(
        "-w",
        type=int,
        required=False,
        help="Specifies which weight vector to be used (maps to a weight vector which is stored as a class attribute "
             "of the GraphManager class). Viable options are: " + ", ".join(weights_options),
        default=0,
        choices={weight_option.value for weight_option in GraphManager.weights_dict.keys()}
    )
    p.add_argument(
        "-f",
        action="store_true",
        help="Acquire maps from Firebase and overwrite existing cache."
    )
    p.add_argument(
        "-F",
        action="store_true",
        help="Upload any graphs to Firebase that are optimized while this script is running. This option is mutually "
             "exclusive with the -c option."
    )
    p.add_argument(
        "-c",
        action="store_true",
        help="Compare graph optimizations by computing two different optimizations for two sub-graphs of the "
             "specified graph: one where the tag vertices are not fixed, and one where they are. This option is "
             "mutually exclusive with the -F and -s flags."
    )
    p.add_argument(
        "-v",
        action="store_true",
        help="Visualize plots"
    )

    pso_options = [f"{pso_option.value}-'{str(pso_option)[len(PrescalingOptEnum.__name__) + 1:]}'"
                   for pso_option in PrescalingOptEnum]
    p.add_argument(
        "--fix",
        type=int,
        nargs="*",
        default=[],
        help="What vertex types to fix during optimization (note: tagpoints are always fixed). Otherwise," +
             " ,".join(pso_options),
        choices={pso_option.value for pso_option in PrescalingOptEnum}
    )
    p.add_argument(
        "--filter",
        type=float,
        required=False,
        help="Removes from the graph observation edges above this many standard deviations from the mean observation "
             "edge chi2 value in the optimized graph. The graph optimization is then re-run with the modified graph. "
             "A negative value performs no filtering.",
        default=-1.0
    )
    p.add_argument(
        "-g",
        action="store_true",
        help="Search for a matching ground truth data set and, if one is found, compute and print the ground truth "
             "metric."
    )

    p.add_argument(
        "--lvv",
        type=float,
        required=False,
        help="Linear velocity variance used for edge information matrix computation (same value is used for the x, y, "
             "and z directions)",
        default=None
    )

    p.add_argument(
        "--avv",
        type=float,
        required=False,
        help="Angular velocity variance used for edge information matrix computation.",
        default=None
    )

    p.add_argument(
        "-s",
        action="store_true",
        help="Sweep the odom-to-tag ratio, linear velocity variance, and angular velocity variance params. Mutually "
             "exclusive with the -c flag."
    )
    p.add_argument(
        "--sbea",
        action="store_true",
        help="(scale_by_edge_amount) Apply a multiplicative coefficient to the odom-to-tag ratio that is found by "
             "computing the ratio of the number of tag edges to odometry edges."
    )
    return p


def sweep_params(mi: MapInfo, ground_truth_data: dict, scale_by_edge_amount: bool) -> None:
    """TODO: Documentation and add SBA weighting to the sweeping
    """
    graph_to_opt = Graph.as_graph(mi.map_dct)
    base_oconfig = OConfig(is_sba=IS_SBA, scale_by_edge_amount=scale_by_edge_amount)

    sweep_arrs: Dict[str, np.ndarray] = {}
    for key, value in SWEEP_CONFIG.items():
        sweep_arrs[key] = value[0](*value[1])

    product_args = []
    for key in ORDERED_SWEEP_CONFIG_KEYS:
        product_args.append(sweep_arrs[key])

    # Set default value of [1, ] for any un-specified sweep parameter
    for key in set(ORDERED_SWEEP_CONFIG_KEYS).difference(sweep_arrs.keys()):
        sweep_arrs[key] = np.array([1, ])

    products = []
    oconfigs = []
    for product, oconfig in OConfig.oconfig_sweep_generator(base_oconfig=base_oconfig, product_args=product_args):
        products.append(product)
        oconfigs.append(oconfig)
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

    # Run the parameter sweep
    if NUM_SWEEP_PROCESSES == 1:  # Skip multiprocessing if only one process is specified
        results_tuples = [_sweep_target(sweep_arg) for sweep_arg in sweep_args]
    else:
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
        raise Exception("Array of results was not completely populated")

    sweep_results = SweepResults(
        gt_results_list=list(results_arr.flatten(order="C")), gt_results_arr_shape=list(results_arr.shape),
        sweep_config={item[0]: list(item[1]) for item in sweep_arrs.items()},
        sweep_config_keys_order=ORDERED_SWEEP_CONFIG_KEYS, base_oconfig=base_oconfig)
    print(f"\nMinimum ground truth value: {sweep_results.min_gt_result:.3f} with args:\n" +
          json.dumps(sweep_results.args_producing_min, indent=2))
    fig = sweep_results.visualize_results_heatmap()
    plt.show()

    results_target_folder = os.path.join(repository_root, "saved_sweeps", map_info.map_name)
    if not os.path.exists(results_target_folder):
        os.mkdir(results_target_folder)
    results_cache_file_name_no_ext = f"{datetime.datetime.now().strftime(NOW_FORMAT)}_{map_info.map_name}_sweep"
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
    print("\n")
    results = GraphManager.optimize_graph(graph=deepcopy(sweep_args_tuple[0]), visualize=False,
                                          optimization_config=sweep_args_tuple[1])
    gt_result = GraphManager.ground_truth_metric_with_tag_id_intersection(
        optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(results[1].tags),
        ground_truth_tags=sweep_args_tuple[2], verbose=False)
    # print(f"Completed sweep {sweep_args_tuple[3][0] + 1}/{sweep_args_tuple[3][1]}")
    return gt_result, sweep_args_tuple[3][0]


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.c and (args.F or args.s):
        print("Mutually exclusive flags with -c used")
        exit(-1)

    # Fetch the service account key JSON file contents
    env_variable = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if env_variable is None:
        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
    else:
        cms = CacheManagerSingleton(firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0)

    if args.f:
        cms.download_all_maps()
        exit(0)

    map_pattern = args.p if args.p else ""
    fixed_tags = set()
    for tag_type in args.fix:
        if tag_type == 0:
            fixed_tags.add(map_processing.VertexType.ODOMETRY)
        elif tag_type == 1:
            fixed_tags.add(map_processing.VertexType.TAG)
        elif tag_type == 2:
            fixed_tags.add(map_processing.VertexType.WAYPOINT)

    matching_maps = cms.find_maps(map_pattern, search_only_unprocessed=not args.u)
    if len(matching_maps) == 0:
        print(f"No matches for {map_pattern} in recursive search of {cms.cache_path}")
        exit(0)

    compute_inf_params = OComputeInfParams()
    if args.lvv is not None:
        compute_inf_params.lin_vel_var = np.ones(3) * args.lvv,
    if args.avv is not None:
        compute_inf_params.ang_vel_var = args.avv

    for map_info in matching_maps:
        if args.s:
            gt_data = cms.find_ground_truth_data_from_map_info(map_info)
            sweep_params(mi=map_info, ground_truth_data=gt_data, scale_by_edge_amount=args.sbea)
        else:
            graph_manager = GraphManager(GraphManager.WeightSpecifier(args.w), cms, pso=args.pso,
                                         scale_by_edge_amount=args.sbea)
            if args.c:
                graph_manager.compare_weights(map_info, args.v)
            else:
                gt_data = cms.find_ground_truth_data_from_map_info(map_info)
                opt_results = graph_manager.process_map(
                    map_info=map_info, visualize=args.v, upload=args.F, fixed_vertices=tuple(fixed_tags),
                    obs_chi2_filter=args.filter, compute_inf_params=compute_inf_params,
                    gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data) if gt_data is not None
                    else None)
                if not args.g:
                    continue

                if gt_data is None:
                    print(f"Could not find any ground truth for the map {map_info.map_name}")
                    continue

                ground_truth_metric_pre = graph_manager.ground_truth_metric_with_tag_id_intersection(
                    optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_results[2].tags),
                    ground_truth_tags=gt_data, verbose=False
                )
                ground_truth_metric_opt = graph_manager.ground_truth_metric_with_tag_id_intersection(
                    optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_results[1].tags),
                    ground_truth_tags=gt_data, verbose=False
                )
                print(f"Ground truth metric for {map_info.map_name}: {ground_truth_metric_opt:.3f} (delta of "
                      f"{ground_truth_metric_opt - ground_truth_metric_pre:.3f} from pre-optimization)")
