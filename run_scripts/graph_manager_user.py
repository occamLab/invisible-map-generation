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

from typing import Tuple, List, Dict
import argparse
from firebase_admin import credentials
from map_processing import graph, PrescalingOptEnum
from map_processing.graph_manager import GraphManager
from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.graph import Graph
from map_processing.weights import Weights
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import datetime
import json
import pickle

import concurrent.futures

NOW_FORMAT = "%y-%m-%d-%H-%M-%S"

NUM_SWEEP_THREADS = 12
ODOM_TAG_RATIO_GEOMSPACE_ARGS = [10, 100, 4]
ANG_VEL_VAR_LINSPACE_ARGS = [3, 10, 4]
LIN_VEL_VAR_LINSPACE_ARGS = [-5, 5, 4]


def make_parser():
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Acquire (from cache or Firebase) graphs, run optimization, and plot")
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
                       for weight_option in GraphManager.WeightSpecifier]
    p.add_argument(
        "-w",
        type=int,
        required=False,
        help="Specifies which weight vector to be used (maps to a weight vector which is stored as a class attribute "
             "of the GraphManager class). Viable options are: " + ", ".join(weights_options),
        default=0,
        choices={weight_option.value for weight_option in GraphManager.WeightSpecifier}
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
        help="What vertex types to fix during optimization. Dummy and Tagpoints are always fixed. Otherwise," +
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
    return p


def download_maps(event):
    cms.get_map_from_unprocessed_map_event(event)


def sweep_params(mi: MapInfo, ground_truth_data: dict):
    """TODO: Documentation and add SBA weighting to the sweeping
    """
    graph_to_opt = Graph.as_graph(mi.map_dct)

    odom_tag_ratio_arr = np.geomspace(*ODOM_TAG_RATIO_GEOMSPACE_ARGS)
    odom_tag_ratio_arr_idx_map: Dict[float, int] = {}
    ang_vel_arr = np.exp(np.linspace(*ANG_VEL_VAR_LINSPACE_ARGS))
    ang_vel_arr_idx_map: Dict[float, int] = {}
    lin_vel_arr = np.exp(np.linspace(*LIN_VEL_VAR_LINSPACE_ARGS))
    lin_vel_arr_idx_map: Dict[float, int] = {}

    sweep_args_list: List[Tuple[float, float, float, Graph, dict, List[Tuple[float, Tuple[float, float, float]]]]] = []
    results_list: List[Tuple[float, Tuple[float, float, float]]] = []
    for i_idx, i in enumerate(odom_tag_ratio_arr):
        odom_tag_ratio_arr_idx_map[i] = i_idx
        for j_idx, j in enumerate(ang_vel_arr):
            if j not in ang_vel_arr_idx_map:
                ang_vel_arr_idx_map[j] = j_idx
            for k_idx, k in enumerate(lin_vel_arr):
                if k not in lin_vel_arr_idx_map:
                    lin_vel_arr_idx_map[k] = k_idx
                sweep_args_list.append((i, j, k, graph_to_opt, ground_truth_data, results_list))

    with concurrent.futures.ThreadPoolExecutor(NUM_SWEEP_THREADS) as executor:
        executor.map(sweep_target, sweep_args_list)

    # Put results in a numpy array
    results_arr = np.zeros([ODOM_TAG_RATIO_GEOMSPACE_ARGS[2], ANG_VEL_VAR_LINSPACE_ARGS[2],
                            LIN_VEL_VAR_LINSPACE_ARGS[2]])
    for result in results_list:
        result_params = result[1]
        results_arr[odom_tag_ratio_arr_idx_map[result_params[0]], ang_vel_arr_idx_map[result_params[1]],
                    lin_vel_arr_idx_map[result_params[2]]] = result[0]

    results_target_folder = os.path.join(repository_root, "saved_sweeps", map_info.map_name)
    if not os.path.exists(results_target_folder):
        os.mkdir(results_target_folder)

    results_cache_file_name_no_ext = f"{datetime.datetime.now().strftime(NOW_FORMAT)}_{map_info.map_name}_sweep"
    results_args_dict = {
        "ODOM_TAG_RATIO_GEOMSPACE_ARGS": ODOM_TAG_RATIO_GEOMSPACE_ARGS,
        "ANG_VEL_VAR_LINSPACE_ARGS": ANG_VEL_VAR_LINSPACE_ARGS,
        "LIN_VEL_VAR_LINSPACE_ARGS": LIN_VEL_VAR_LINSPACE_ARGS
    }
    with open(os.path.join(results_target_folder, results_cache_file_name_no_ext + ".json"), "w") as f:
        json.dump(obj=results_args_dict, fp=f, indent=2)

    with open(os.path.join(results_target_folder, results_cache_file_name_no_ext + ".pickle"), "wb") as f:
        pickle.dump(obj=results_arr, file=f, protocol=pickle.HIGHEST_PROTOCOL)

    # noinspection PyArgumentList
    min_ground_truth = results_arr.min()

    # noinspection PyTypeChecker
    where_min_pre: Tuple[np.ndarray, np.ndarray, np.ndarray] = np.where(results_arr == min_ground_truth)
    where_min = tuple([arr[0] for arr in where_min_pre])  # Select first result if there are multiple
    print(f"Minimum ground truth value: {min_ground_truth} (with odom-tag ratio of {odom_tag_ratio_arr[where_min[0]]}, "
          f"lin-vel-var of {lin_vel_arr[where_min[1]]}, and ang-vel-var of {ang_vel_arr[where_min[2]]})")

    xx, yy = np.meshgrid(ang_vel_arr, lin_vel_arr)
    # Get rid of the first dimension by indexing into it where the minimum occurs
    zz = results_arr[where_min[0], :, :]  # Ground truth metric
    ax: plt.Axes
    fig: plt.Figure
    fig, ax = plt.subplots()
    ax.set_title(f"Ground truth metric vs. ang. and\nlin. vel. variance (odom-tag ratio="
                 f"{odom_tag_ratio_arr[where_min[0]]})")
    ax.set_xlabel("Angular velocity variance")
    ax.set_ylabel("Linear velocity variance")
    c = ax.pcolor(xx, yy, zz, shading="auto")
    fig.colorbar(c, ax=ax)
    plt.savefig(os.path.join(results_target_folder, results_cache_file_name_no_ext + ".png"), dpi=300)
    plt.show()


def sweep_target(sweep_args_tuple: Tuple[float, float, float, Graph, dict,
                                         List[Tuple[float, Tuple[float, float, float]]]]) -> None:
    """
    Args:
        sweep_args_tuple: Odom-tag ratio, angular velocity variance, linear velocity variance, and the graph object to
         optimize. Note: the graph object is deep-copied before being passed as the argument.

    Returns:
        Return value from GraphManager.optimize_graph
    """
    results = GraphManager.optimize_graph(
        is_sba=True,
        graph=deepcopy(sweep_args_tuple[3]),
        tune_weights=False,
        visualize=False,
        weights=Weights(odom_tag_ratio=sweep_args_tuple[0], dummy=np.array([0.1, 4, 0.1])),
        obs_chi2_filter=-1,
        compute_inf_params={
            "ang_vel_var": sweep_args_tuple[1],
            "lin_vel_var": sweep_args_tuple[2] * np.ones(3)
        }
    )
    gt_result = GraphManager.ground_truth_metric_with_tag_id_intersection(
        optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(results[1]["tags"]),
        ground_truth_tags=sweep_args_tuple[4], verbose=False
    )
    sweep_args_tuple[5].append((gt_result, (sweep_args_tuple[0], sweep_args_tuple[1], sweep_args_tuple[2])))


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.c and (args.F or args.s):
        print("Mutually exclusive flags with -c used")
        exit(-1)

    # Fetch the service account key JSON file contents
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    cms = CacheManagerSingleton(cred, max_listen_wait=0)

    if args.f:
        cms.download_all_maps()

    map_pattern = args.p if args.p else ""
    fixed_tags = set()
    for tag_type in args.fix:
        if tag_type == 0:
            fixed_tags.add(graph.VertexType.ODOMETRY)
        elif tag_type == 1:
            fixed_tags.add(graph.VertexType.TAG)
        elif tag_type == 2:
            fixed_tags.add(graph.VertexType.WAYPOINT)

    matching_maps = cms.find_maps(map_pattern, search_only_unprocessed=not args.u)
    if len(matching_maps) == 0:
        print("No matches for {} in recursive search of {}".format(map_pattern, cms.cache_path))
        exit(-1)

    compute_inf_params = {}
    if args.lvv is not None:
        compute_inf_params["lin_vel_var"] = np.ones(3) * args.lvv,
    if args.avv is not None:
        compute_inf_params["ang_vel_var"] = args.avv

    for map_info in matching_maps:
        if args.s:
            gt_data = cms.find_ground_truth_data_from_map_info(map_info)
            sweep_params(mi=map_info, ground_truth_data=gt_data)
        else:
            graph_manager = GraphManager(GraphManager.WeightSpecifier(args.w), cms, pso=args.pso)
            if args.c:
                graph_manager.compare_weights(map_info, args.v)
            else:
                opt_results = graph_manager.process_map(
                    map_info=map_info,
                    visualize=args.v,
                    upload=args.F,
                    fixed_vertices=tuple(fixed_tags),
                    obs_chi2_filter=args.filter,
                    compute_inf_params=compute_inf_params
                )
                if not args.g:
                    continue

                gt_data = cms.find_ground_truth_data_from_map_info(map_info)
                if gt_data is None:
                    print(f"Could not find any ground truth for the map {map_info.map_name}")
                    continue

                ground_truth_metric_pre = graph_manager.ground_truth_metric_with_tag_id_intersection(
                    optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_results[2]["tags"]),
                    ground_truth_tags=gt_data, verbose=False
                )
                ground_truth_metric_opt = graph_manager.ground_truth_metric_with_tag_id_intersection(
                    optimized_tags=GraphManager.tag_pose_array_with_metadata_to_map(opt_results[1]["tags"]),
                    ground_truth_tags=gt_data, verbose=False
                )
                print(f"Ground truth metric for {map_info.map_name}: {ground_truth_metric_opt} (delta of "
                      f"{ground_truth_metric_opt - ground_truth_metric_pre} from pre-optimization)")
