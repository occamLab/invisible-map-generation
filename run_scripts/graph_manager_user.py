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

import argparse
import os

import numpy as np
from firebase_admin import credentials

from map_processing import PrescalingOptEnum, VertexType
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import OComputeInfParams, GTDataSet
from map_processing.graph_manager import GraphManager
from map_processing.sweep import sweep_params
import map_processing


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
        help=f"Sweep the parameters as specified by the SWEEP_CONFIG dictionary in {map_processing.sweep.__name__}. "
             f"Mutually exclusive with the -c flag."
    )
    p.add_argument(
        "--sbea",
        action="store_true",
        help="(scale_by_edge_amount) Apply a multiplicative coefficient to the odom-to-tag ratio that is found by "
             "computing the ratio of the number of tag edges to odometry edges."
    )
    return p


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
            fixed_tags.add(VertexType.ODOMETRY)
        elif tag_type == 1:
            fixed_tags.add(VertexType.TAG)
        elif tag_type == 2:
            fixed_tags.add(VertexType.WAYPOINT)

    matching_maps = cms.find_maps(map_pattern, search_only_unprocessed=not args.u)
    if len(matching_maps) == 0:
        print(f"No matches for {map_pattern} in recursive search of {CacheManagerSingleton.CACHE_PATH}")
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
            compare: bool = False
            if args.c:
                compare = True
            gt_data = cms.find_ground_truth_data_from_map_info(map_info)
            opt_result = graph_manager.holistic_optimize(
                map_info=map_info, visualize=args.v, upload=args.F, compare=compare, fixed_vertices=tuple(fixed_tags),
                obs_chi2_filter=args.filter, compute_inf_params=compute_inf_params, verbose=True,
                gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data) if gt_data is not None else None)
