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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import argparse
from firebase_admin import credentials
from map_processing import graph
from map_processing.graph_manager import GraphManager
from map_processing.cache_manager import CacheManagerSingleton


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
             "mutually exclusive with the -F option."
    )
    p.add_argument(
        "-v",
        action="store_true",
        help="Visualize plots"
    )
    p.add_argument(
        "--fix",
        type=int,
        nargs="*",
        default=[],
        help="What vertex types to fix during optimization. Dummy and Tagpoints are always fixed. Otherwise,"
             " 0-Odometry,"
             " 1-Tag,"
             " 2-Waypoint.",
        choices={0, 1, 2}
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
    return p


def download_maps(event):
    cms.get_map_from_unprocessed_map_event(event)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.c and args.F:
        print("Mutually exclusive flags -c and -F used")
        exit(-1)

    # Fetch the service account key JSON file contents
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    cms = CacheManagerSingleton(cred, max_listen_wait=0)
    graph_manager = GraphManager(GraphManager.WeightSpecifier(args.w), cms, pso=args.pso)

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

    for map_info in matching_maps:
        if args.c:
            graph_manager.compare_weights(map_info, args.v)
        else:
            opt_results = graph_manager.process_map(
                map_info=map_info,
                visualize=args.v,
                upload=args.F,
                fixed_vertices=tuple(fixed_tags),
                obs_chi2_filter=args.filter
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
