"""
Script that makes use of the GraphManager class.

Print the usage instructions:
>> python3 optimize_graphs_and_manage_cache.py -h

Example usage that listens to the unprocessed maps' database reference:
>> python3 optimize_graphs_and_manage_cache.py -f

Example usage that optimizes and plots all graphs matching the pattern specified by the -p flag:
>> python3 optimize_graphs_and_manage_cache.py -p "unprocessed_maps/**/*Living Room*"

Notes:
- This script was adapted from the script test_firebase_sba as of commit 74891577511869f7cd3c4743c1e69fb5145f81e0
- The maps that are *processed* and cached are of a different format than the unprocessed graphs and cannot be-loaded
  for further processing.
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import argparse
import numpy as np
from firebase_admin import credentials
from typing import Dict, Callable, Iterable, Any, Tuple

import map_processing
from map_processing import PrescalingOptEnum, VertexType
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import OComputeInfParams, GTDataSet, OConfig
from map_processing.graph_opt_hl_interface import holistic_optimize, WEIGHTS_DICT, WeightSpecifier
from map_processing.sweep import sweep_params



SWEEP_CONFIG: Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.OConfigEnum.ODOM_TAG_RATIO: (np.linspace, [1, 1, 1]),
    OConfig.OConfigEnum.LIN_VEL_VAR: (np.geomspace, [1e-6, 1e-2, 10]),
    OConfig.OConfigEnum.ANG_VEL_VAR: (np.geomspace, [1e-6, 1e-2, 10]),
    OConfig.OConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-2, 1e1, 10]),
    # OConfig.OConfigEnum.GRAV_MAG: (np.linspace, [1, 1, 1]),
}


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Graph optimization utility for optimizing, plotting, and database "
                                            "upload/download", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-p",
        type=str,
        help="Pattern to match to graph names; matching graph names in cache are optimized (e.g., '-g *Living_Room*' "
             "will plot any cached map with 'Living_Room' in its name). The cache directory is searched recursively, "
             "and '**/' is automatically prepended to the pattern."
    )
    p.add_argument(
        "-u",
        action="store_true",
        help="Specifies the recursive search (with the pattern given by the -p argument) to be rooted in the cache "
             "folder's root (default is to be rooted in the unprocessed_maps/ sub-directory of the cache).",
        default=True,
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

    weights_options = [f"{weight_option.value}-'{str(weight_option)[len(WeightSpecifier.__name__) + 1:]}'"
                       for weight_option in WEIGHTS_DICT.keys()]
    p.add_argument(
        "-w",
        type=int,
        required=False,
        help="Specifies which weight vector to be used (maps to a weight vector which is stored as a class attribute "
             "of the GraphManager class). Viable options are: " + ", ".join(weights_options),
        default=0,
        choices={weight_option.value for weight_option in WEIGHTS_DICT.keys()}
    )
    p.add_argument(
        "-f",
        action="store_true",
        help="Acquire maps from Firebase and overwrite existing cache.",
        default=False,
    )
    p.add_argument(
        "-F",
        action="store_true",
        help="Upload any graphs to Firebase that are optimized while this script is running. This option is mutually "
             "exclusive with the -c option.",
        default=False,
    )
    p.add_argument(
        "-c",
        action="store_true",
        help="Compare graph optimizations by computing two different optimizations for two sub-graphs of the "
             "specified graph: one where the tag vertices are not fixed, and one where they are. This option is "
             "mutually exclusive with the -F and -s flags.",
        default=False,
    )
    p.add_argument(
        "-v",
        action="store_true",
        help="Visualize plots",
        default=False,
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
        default=-1.0,
    )
    p.add_argument(
        "-g",
        action="store_true",
        help="Search for a matching ground truth data set and, if one is found, compute and print the ground truth "
             "metric.",
        default=False,
    )

    p.add_argument(
        "--lvv",
        type=float,
        required=False,
        help="Magnitude of the linear velocity variance vector used for edge information matrix computation",
        default=1.0,
    )

    p.add_argument(
        "--avv",
        type=float,
        required=False,
        help="Angular velocity variance used for edge information matrix computation.",
        default=1.0,
    )

    p.add_argument(
        "--tsv",
        type=float,
        required=False,
        help="Variance value used the tag (SBA) variance (i.e., noise variance in pixel space)",
        default=1.0,
    )

    p.add_argument(
        "-s",
        action="store_true",
        help=f"Sweep the parameters as specified by the SWEEP_CONFIG dictionary in {map_processing.sweep.__name__}. "
             f"Mutually exclusive with the -c flag.",
        default=False,
    )
    p.add_argument(
        "--sbea",
        action="store_true",
        help="(scale_by_edge_amount) Apply a multiplicative coefficient to the odom-to-tag ratio that is found by "
             "computing the ratio of the number of tag edges to odometry edges.",
        default=False,
    )
    p.add_argument(
        "--np",
        type=int,
        required=False,
        help="Number of processes to use when parameter sweeping.",
        default=1,
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

    # Download all maps from Firebase
    if args.f:
        cms.download_all_maps()
        exit(0)

    map_pattern = args.p if args.p else ""
    matching_maps = cms.find_maps(map_pattern, search_only_unprocessed=True)

    if len(matching_maps) == 0:
        print(f"No matches for {map_pattern} in recursive search of {CacheManagerSingleton.CACHE_PATH}")
        exit(0)

    compute_inf_params = OComputeInfParams(lin_vel_var=np.ones(3) * np.sqrt(3) * args.lvv, tag_sba_var=args.tsv,
                                           ang_vel_var=args.avv)
    for map_info in matching_maps:
        # If you want to sweep through parameters (no optimization)
        if args.s:
            gt_data = cms.find_ground_truth_data_from_map_info(map_info)
            sweep_params(mi=map_info, ground_truth_data=gt_data,
                         base_oconfig=OConfig(is_sba=args.pso == PrescalingOptEnum.USE_SBA.value,
                                              compute_inf_params=compute_inf_params),
                         sweep_config=SWEEP_CONFIG, ordered_sweep_config_keys=[key for key in SWEEP_CONFIG.keys()],
                         verbose=True, generate_plot=True, show_plot=args.v, num_processes=args.np)
        
        # If you simply want to run the optimizer 
        else:
            gt_data = cms.find_ground_truth_data_from_map_info(map_info)
            oconfig = OConfig(is_sba=args.pso == 0, weights=WEIGHTS_DICT[WeightSpecifier(args.w)],
                              scale_by_edge_amount=args.sbea, compute_inf_params=compute_inf_params)
            fixed_vertices = set()
            for tag_type in args.fix:
                fixed_vertices.add(VertexType(tag_type))
            opt_result = holistic_optimize(
                map_info=map_info, pso=PrescalingOptEnum(args.pso), oconfig=oconfig,
                fixed_vertices=fixed_vertices, verbose=True, visualize=args.v, compare=args.c, upload=args.F,
                gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data) if gt_data is not None else None)
