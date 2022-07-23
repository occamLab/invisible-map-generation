"""
Script that first finds a map without applying sba, and then applies sba on that.
"""
from copy import deepcopy
import json
import os
import pdb
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import argparse
import numpy as np
from firebase_admin import credentials

import map_processing
from map_processing import PrescalingOptEnum, VertexType
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import OComputeInfParams, GTDataSet, OConfig
from map_processing.graph_opt_hl_interface import holistic_optimize, WEIGHTS_DICT, WeightSpecifier
from map_processing.graph_opt_utils import rotation_metric
from map_processing.sweep import sweep_params
from map_processing.transform_utils import transform_vector_to_matrix
import optimize_graphs_and_manage_cache as ogmc

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
    p.add_argument(
        "-nsb",
        action="store_true",
        help="Flag to run no SBA baseline against SBA to test SBA effectiveness. Must be done with --pso 0 (SBA) and a parameter sweep.",
        default=False
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
    p.add_argument(
        "-t",
        action="store_true",
        help="Throw out data values that are too far off",
        default=False
    )
    p.add_argument(
        "-ntsba",
        action="store_true",
        help="First run with no SBA and then with SBA on those results",
        default=False
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

    if args.pso != 0 and args.nsb or args.nsb and not args.s:
        print("No SBA Baseline must be run with SBA (pso 0) and a parameter sweep.")
        exit(-1)

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
    compute_inf_params = OComputeInfParams(lin_vel_var=np.ones(3) * np.sqrt(3) * args.lvv, tag_sba_var=args.tsv,
                                           ang_vel_var=args.avv)

    if args.ntsba:
        ntsba_oresults = ogmc.find_optimal_map(cms, args.fix, compute_inf_params, weights=args.w, remove_bad_tag=args.t, sweep=args.s,
                              sba=1, visualize=False, map_pattern=map_pattern, sbea=args.sbea, compare=args.F,
                              num_processes=args.np, ntsba=False)

        # Edit json to match pre-processed estimate pattern
        for oresult in ntsba_oresults:
            semi_processed_map_dct = deepcopy(oresult[1].map_dct)
            new_tag_estimates = oresult[0].map_opt.tags
            new_odom_estimates = oresult[0].map_opt.locations

            # Edit tag estimates
            new_tag_estimates_dcts = []
            for tag_estimate in new_tag_estimates:
                temp_tag_dct = {}
                temp_tag_dct["tag_id"] = int(tag_estimate[-1])
                temp_tag_dct["tag_pose"] = list(transform_vector_to_matrix(tag_estimate[:-1]).flatten())
                new_tag_estimates_dcts.append([temp_tag_dct])

            # Edit odom estimates
            new_odom_estimates_dcts = []
            for odom_estimate in new_odom_estimates:
                temp_odom_dct = {}
                temp_odom_dct["id"] = int(odom_estimate[8])
                temp_odom_dct["pose"] = list(transform_vector_to_matrix(odom_estimate[:-2]).flatten())
                new_odom_estimates_dcts.append([temp_odom_dct])

            # Write data to json            
            semi_processed_map_dct["tag_estimates"] = sorted(new_tag_estimates_dcts, key=lambda x: x[0]["tag_id"])
            semi_processed_map_dct["odom_estimates"] = new_odom_estimates_dcts
            with open(f".cache/SemiProcessedMap/semi_processed_{oresult[1].map_name}.json", 'w') as write_file:
                json.dump(semi_processed_map_dct, write_file, indent=2, sort_keys=True)

        map_pattern = "semi_processed_" + map_pattern.split("*")[0] + "*"

        # Run on processed map
        ogmc.find_optimal_map(cms, args.fix, compute_inf_params, weights=args.w, remove_bad_tag=args.t, sweep=args.s,
                              sba=1, visualize=False, map_pattern=map_pattern, sbea=args.sbea, compare=args.F,
                              num_processes=args.np, ntsba=True)

    else:
        # Run basic version - without needing to run twice
        ogmc.find_optimal_map(cms, args.fix, compute_inf_params, weights=args.w, remove_bad_tag=args.t, sweep=args.s,
                              sba=args.sba, visualize=args.v, map_pattern=map_pattern, sbea=args.sbea, compare=args.F,
                              num_processes=args.np, ntsba=False)
