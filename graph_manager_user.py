"""
Script that makes use of the GraphManager class.

Print the usage instructions:
>> python3 graph_manager_user.py -h

Example usage that listens to the unprocessed maps database reference:
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

from firebase_admin import credentials

from GraphManager import GraphManager


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
             "recursively, and '**/' is automatically prepended to the pattern"
    )
    p.add_argument(
        "--pso",
        type=int,
        required=False,
        help="Specifies the prescaling option used in the as_graph method. Viable options are: "
             " 0-Sparse bundle adjustment, "
             " 1-Tag prescaling uses the full covariance matrix,"
             " 2-Tag prescaling uses only the covariance matrix diagonal,"
             " 3-Tag prescaling is a matrix of ones.",
        default=0,
        choices={0, 1, 2, 3}
    )
    p.add_argument(
        "-w",
        type=int,
        required=False,
        help="Specifies which weight vector to be used (maps to a weight vector which is stored as a class attribute "
             "of the GraphManager class). Viable options are: "
             " 0-'sensible_default_weights',"
             " 1-'trust_odom',"
             " 2-'trust_tags',"
             " 3-'genetic_results',"
             " 4-'comparison_baseline'.",
        default=0,
        choices={0, 1, 2, 3, 4}
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
    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.c and args.F:
        print("Mutually exclusive flags -c and -F used")
        exit(-1)

    # Fetch the service account key JSON file contents
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    graph_manager = GraphManager(args.w, cred, pso=args.pso)

    if args.f:
        graph_manager.firebase_listen()

    map_pattern = args.p if args.p else ""
    graph_manager.process_maps(map_pattern, visualize=args.v, upload=args.F, compare=args.c, new_pso=args.pso,
                               new_weights_specifier=args.w)
