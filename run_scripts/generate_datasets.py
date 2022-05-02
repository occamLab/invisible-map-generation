"""
Generate artificial datasets to be used optimization.
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import argparse
import re
from typing import Tuple, Dict, Union

import numpy as np

from map_processing import ASSUMED_TAG_SIZE, GT_TAG_DATASETS
from map_processing.graph_generator import GraphGenerator
from map_processing.data_models import UGDataSet
from run_scripts import graph_manager_user
from map_processing.cache_manager import CacheManagerSingleton


def make_parser() -> argparse.ArgumentParser:
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Acquire (from cache or Firebase) graphs, run optimization, and plot",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument(
        "-p",
        type=str,
        required=False,
        help="Specifies which path to use as the path that the simulated phone follows. Options:"
             "'e'-Ellipse, whose x-width, z-width, centerpoint, and xz-plane height are specified by the '--e_xw', "
             "'--e_zw', '--e_cp', and '--xzp' arguments, respectively; "
             "'d': Data set-based path, meaning that the path and tag positions recorded in the given data set are "
             "used to generate new observations (to specify a data set, refer to the '--d_p' argument).",
        default="e",
        choices=["e", "d"]
    )
    p.add_argument(
        "--d_p",
        type=str,
        required=False,
        help=f"If a data set-based path is specified, this defines the pattern that is used to search for the cached "
             f"data set. This argument functions the same way the '-p' argument does for the "
             f"{graph_manager_user.__name__}.py script when only unprocessed maps are searched, so refer to that help "
             f"message for more information. Note that if multiple paths are matched, then data sets are generated for "
             f"each of them."
    )
    p.add_argument(
        "--e_xw",
        type=float,
        required=False,
        help="If an elliptical path is specified, this defines the ellipse's width along the x-axis.",
        default=8.0
    )
    p.add_argument(
        "--e_zw",
        type=float,
        required=False,
        help="If an elliptical path is specified, this defines the ellipse's width along the z-axis.",
        default=4.0
    )
    p.add_argument(
        "--e_cp",
        type=str,
        required=False,
        help="If an elliptical path is specified, this defines the (x, z) centerpoint of the ellipse. The input is "
             "expected to be formatted as two integer or floating point values that are delimited by some non-digit or "
             "decimal character (e.g., '3.0,4' and '(3.0 4)' are both acceptable).",
        default="0, 0"
    )
    p.add_argument(
        "--xzp",
        type=float,
        required=False,
        help="For a parameterized path that is parameterized only in the x- and z-directions, this argument defines "
             "the y-value of the xz-coplanar plane that the path is in.",
        default=0.0
    )
    p.add_argument(
        "-t",
        type=str,
        help="In the case of a parameterized path, specifies which set of tag poses to use. Options:"
             "-'3line': 3 tags in the xz plane with (x, z) coordinates of (-3, 4), (0, 4), and (3, 4), respectively."
             "-'occam': The tags in the room adjacent to the OCCaM lab room. The origin of the coordinate system is "
             "defined at the point at the floor beneath the tag of ID 0 where the z-axis is pointing out of the wall."
             "If facing the tag, then the x-axis points to the right.",
        default="3line",
        choices=[key for key in GT_TAG_DATASETS.keys()]
    )
    p.add_argument(
        "--t_max",
        type=float,
        help="Maximum parameter value to evaluate the path at (where the starting value is 0).",
        default=6 * np.pi,
    )
    p.add_argument(
        "--np",
        type=int,
        help="Number of poses to compute along the curve (at evenly-spaced time intervals)",
        default=100,
    )
    p.add_argument(
        "--odom_noise",
        type=str,
        help="Length-4 tuple of floating point values that specifies the variance parameters (in units of 1/deltaT) "
             "for the odometry noise model's distributions. Example input: '0.01, 0.01, 0.01, 0.001'. The variance for "
             "the noise along the x, y, and z axes, and the pose rotation noise with respect to rotation around "
             "the global vertical axis, is computed between each pose computation by multiplying these parameters by "
             "the deltaT (i.e., delta of the time parameter) between the poses. The noise is sampled from a normal "
             "distribution and is used to construct a perturbing transform that is applied to each odometry pose. "
             "Therefore, the odometry path diverges from the specified path.",
        default="0, 0, 0, 0"
    )
    p.add_argument(
        "--obs_noise",
        type=float,
        help="Variance parameter for the observation model. Specifies the variance for the distribution from which "
             "pixel noise is sampled and added to the simulated tag corner pixel observations. Note that the simulated "
             "tag observation poses are re-derived from these noise pixel observations."
    )
    p.add_argument(
        "-v",
        action="store_true",
        help="Flag to visualize a plot of the graph generation."
    )
    return p


def parse_str_as_tuple(tuple_str: str, expected_length: int) -> Tuple[float, ...]:
    """Convert a string of non-decimal- or digit-delimited floating point or integer values into a tuple of floats.

    Examples:
         >>> parse_str_as_tuple("(1.2, 3)", 2)
         >>> (1.2, 3.)

         >>> parse_str_as_tuple("1 2.3", 2)
         >>> (1., 2.3)

         >>> parse_str_as_tuple("1, 2, 3", 2)  # Raises a ValueError exception

    Args:
        tuple_str: String of non-decimal- or digit-delimited floating point or integer values. Values must be
         represented in decimal form (i.e., scientific notation, hexadecimal, and other notations are not supported).
        expected_length: Expected length of the resulting tuple.

    Returns:
        Tuple of floats

    Raises:
        ValueError: if the number of extracted numerical values is not equivalent to the expected_length argument.
    """
    tuples_of_matches = re.findall(r"(\d+\.\d*)|(\d*\.\d+)|(\d+)", tuple_str)
    match_list = []
    for match_tuple in tuples_of_matches:
        for match_str in match_tuple:
            if len(match_str) != 0:
                match_list.append(match_str)
                break
    if len(match_list) != expected_length:
        raise ValueError(f"Value passed for 'e_cp' arg of {tuple_str} cannot be converted into a tuple of "
                         f"{expected_length} floating point values")
    return tuple([float(coord_str) for coord_str in match_list])


def extract_parameterized_path_args(arguments: argparse.Namespace) -> Dict[str, Union[float, Tuple[float, float]]]:
    """Construct the dictionary to be used as the path arguments for path evaluation.

    Notes:
        The value for the -p argument is used to filter the path arguments. This is done by looking for arguments whose
        names start with f"{arguments.p}_". E.g., if 'e' is the value for the -p argument (specifying an elliptical
        path), then the path arguments' names are expected to start with 'e_'.

    Args:
        arguments: Parsed command line arguments.

    Returns:
        A dictionary mapping the path arguments to their corresponding values. In the case of path arguments that are
         of a string tpe (e.g., --e_cp), the strings are parsed into their object representation.

    Raises:
        ValueError: If the values for the relevant string-type arguments cannot be parsed, then a ValueError from
         parse_str_as_tuple goes uncaught.
    """
    path_args = {arg: arguments.__getattribute__(arg) for arg in dir(args) if (
            arg.startswith(arguments.p + "_") or
            arg == "xzp"
    )}
    # Handle special cases requiring string parsing
    if "e_cp" in path_args:
        e_cp_value = path_args["e_cp"]
        path_args["e_cp"] = parse_str_as_tuple(e_cp_value, 2)
    return path_args


if __name__ == "__main__":
    parser = make_parser()
    args: argparse.Namespace = parser.parse_args()

    try:
        odom_noise_tuple = parse_str_as_tuple(args.odom_noise, 4)
        odom_noise = {noise_param_enum: odom_noise_tuple[i] for i, noise_param_enum in
                      enumerate(GraphGenerator.OdomNoiseDims.ordering())}
    except ValueError as ve:
        raise Exception(f"Could not parse the '--odom_noise' argument due to the following exception raised when "
                        f"parsing it: {ve}")

    if args.p == "e":  # e specifies an elliptical path, so acquire the arguments
        path_arguments = extract_parameterized_path_args(args)
        # Ignore unbound local variable warning for odometry_noise (it is guaranteed to be defined)
        # noinspection PyUnboundLocalVariable
        gg = GraphGenerator(path_from=GraphGenerator.PARAMETERIZED_PATH_ALIAS_TO_CALLABLE[args.p], dataset_name=args.t,
                            parameterized_path_args=path_arguments, t_max=args.t_max, n_poses=args.np,
                            tag_poses=GT_TAG_DATASETS[args.t], tag_size=ASSUMED_TAG_SIZE,
                            odometry_noise=odom_noise, obs_noise_var=args.obs_noise)
        if args.v:
            gg.visualize()

        gg.export_to_map_processing_cache()
    elif args.p == "d":  # d specifies a data set-based path, so get a CacheManagerSingleton instance ready
        # Fetch the service account key JSON file contents

        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
        matching_maps = cms.find_maps(args.d_p, search_only_unprocessed=True)
        if len(matching_maps) == 0:
            print(f"No matches for {args.d_p} in recursive search of {cms.cache_path}")
            exit(0)

        for map_info in matching_maps:
            data_set_parsed = UGDataSet(**map_info.map_dct)
            gg = GraphGenerator(path_from=data_set_parsed, dataset_name=map_info.map_name, tag_size=ASSUMED_TAG_SIZE,
                                odometry_noise=odom_noise, obs_noise_var=args.obs_noise)
            if args.v:
                gg.visualize()

            gg.export_to_map_processing_cache()
    else:
        raise Exception("Encountered unhandled value for the '-p' parameter: " + args.p)
