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

from map_processing import ASSUMED_TAG_SIZE
from map_processing.dataset_generation.graph_generator import GraphGenerator


def make_parser() -> argparse.ArgumentParser:
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Acquire (from cache or Firebase) graphs, run optimization, and plot")
    p.add_argument(
        "-p",
        type=str,
        required=False,
        help="Specifies which path to use as the path that the simulated phone follows. (TODO: add more options besides"
             "the ellipse.) Options:"
             "-'e': Ellipse, whose x-width, z-width, centerpoint, and xz-plane height are specified by the '--e_xw', "
             "'--e_zw', '--e_cp', and '--xzp' arguments, respectively",
        default="e"
    )
    p.add_argument(
        "--e_xw",
        type=float,
        required=False,
        help="If an ellipse-shaped path is specified, this defines the ellipse's width along the x-axis.",
        default=8.0
    )
    p.add_argument(
        "--e_zw",
        type=float,
        required=False,
        help="If an ellipse-shaped path is specified, this defines the ellipse's width along the z-axis.",
        default=4.0
    )
    p.add_argument(
        "--e_cp",
        type=str,
        required=False,
        help="If an ellipse-shaped path is specified, this defines the (x, z) centerpoint of the ellipse. The input is "
             "expected to be formatted as two integer or floating point values that are delimited by some non-digit or "
             "decimal character (e.g., '3.0,4' and '(3.0 4)' are both acceptable).",
        default="0, 0"
    )
    p.add_argument(
        "--xzp",
        type=float,
        required=False,
        help="For a path that is parameterized only in the x- and z-directions, this argument defines the y-value of "
             "the xz-coplanar plane that the path is in.",
        default=0.0
    )
    p.add_argument(
        "-t",
        type=str,
        help="Specifies which set of tag poses to use. Options:"
             "-'3line': 3 tags in the xz plane with (x, z) coordinates of (-3, 4), (0, 4), and (3, 4), respectively."
             "-'occam': The tags in the room adjacent to the OCCaM lab room. The origin of the coordinate system is "
             "defined at the point at the floor beneath the tag of ID 0 where the z-axis is pointing out of the wall."
             "If facing the tag, then the x-axis points to the right.",
        default="3line",
        choices=["3line", "occam"]
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
        "--noise",
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


def extract_path_args(arguments: argparse.Namespace) -> Dict[str, Union[float, Tuple[float, float]]]:
    """Construct the dictionary to be used as the path arguments for path evaluation.

    Notes:
        The value for the -p argument is used to filter the path arguments. This is done by looking for arguments whose
        names start with f"{arguments.p}_". E.g., if 'e' is the value for the -p argument (specifying an elliptical
        path, then the path arguments' names are expected to start with 'e_'.

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
    path_arguments = extract_path_args(args)

    if args.p not in GraphGenerator.PATH_ALIAS_TO_CALLABLE:
        print(f"Accepted argument to flag '-p' of {args.p} does not have a matching key in the path alias to "
              "callable map of the GraphGenerator.")
        exit(-1)

    if args.t not in GraphGenerator.TAG_DATASETS:
        print(f"Accepted argument to flag '-t' of {args.t} does not have a matching key in the dictionary of tag "
              f"datasets.")
        exit(-1)

    try:
        odom_noise_tuple = parse_str_as_tuple(args.noise, 4)
        odom_noise = {noise_param_enum: odom_noise_tuple[i] for i, noise_param_enum in
                      enumerate(GraphGenerator.OdomNoiseDims.ordering())}
    except ValueError as ve:
        print(f"Could not parse the --noise argument due to the following exception raised when parsing it: {ve}")
        exit(-1)

    # Ignore unbound local variable warning for odometry_noise (it is guaranteed to be defined)
    # noinspection PyUnboundLocalVariable
    gg = GraphGenerator(
        path=GraphGenerator.PATH_ALIAS_TO_CALLABLE[args.p],
        dataset_name=args.t,
        path_args=path_arguments,
        tag_poses=GraphGenerator.TAG_DATASETS[args.t],
        t_max=args.t_max,
        n_poses=args.np,
        tag_size=ASSUMED_TAG_SIZE,
        odometry_noise=odom_noise
    )

    if args.v:
        gg.visualize()

    gg.export_to_map_processing_cache()
