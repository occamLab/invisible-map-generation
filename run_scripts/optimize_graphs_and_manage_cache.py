"""
Script that makes use of the GraphManager class.

Print the usage instructions:
>> python3 optimize_graphs_and_manage_cache.py -h

Example usage that listens to the unprocessed maps' database reference:
>> python3 optimize_graphs_and_manage_cache.py -f

Example usage that optimizes and plots all graphs matching the pattern specified 
by the -p flag:
>> python3 optimize_graphs_and_manage_cache.py -p "unprocessed_maps/**/*Living Room*"

Notes:
- This script was adapted from the script test_firebase_sba as of commit
  74891577511869f7cd3c4743c1e69fb5145f81e0
- The maps that are *processed* and cached are of a different format than the
  unprocessed graphs and cannot be-loaded for further processing.
"""
import os
import random
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import argparse
import numpy as np
from firebase_admin import credentials
from typing import Dict, Callable, Iterable, Any, Tuple
from collections import defaultdict

from map_processing import PrescalingOptEnum, VertexType
from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.data_models import OComputeInfParams, GTDataSet, OConfig
from map_processing.graph_opt_hl_interface import (
    holistic_optimize,
    WEIGHTS_DICT,
    WeightSpecifier,
)
from map_processing.sweep import sweep_params
import map_processing.throw_out_bad_tags as tag_filter
import map_processing

from dotenv import load_dotenv

load_dotenv()


SBA_SWEEP_CONFIG: Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.OConfigEnum.ODOM_TAG_RATIO: (np.linspace, [1, 1, 1]),
    OConfig.OConfigEnum.LIN_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.ANG_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.GRAV_MAG: (np.linspace, [1, 1, 1]),
}

NO_SBA_SWEEP_CONFIG: Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.OConfigEnum.ODOM_TAG_RATIO: (np.linspace, [1, 1, 1]),
    OConfig.OConfigEnum.LIN_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.ANG_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.TAG_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.GRAV_MAG: (np.linspace, [1, 1, 1]),
}


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Graph optimization utility for optimizing, plotting, and database"
        " upload/download",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-p",
        type=str,
        help="Pattern to match to graph names; matching graph names in cache are "
        "optimized (e.g., '-g *Living_Room*' will plot any cached map with "
        "'Living_Room' in its name). The cache directory is searched recursively, "
        "and '**/' is automatically prepended to the pattern.",
    )
    p.add_argument(
        "-u",
        action="store_true",
        help="Specifies the recursive search (with the pattern given by the -p "
        "argument) to be rooted in the cache folder's root (default is to be rooted "
        "in the unprocessed_maps/ sub-directory of the cache).",
        default=True,
    )
    p.add_argument(
        "--pso",
        type=int,
        required=False,
        help="Specifies the prescaling option used in the Graph.as_graph class method "
        "(according to the PrescalingOptEnum enum). Viable options are: "
        " 0-Sparse bundle adjustment, "
        " 1-Tag prescaling uses the full covariance matrix,"
        " 2-Tag prescaling uses only the covariance matrix diagonal,"
        " 3-Tag prescaling is a matrix of ones.",
        default=0,
        choices={0, 1, 2, 3},
    )

    weights_options = [
        f"{weight_option.value}-"
        f"'{str(weight_option)[len(WeightSpecifier.__name__) + 1:]}'"
        for weight_option in WEIGHTS_DICT.keys()
    ]
    p.add_argument(
        "-w",
        type=int,
        required=False,
        help="Specifies which weight vector to be used (maps to a weight vector which "
        "is stored as a class attribute of the GraphManager class). Viable options "
        "are: " + ", ".join(weights_options),
        default=0,
        choices={weight_option.value for weight_option in WEIGHTS_DICT.keys()},
    )
    p.add_argument(
        "-f",
        action="store_true",
        help="Acquire maps from Firebase and overwrite existing cache.",
        default=False,
    )
    p.add_argument(
        "-a",
        action="store_true",
        help="Assume cloud anchor transforms are transforms in the global frame (use for STEP Mapping data).",
        default=False,
    )
    p.add_argument(
        "-fs",
        type=str,
        required=False,
        help="Acquire maps from a specific bucket (device id) in firebase. This is done by linking"
        "your firebase device_id to your name in the firebase_device_config.json",
        default=None,
    )
    p.add_argument(
        "-fc",
        action="store_true",
        help="Find a combine maps with shared cloud anchors",
        default=None,
    )
    p.add_argument(
        "-F",
        action="store_true",
        help="Upload any graphs to Firebase that are optimized while this script is "
        "running. This option is mutually exclusive with the -c option.",
        default=False,
    )
    p.add_argument(
        "-ca",
        action="store_true",
        help="Optimize Graph with Cloud Anchors",
        default=False,
    )
    p.add_argument(
        "-c",
        action="store_true",
        help="Compare graph optimizations by computing two different optimizations "
        "for two sub-graphs of the specified graph: one where the tag vertices are "
        "not fixed, and one where they are. This option is mutually exclusive with "
        "the -F and -s flags.",
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
        default=False,
    )

    pso_options = [
        f"{pso_option.value}-'{str(pso_option)[len(PrescalingOptEnum.__name__) + 1:]}'"
        for pso_option in PrescalingOptEnum
    ]
    p.add_argument(
        "--fix",
        type=int,
        nargs="*",
        default=[],
        help="What vertex types to fix during optimization (note: tagpoints are "
        "always fixed). Otherwise," + " ,".join(pso_options),
        choices={pso_option.value for pso_option in PrescalingOptEnum},
    )
    p.add_argument(
        "--filter",
        type=float,
        required=False,
        help="Removes from the graph observation edges above this many standard "
        "deviations from the mean observation edge chi2 value in the optimized graph. "
        "The graph optimization is then re-run with the modified graph. A negative "
        "value performs no filtering.",
        default=-1.0,
    )

    p.add_argument(
        "--lvv",
        type=float,
        required=False,
        help="Magnitude of the linear velocity variance vector used for edge "
        "information matrix computation",
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
        help="Variance value used the tag (SBA) variance (i.e., noise variance in "
        "pixel space)",
        default=1.0,
    )

    p.add_argument(
        "-s",
        action="store_true",
        help="Sweep the parameters as specified by the SWEEP_CONFIG dictionary in "
        f"{map_processing.sweep.__name__}. Mutually exclusive with the -c flag.",
        default=False,
    )
    p.add_argument(
        "--sbea",
        action="store_true",
        help="(scale_by_edge_amount) Apply a multiplicative coefficient to the "
        "odom-to-tag ratio that is found by computing the ratio of the number of "
        "tag edges to odometry edges.",
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
        raise ValueError("Mutually exclusive flags with -c used")

    if args.ca and (args.pso == PrescalingOptEnum.USE_SBA.value or not args.s):
        raise ValueError(
            "Cloud Anchors are currently only supported in no SBA parameter sweeps"
        )

    # Fetch the service account key JSON file contents
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_variable is None:
        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
    else:
        cms = CacheManagerSingleton(
            firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
        )
    # Download all maps from Firebase
    if args.fs is not None:
        cms.download_maps_for_device(device_id_name=args.fs)
        exit(0)
    elif args.f:
        cms.download_all_maps()
        exit(0)
    elif args.fc:
        cms.combine_shared_maps()
        exit(0)

    map_pattern = args.p if args.p else ""
    map_pattern = map_pattern.split("+")

    map_dictionary = defaultdict(list)
    map_data = []
    id_len = 0
    add_bound = 0
    map_bounds = {}

    anchor_info = {}
    for i, map_name in enumerate(map_pattern):
        matching_map = cms.find_maps(map_name, search_restriction=0)
        if len(matching_map) == 0:
            print(
                f"No matches for {map_pattern} in recursive search of {CacheManagerSingleton.CACHE_PATH}"
            )
            exit(0)
        map_name = ""

        for map_set in matching_map:
            map_json_name = map_set.map_json_blob_name
            map_name += map_set.map_name
            map_data.append(map_set)
            anchor_info[map_set.map_name] = {}
            for pose_data in map_set.map_dct["pose_data"]:
                pose_data["id"] += id_len
            for cloud_data in map_set.map_dct["cloud_data"]:
                for instance in cloud_data:
                    instance["poseId"] += id_len
                    if not args.a:
                        fixed = (
                            np.reshape(pose_data[instance["poseId"]], (4, 4))
                            .transpose()
                            .dot(np.reshape(instance["pose"], (4, 4)).transpose())
                        )
                        instance["pose"] = np.transpose(fixed).reshape((4, 4))
                    anchor_info[map_set.map_name][
                        instance["cloudIdentifier"]
                    ] = instance["pose"]
            for key, values in map_set.map_dct.items():
                map_dictionary[key].extend(values)
            id_len += len(map_set.map_dct["pose_data"])
            map_bounds[map_set.map_name] = id_len

    if len(matching_map) > 1:
        map_json_name = args.p

    map_dictionary["map_id"] = args.p
    map_bounds = dict(sorted(map_bounds.items(), key=lambda item: item[1]))

    if len(anchor_info) > 1:
        all_anchor_ids = [set(anchor_info[anchor].keys()) for anchor in anchor_info]
        intersect = set.intersection(*all_anchor_ids)

        if len(intersect) == 0:
            print("No shared anchors between maps.")
            exit(1)
        anchor_id = random.choice(list(intersect))

        anchor_positions = {
            map_id: np.transpose(np.reshape(anchor_info[map_id][anchor_id], (4, 4)))
            for map_id in anchor_info
        }

        for pose_data in map_dictionary["pose_data"]:
            for map in map_bounds:
                if pose_data["id"] < map_bounds[map]:
                    fixed = np.linalg.inv(anchor_positions[map]).dot(
                        np.transpose(np.reshape(pose_data["pose"], (4, 4)))
                    )
                    pose_data["pose"] = list(
                        np.reshape(np.transpose(fixed), (1, 16))[0]
                    )
                    break

        for cloud_data in map_dictionary["cloud_data"]:
            for instance in cloud_data:
                for map in map_bounds:
                    if instance["poseId"] < map_bounds[map]:
                        fixed = np.linalg.inv(anchor_positions[map]).dot(
                            np.transpose(np.reshape(instance["pose"], (4, 4)))
                        )
                        instance["pose"] = list(
                            np.reshape(np.transpose(fixed), (1, 16))[0]
                        )
                        break

    complete_map = MapInfo(
        map_name=map_name,
        map_dct=map_dictionary,
        map_json_name=map_json_name,
        map_bounds=map_bounds,
    )

    # Remove tag and cloud anchor observations that are bad
    if args.t:
        complete_map.map_dct = tag_filter.throw_out_bad_tags(
            complete_map.map_dct, verbose=True
        )

    compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3) * args.lvv,
        tag_sba_var=args.tsv,
        ang_vel_var=args.avv,
    )
    gt_data = cms.find_ground_truth_data_from_map_info(map_data)
    # If you want to sweep through optimization parameters
    if args.s:
        sweep_config = NO_SBA_SWEEP_CONFIG if args.pso == 1 else SBA_SWEEP_CONFIG

        sweep_params(
            mi=complete_map,
            ground_truth_data=gt_data,
            base_oconfig=OConfig(
                is_sba=args.pso == PrescalingOptEnum.USE_SBA.value,
                compute_inf_params=compute_inf_params,
            ),
            sweep_config=sweep_config,
            ordered_sweep_config_keys=[key for key in sweep_config.keys()],
            verbose=True,
            generate_plot=True,
            show_plots=args.v,
            num_processes=args.np,
            upload_best=args.F,
            cms=cms,
            use_cloud_anchors=args.ca,
        )

    # If you simply want to run the optimizer with specified weights
    else:
        oconfig = OConfig(
            is_sba=args.pso == 0,
            weights=WEIGHTS_DICT[WeightSpecifier(args.w)],
            scale_by_edge_amount=args.sbea,
            compute_inf_params=compute_inf_params,
        )
        fixed_vertices = set()
        for tag_type in args.fix:
            fixed_vertices.add(VertexType(tag_type))
        opt_result = holistic_optimize(
            map_info=complete_map,
            pso=PrescalingOptEnum(args.pso),
            oconfig=oconfig,
            fixed_vertices=fixed_vertices,
            verbose=True,
            visualize=args.v,
            compare=args.c,
            upload=args.F,
            cms=cms,
            gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data)
            if gt_data is not None
            else None,
        )
