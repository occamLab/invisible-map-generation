"""
Find the correlation between two metrics for weight optimization
"""

import os
import sys

from collections import OrderedDict

# Ensure that the map_processing module is imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import argparse
from firebase_admin import credentials
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

from map_processing.graph import Graph
from map_processing.graph_manager import GraphManager
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import Weights
import typing

SpearmenrResult = typing.NamedTuple("SpearmenrResult", [("correlation", float), ("pvalue", float)])

# noinspection SpellCheckingInspection
MAP_TO_ANALYZE = os.path.join("unprocessed_maps", "rawMapData", "HQV39qzyDeeuU3UQDGtcywzI9sY2",
                              "duncan-occam-room-10-1-21-2-48 26773176629225.json")

SWEEP = np.arange(-10, 10.1, 4)

SECOND_SUBGRAPH_WEIGHTS_KEY_ORDER = [
    GraphManager.WeightSpecifier.IDENTITY,
    GraphManager.WeightSpecifier.TRUST_TAGS,
    GraphManager.WeightSpecifier.TRUST_ODOM,
    GraphManager.WeightSpecifier.SENSIBLE_DEFAULT_WEIGHTS,
]


def make_parser() -> argparse.ArgumentParser:
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Find the best set of weights to use for graph optimization")
    p.add_argument(
        "-l",
        action="store_true",
        help="Load data from file stored in correlation_results.json"
    )
    return p


def do_sweeping(sweep: np.ndarray):
    """
    Args:
        sweep: Array of odometry-to-tag weight ratio values to consider.

    Returns:

    """
    total_runs = sweep.shape[0]
    single_graph_gt = np.zeros(total_runs)
    single_graph_chi2 = np.zeros(total_runs)

    cred = credentials.Certificate(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
    cms = CacheManagerSingleton(cred)
    gm = GraphManager(GraphManager.WeightSpecifier.SENSIBLE_DEFAULT_WEIGHTS, cms)

    map_info = cms.map_info_from_path(MAP_TO_ANALYZE)
    if map_info is None:
        print("Could not find the map {}".format(MAP_TO_ANALYZE))
        return
    ground_truth_dict = cms.find_ground_truth_data_from_map_info(map_info)
    if ground_truth_dict is None:
        print(f"Could not find ground truth data associated with {map_info.map_name}")
        exit(-1)

    graph = Graph.as_graph(map_info.map_dct)
    sg0, sg1 = gm.create_graphs_for_chi2_comparison(map_info.map_dct)

    subgraph_pair_chi2_diff = OrderedDict()
    for key in SECOND_SUBGRAPH_WEIGHTS_KEY_ORDER:
        subgraph_pair_chi2_diff[key] = []

    for run in range(total_runs):
        weights = Weights(
            **Weights.legacy_weight_dict_from_array(np.array([sweep[run], sweep[run], -sweep[run], -sweep[run]]))
        )
        
        print("optimizing...")

        opt_chi2 = gm.optimize_from_weights(graph, weights).chi2s.chi2_all_after
        single_graph_chi2[run] = opt_chi2

        print("standard optimization ground truth:")
        single_graph_gt[run] = gm.optimize_and_get_ground_truth_error_metric(
            weights=weights, graph=graph, ground_truth_tags=ground_truth_dict).gt_metric_opt

        print("subgraph pair optimization...")
        for second_subgraph_weights_key in subgraph_pair_chi2_diff.keys():
            print(second_subgraph_weights_key)
            subgraph_pair_chi2_diff[second_subgraph_weights_key].append(
                gm.subgraph_pair_optimize(
                    subgraph_0_weights=weights,
                    subgraphs=(sg0, sg1),
                    subgraph_1_weights=GraphManager.weights_dict[second_subgraph_weights_key],
                    verbose=True
                ).chi2_diff
            )

        print(f"An Odom to Tag ratio of {sweep[run]:.6f} gives chi2s of:")
        for second_subgraph_weights_key in subgraph_pair_chi2_diff:
            print(f"\t{second_subgraph_weights_key}: {subgraph_pair_chi2_diff[second_subgraph_weights_key][-1]},")
        print(f"\ta ground truth metric of {single_graph_gt[run]}")
        print(f"\tand an optimized chi2 of {single_graph_chi2[run]}.\n")

    # with open("saved_sweeps/metric_correlation/correlation_results.json", "w") as file:
    #     json.dump({
    #         "odom_tag_ratio": sweep.tolist(),
    #         "subgraph_pair_chi2_diff": subgraph_pair_chi2_diff,
    #         "single_graph_gt": single_graph_gt,
    #         "optimized_chi2s": single_graph_chi2,
    #     }, file, indent=2)
    return single_graph_gt, single_graph_chi2, subgraph_pair_chi2_diff


def main():
    parser = make_parser()
    args = parser.parse_args()
    sweep: np.ndarray

    if args.l:
        with open("saved_sweeps/metric_correlation/correlation_results.json", "r") as results_file:
            dct = json.loads(results_file.read())
        sweep = np.array(dct["odom_tag_ratio"])
        single_graph_gt = dct["single_graph_gt"]
        subgraph_pair_chi2_diff = dct["subgraph_pair_chi2_diff"]
        single_graph_chi2 = dct["single_graph_chi2"]
    else:
        sweep = SWEEP
        single_graph_gt, single_graph_chi2, subgraph_pair_chi2_diff = do_sweeping(SWEEP)

    stacked_data = np.vstack(
        [
            np.array(single_graph_gt),
            np.array(single_graph_chi2),
            np.array([subgraph_pair_chi2_diff[w] for w in subgraph_pair_chi2_diff])
        ]
    )

    # Disable type checking here (a can be a 1D or 2D array). With axis=1, each row represents a variable and the
    # columns contain the observations.
    # noinspection PyTypeChecker
    corr: SpearmenrResult = stats.spearmanr(a=stacked_data, axis=1)
    print(f"The correlation between gt metrics and chi2 metrics are:")
    print(corr.correlation)

    plt.plot(sweep, np.array(single_graph_gt), "-ob")
    plt.xlabel("odom/tag")
    plt.ylabel("Ground Truth Translation Metric (m)")
    plt.title("Ground truth metric")
    plt.show()

    plotted_weights = "comparison_baseline"
    plt.plot(sweep, np.log(np.array(subgraph_pair_chi2_diff[plotted_weights])), "-ob")
    plt.xlabel("odom/tag")
    plt.ylabel("log(Chi2)")
    plt.title(f"Chi2 based on {plotted_weights}")
    plt.show()


if __name__ == "__main__":
    main()
