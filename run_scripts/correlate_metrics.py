"""
Find the correlation between two metrics for weight optimization
"""
import os
import sys

# Ensure that the map_processing module is imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import argparse
from firebase_admin import credentials
import json
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats

import map_processing.graph_opt_utils
from map_processing import OCCAM_ROOM_TAGS_DICT
from map_processing.graph import Graph
from map_processing.graph_manager import GraphManager
from map_processing.cache_manager import CacheManagerSingleton
import typing

SpearmenrResult = typing.NamedTuple("SpearmenrResult", [("correlation", float), ("pvalue", float)])


MAP_TO_ANALYZE = os.path.join("unprocessed_maps", "rawMapData", "HQV39qzyDeeuU3UQDGtcywzI9sY2",
                              "duncan-occam-room-10-1-21-2-48 26773176629225.json")

SWEEP = np.arange(-10, 10.1, 0.25)


def make_parser() -> argparse.ArgumentParser:
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Find the best set of weights to use for graph optimization")
    p.add_argument(
        '-l',
        action='store_true',
        help='Load data from file stored in correlation_results.json'
    )
    return p


def do_sweeping(sweep: np.ndarray):
    """

    Args:
        sweep:

    Returns:
        gt_metrics:
        optimized_total_chi2s: An array where each value represents the sum of the optimized graph's edges' chi2 values
         (with the optimization being performed using the SENSIBLE_DEFAULT_WEIGHTS).
    """
    total_runs = sweep.shape[0]
    gt_metrics = np.zeros(total_runs)
    optimized_total_chi2s = np.zeros(total_runs)

    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    cms = CacheManagerSingleton(cred)
    gm = GraphManager(GraphManager.WeightSpecifier.SENSIBLE_DEFAULT_WEIGHTS, cms)

    map_info = cms.map_info_from_path(MAP_TO_ANALYZE)
    if map_info is None:
        print("Could not find the map {}".format(MAP_TO_ANALYZE))
        return

    map_dct = map_info.map_dct
    graph = Graph.as_graph(map_dct)
    sg1, sg2 = gm.create_graphs_for_chi2_comparison(map_dct)

    chi2s = {
        'comparison_baseline': [],
        'trust_tags': [],
        'trust_odom': [],
        'sensible_default_weights': []
    }

    for run in range(total_runs):
        weights = map_processing.graph_opt_utils.weight_dict_from_array(np.array([sweep[run], sweep[run],
                                                                                  -sweep[run], -sweep[run]]))
        print('optimizing...')
        opt_chi2, tag_verts = gm.get_optimized_graph_info(graph, weights)
        optimized_total_chi2s[run] = opt_chi2

        print('ground truth')
        gt_metrics[run] = gm.get_ground_truth_from_optimized_tags(tag_verts, OCCAM_ROOM_TAGS_DICT)
        for weight_name in chi2s:
            print(weight_name)
            chi2s[weight_name].append(gm.get_chi2_from_subgraphs(weights, (sg1, sg2), weight_name))

        print(f'An Odom to Tag ratio of {sweep[run]:.6f} gives chi2s of:')
        for weight_name in chi2s:
            print(f'\t{weight_name}: {chi2s[weight_name][-1]},')
        print(f'\ta ground truth metric of {gt_metrics[run]}')
        print(f'\tand an optimized chi2 of {optimized_total_chi2s[run]}.\n')

    with open('saved_sweeps/metric_correlation/correlation_results.json', 'w') as file:
        json.dump({
            'odom_tag_ratio': sweep.tolist(),
            'duncan_chi2s': chi2s,
            'gt_metrics': gt_metrics,
            'optimized_chi2s': optimized_total_chi2s,
        }, file, indent=2)
    return gt_metrics, optimized_total_chi2s, chi2s


def main():
    parser = make_parser()
    args = parser.parse_args()
    sweep: np.ndarray

    if args.l:
        with open('saved_sweeps/metric_correlation/correlation_results.json', 'r') as results_file:
            dct = json.loads(results_file.read())
        sweep = np.array(dct['odom_tag_ratio'])
        gt_metrics = dct['gt_metrics']
        chi2s = dct['duncan_chi2s']
        optimized_chi2s = dct['optimized_chi2s']
    else:
        sweep = SWEEP
        gt_metrics, optimized_chi2s, chi2s = do_sweeping(SWEEP)

    stacked_data = np.vstack(
        [
            np.array(gt_metrics),
            np.array(optimized_chi2s),
            np.array([chi2s[w] for w in chi2s])
        ]
    )

    # Disable type checking here (a can be a 1D or 2D array). With axis=1, each row represents a variable and the
    # columns contain the observations.
    # noinspection PyTypeChecker
    corr: SpearmenrResult = stats.spearmanr(a=stacked_data, axis=1)
    print(f'The correlation between gt metrics and chi2 metrics are:')
    print(corr.correlation)

    plt.plot(sweep, np.array(gt_metrics), '-ob')
    plt.xlabel('odom/tag')
    plt.ylabel('Ground Truth Translation Metric (m)')
    plt.title('Ground truth metric')
    plt.show()

    plotted_weights = 'comparison_baseline'
    plt.plot(sweep, np.log(np.array(chi2s[plotted_weights])), '-ob')
    plt.xlabel('odom/tag')
    plt.ylabel('log(Chi2)')
    plt.title(f'Chi2 based on {plotted_weights}')
    plt.show()


if __name__ == '__main__':
    main()
