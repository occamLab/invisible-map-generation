"""
Find the correlation between two metrics for weight optimization
"""
import argparse
from firebase_admin import credentials
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import stats

import graph_utils
from as_graph import as_graph
from GraphManager import GraphManager
from graph_utils import occam_room_tags

CACHE_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)),".cache","unprocessed_maps","myTestFolder")
MAP_JSON = "2900094388220836-17-21 OCCAM Room.json"


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


def main():
    parser = make_parser()
    args = parser.parse_args()

    if args.l:
        with open('correlation_results.json', 'r') as results_file:
            dct = json.loads(results_file.read())
        sweep = np.array(dct['odom_tag_ratio'])
        gt_metrics = dct['gt_metrics']
        chi2s = dct['chi2s']
    else:
        sweep = np.exp(np.arange(-10, 10.1, 0.1))
        total_runs = sweep.shape[0]

        cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        gm = GraphManager(0, cred)
        map_json_path = os.path.join(CACHE_DIRECTORY, MAP_JSON)
        with open(map_json_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()
        map_dct = json.loads(json_string)
        graph = as_graph(map_dct)
        sg1, sg2 = gm.create_graphs_for_chi2_comparison(map_dct)

        gt_metrics = [0.0] * total_runs
        optimized_chi2s = [0.0] * total_runs
        chi2s = {
            'comparison_baseline': []
        }
        for run in range(total_runs):
            weights = graph_utils.weights_from_ratio(sweep[run])
            print('gt')
            gt_metrics[run] = gm.get_ground_truth_from_graph(weights, graph, occam_room_tags)
            print('optimized chi2')
            optimized_chi2s[run] = gm.get_optimized_graph_info(graph, weights)[0]
            for weight_name in chi2s:
                print(weight_name)
                chi2s[weight_name].append(gm.get_chi2_from_subgraphs(weights, (sg1, sg2), weight_name))

            print(f'An Odom to Tag ratio of {sweep[run]:.6f} gives chi2s of:')
            for weight_name in chi2s:
                print(f'\t{weight_name}: {chi2s[weight_name][-1]},')
            print(f'\ta ground truth metric of {gt_metrics[run]}')
            print(f'\tand an optimized chi2 of {optimized_chi2s[run]}.\n')

        with open('correlation_results.json', 'w') as file:
            json.dump({
                'odom_tag_ratio': sweep.tolist(),
                'chi2s': chi2s,
                'gt_metrics': gt_metrics,
                'optimized_chi2s': optimized_chi2s,
            }, file)

    corr = stats.spearmanr(np.vstack((np.array(gt_metrics), np.array([chi2s[w] for w in chi2s]))), axis=1)
    print(f'The correlation between gt metrics and chi2 metrics are is:')
    print(corr)

    plt.plot(np.log(sweep), np.array(gt_metrics), '-ob')
    plt.xlabel('log(odom/tag)')
    plt.ylabel('Ground Truth Translation Metric (m)')
    plt.title('Ground truth metric')
    plt.show()

    plotted_weights = 'comparison_baseline'
    plt.plot(np.log(sweep), np.log(np.array(chi2s[plotted_weights])), '-ob')
    plt.xlabel('log(odom/tag)')
    plt.ylabel('log(Chi2)')
    plt.title(f'Chi2 based on {plotted_weights}')
    plt.show()


if __name__ == '__main__':
    main()
