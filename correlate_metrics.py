"""
Find the correlation between two metrics for weight optimization
"""
import argparse
from firebase_admin import credentials
from g2o import SE3Quat
import json
from matplotlib import pyplot as plt
import numpy as np
import os
from scipy import stats

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
        odom_sweep = np.array(dct['odometry'])
        tag_sweep = np.array(dct['tag'])
        sweep = np.vstack((odom_sweep, tag_sweep)).transpose()
        gt_metrics = dct['gt_metrics']
        chi2s = dct['chi2s']
    else:
        odom_sweep = np.arange(-10, 10, 0.1)
        tag_sweep = -odom_sweep  # for 2-D data, it is symmetrical across y=-x
        sweep = np.vstack((odom_sweep, tag_sweep)).transpose()
        total_runs = sweep.shape[0]

        cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        gm = GraphManager(0, cred)
        map_json_path = os.path.join(CACHE_DIRECTORY, MAP_JSON)
        with open(map_json_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()
        map_dct = json.loads(json_string)
        graph = as_graph(map_dct)
        sg1, sg2 = gm.create_graphs_for_weight_comparison(map_dct)

        gt_metrics = [0.0] * total_runs
        chi2s = {
            'sensible_default_weights': [],
            'trust_odom': [],
            'trust_tags': [],
            'genetic_results': [],
            'comparison_baseline': []
        }
        for run in range(total_runs):
            base_weights = sweep[run]
            weights = {
                'odometry': np.array([base_weights[0]] * 6),
                'tag_sba': np.array([base_weights[1]] * 2),
                'tag': np.array([base_weights[1]] * 6),
                'dummy': np.array([-1, 1e2, -1])
            }

            gt_metrics[run] = gm.get_ground_truth_from_graph(weights, graph, occam_room_tags)
            for weight_name in chi2s:
                chi2s[weight_name].append(gm.get_chi2_from_subgraphs(base_weights, sg1, sg2, weight_name))

            print(f'Odom: {base_weights[0]}, Tag: {base_weights[1]} gives chi2s of:')
            for weight_name in chi2s:
                print(f'\t{weight_name}: {chi2s[weight_name][-1]}')
            print(f'\tand a ground truth metric of {gt_metrics[run]}\n')

        with open('correlation_results.json', 'w') as file:
            json.dump({
                'odometry': odom_sweep.tolist(),
                'tag': tag_sweep.tolist(),
                'chi2s': chi2s,
                'gt_metrics': gt_metrics
            }, file)
        print(np.array(gt_metrics).shape)
        print(np.array([chi2s[w] for w in chi2s]).shape)

    corr = stats.spearmanr(np.vstack((np.array(gt_metrics), np.array([chi2s[w] for w in chi2s]))), axis=1)
    print(f'The correlation between gt metrics and chi2 is:')
    print(corr[0])

    plt.plot(odom_sweep, np.array(gt_metrics), '-ob')
    plt.xlabel('Odometry weights')
    plt.ylabel('Ground Truth Translation Metric (m)')
    plt.title('Ground truth metric by odometry weights')
    plt.show()

    plotted_weights = 'trust_tags'
    plt.plot(odom_sweep, np.log(np.array(chi2s[plotted_weights])), '-ob')
    plt.xlabel('Odometry weights')
    plt.ylabel('Chi2')
    #plt.ylim((-500, 4000))
    plt.title(f'Chi2 by odometry weights based on {plotted_weights}')
    plt.show()


if __name__ == '__main__':
    main()
