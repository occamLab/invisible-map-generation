"""
Program to visualize the chi2s of each edge type, sweeped over weights
"""
import os
import sys

# Ensure that the map_processing module is imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

import argparse
import json

from firebase_admin import credentials
from matplotlib import pyplot as plt
import numpy as np

from map_processing.graph_manager import GraphManager
from map_processing import graph_opt_utils
from map_processing.graph import Graph
from map_processing.cache_manager import CacheManagerSingleton

CACHE_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../.cache", "unprocessed_maps",
                               "myTestFolder")
MAP_JSON = "279953291259OCCAM Room Lidar Aligned.json"


def make_parser():
    """Makes an argument p object for this program

    Returns:
        An argument parser
    """
    p = argparse.ArgumentParser(description="Visualize the chi2 values split by edge type")
    p.add_argument(
        '-m',
        action='store_true',
        help='Visualize the results of the chi2 metric instead of optimized map'
    )
    return p


def main():
    parser = make_parser()
    args = parser.parse_args()

    file_name = 'chi2_metric_results.json' if args.m else 'chi2_results.json'

    if os.path.isfile(file_name):
        with open(file_name, 'r') as file:
            dct = json.loads(file.read())
        sweep = np.array(dct['odom_tag_ratio'])
        full_chi2s = dct['chi2s']
        chi2s = {edge: [info['sum'] for info in full_chi2s[edge]] for edge in ('odometry', 'tag', 'gravity')}
    else:
        cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
        cms = CacheManagerSingleton(firebase_creds=cred)
        gm = GraphManager(0, cms)

        map_json_path = os.path.join(CACHE_DIRECTORY, MAP_JSON)
        with open(map_json_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()
        map_dct = json.loads(json_string)
        graph = Graph.as_graph(map_dct)

        sweep = np.exp(np.arange(-10, 10.1, 0.1))
        total_runs = sweep.shape[0]
        chi2s = {
            'gravity': [0] * total_runs,
            'odometry': [0] * total_runs,
            'tag': [0] * total_runs,
        }
        full_chi2s = {
            'gravity': [{}] * total_runs,
            'odometry': [{}] * total_runs,
            'tag': [{}] * total_runs,
            'total_chi2': [[]] * total_runs,
            'actual_chi2s': [[]] * total_runs,
        }
        subgraphs = gm.create_graphs_for_chi2_comparison(map_dct)

        for run in range(total_runs):
            optimizer = gm.optimize_and_return_optimizer(graph, sweep[run])
            if args.m:
                run_chi2s = gm.subgraph_pair_optimize_and_categorize_chi2(
                    sweep[run], subgraphs, subgraph_1_weights=GraphManager.weights_dict[
                        GraphManager.WeightSpecifier.TRUST_TAGS])
            else:
                run_chi2s = graph.get_chi2_by_edge_type(optimizer, False)
            print(f'odom:tag of e^{np.log(sweep[run]):.1f} gives: {run_chi2s}')
            for edge, chi2 in run_chi2s.items():
                chi2s[edge][run] = chi2['sum']
                full_chi2s[edge][run] = chi2
            full_chi2s['actual_chi2s'][run] = graph_opt_utils.sum_optimizer_edges_chi2(optimizer)
            full_chi2s['total_chi2'][run] = sum([run_chi2s[edge]['sum'] for edge in ('odometry', 'tag', 'gravity')])

        with open(os.path.join('saved_sweeps', 'chi2_visualization', file_name), 'w') as file:
            json.dump({
                'odom_tag_ratio': sweep.tolist(),
                'chi2s': full_chi2s
            }, file)

    # for w in chi2s:
    #     print(w)
    #     print(np.array(chi2s[w]).shape)
    #     for chi2_info in chi2s[w]:
    #         print(chi2_info)
    #     print(full_chi2s[w][0]['edges'])
    #     print()
    # print(sweep)
    # print(sweep.shape)

    last_odom = chi2s['odometry'][-1]
    last_tag = chi2s['tag'][-1]
    last_gravity = chi2s['gravity'][-1]
    odom_edges = full_chi2s['odometry'][0]['edges']
    tag_edges = full_chi2s['tag'][0]['edges']
    gravity_edges = full_chi2s['gravity'][0]['edges']
    print(last_odom)
    print(last_tag)
    print(last_gravity)
    print(last_odom + last_tag + last_gravity)
    print()
    print(f'Total edges: {odom_edges + tag_edges + gravity_edges}')

    plt.stackplot(np.log(sweep), chi2s['gravity'], chi2s['odometry'], chi2s['tag'],
                  labels=['Gravity', 'Odometry', 'Tag'])
    plt.xlabel('log(odom/tag)')
    plt.ylabel('Chi^2')
    plot_title = 'Chi2 Metric by Edge Type' if args.m else 'Resulting Chi^2 of Optimized Graph by Edge Type'
    plt.title(plot_title)
    plt.legend()
    plt.plot(np.log(sweep[int(sweep.shape[0]/2):]), np.exp(-sweep[int(sweep.shape[0]/2):]), '-b')
    plt.show()
    plt.plot(np.log(sweep), )


if __name__ == '__main__':
    main()
