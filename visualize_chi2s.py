"""
Program to visualize the chi2s of each edge type, sweeped over weights
"""
import json
from matplotlib import pyplot as plt
import numpy as np
import os

import graph_utils
from as_graph import as_graph
from graph import Graph

CACHE_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)),".cache","unprocessed_maps","myTestFolder")
MAP_JSON = "1448048273429586-17-21 MAC 3 Loops.json"


def main():
    if os.path.isfile('chi2_results.json'):
        with open('chi2_results.json', 'r') as file:
            dct = json.loads(file.read())
        sweep = np.array(dct['odom_tag_ratio'])
        full_chi2s = dct['chi2s']
        chi2s = {edge: [info['sum'] for info in full_chi2s[edge]] for edge in ('odometry', 'tag', 'dummy')}
    else:
        map_json_path = os.path.join(CACHE_DIRECTORY, MAP_JSON)
        with open(map_json_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()
        map_dct = json.loads(json_string)
        graph = as_graph(map_dct)

        sweep = np.exp(np.arange(-10, 10.1, 0.1))
        total_runs = sweep.shape[0]
        chi2s = {
            'dummy': [0] * total_runs,
            'odometry': [0] * total_runs,
            'tag': [0] * total_runs,
        }
        full_chi2s = {
            'dummy': [{}] * total_runs,
            'odometry': [{}] * total_runs,
            'tag': [{}] * total_runs,
            'total_chi2': [[]] * total_runs,
            'actual_chi2s': [[]] * total_runs,
        }
        for run in range(total_runs):
            weights = graph_utils.weights_from_ratio(sweep[run])
            graph.weights = weights
            graph.update_edges()
            optimizer = graph.graph_to_optimizer()
            optimizer.initialize_optimization()
            optimizer.optimize(1024)
            run_chi2s = graph.get_chi2_by_edge_type(optimizer, False)
            print(f'odom:tag of e^{np.log(sweep[run]):.1f} gives: {run_chi2s}')
            for edge, chi2 in run_chi2s.items():
                chi2s[edge][run] = chi2['sum']
                full_chi2s[edge][run] = chi2
            full_chi2s['actual_chi2s'][run] = Graph.check_optimized_edges(optimizer)
            full_chi2s['total_chi2'][run] = sum([run_chi2s[edge]['sum'] for edge in ('odometry', 'tag', 'dummy')])

        with open('chi2_results.json', 'w') as file:
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
    last_dummy = chi2s['dummy'][-1]
    odom_edges = full_chi2s['odometry'][0]['edges']
    tag_edges = full_chi2s['tag'][0]['edges']
    dummy_edges = full_chi2s['dummy'][0]['edges']
    print(last_odom)
    print(last_tag)
    print(last_dummy)
    print(last_odom + last_tag + last_dummy)
    print()
    print(f'Total edges: {odom_edges + tag_edges + dummy_edges}')

    plt.stackplot(np.log(sweep), chi2s['dummy'], chi2s['odometry'], chi2s['tag'], labels=['Dummy', 'Odometry', 'Tag'])
    plt.xlabel('log(odom/tag)')
    plt.ylabel('Chi^2')
    plt.title("Resulting Chi^2 of Optimized Graph by Edge Type")
    plt.legend()
    plt.show()

    plt.plot(np.log(sweep), )


if __name__ == '__main__':
    main()