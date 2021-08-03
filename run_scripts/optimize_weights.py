"""
Uses a genetic algorithm to optimize the weights for the graph optimization
"""

import argparse
import json

import numpy as np
from firebase_admin import credentials
import os

from map_processing import graph_utils
from map_processing.graph_manager import GraphManager

CACHE_DIRECTORY = os.path.join("unprocessed_maps", "rawMapData")
MAP_JSON = "127027593745666Partial MAC Multiple Plane Detection 7-19-21.json"


def make_parser():
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Find the best set of weights to use for graph optimization")
    p.add_argument(
        '-s',
        type=float,
        nargs='*',
        action='store',
        help='If included, will sweep the parameters instead of running a genetic model. Optionally, add arguments '
             'for bounds and step size. Default bounds are (-10, 10). Default step size is 0.5. One or two arguments '
             'specifies the lower and upper bound. Three specifies the bounds and step size. Any more will be ignored.'
    )
    p.add_argument(
        '-l',
        action='store_true',
        help='Load data from file stored in sweep_results.json'
    )
    p.add_argument(
        '-v',
        action='store_true',
        help='Visualize plots and provide chi2 output. Note that running the genetic model will produce progress and '
             'a plot whether or not this is given.'
    )
    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    graph_manager = GraphManager(0, cred)
    map_json_path = os.path.join(CACHE_DIRECTORY, MAP_JSON)

    if args.s is not None:
        if args.l:
            with open('saved_sweeps/weight_sweep/sweep_results.json', 'r') as results_file:
                dct = json.loads(results_file.read())
                sweep_range = np.asarray(dct['sweep_range'])
                metrics = np.asarray(dct['metrics'])
                if args.v:
                    for i1, w1 in enumerate(sweep_range):
                        for i2, w2 in enumerate(sweep_range):
                            print(f'[{w1}, {w2}]: {metrics[i1, i2]}')
                    filtered_metrics = metrics == -1
                    reprocessed_metrics = metrics + 1e5 * filtered_metrics
                    best_metric = reprocessed_metrics.min()
                    best_weights = [np.log(sweep_range[i[0]]) for i in np.where(metrics == best_metric)]
                    print(f'\nBEST METRIC: {best_weights}: {best_metric}')
                    graph_utils.plot_metrics(sweep_range, reprocessed_metrics, True, True)
        else:
            bounds = (-10, 10)
            step = 0.5
            if len(args.s) == 1:
                bounds = (-args.s[0], args.s[0])
            elif len(args.s) > 1:
                bounds = (args.s[0], args.s[1])
                if len(args.s) > 2:
                    step = args.s[2]
            sweep = np.exp(np.arange(bounds[0], bounds[1], step))
            metrics = graph_manager.sweep_weights(map_json_path, sweep=sweep, verbose=args.v, visualize=False)
            with open('saved_sweeps/weight_sweep/sweep_results.json', 'w') as results_file:
                mesh_grid = [sweep.tolist()] * sweep.size
                dct = {
                    'odom_tag_ratio': mesh_grid,
                    'pose_orientation_ratio': np.array(mesh_grid).transpose().tolist(),
                    'metrics': metrics.tolist()
                }
                json.dump(dct, results_file)
            graph_utils.plot_metrics(sweep, metrics, True, True)
    else:
        if args.l:
            print('l flag must be included with s flag - ignoring l flag.')
        print(graph_manager.optimize_weights(map_json_path, verbose=args.v))

