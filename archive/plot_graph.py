#!/usr/bin/env python
"""
Plot the unoptimized and optimized graphs.
"""

import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from map_processing.graph_utils import optimizer_to_map


if len(sys.argv) < 2:
    FILENAME = 'converted-data/academic_center.pkl'
else:
    FILENAME = sys.argv[1]

with open(FILENAME, 'rb') as data:
    GRAPH = pickle.load(data)


def main():
    """Plot the unoptimized and optimized graphs.
    """
    GRAPH.generate_unoptimized_graph()
    GRAPH.optimize_graph()
    GRAPH.generate_maximization_params()
    GRAPH.tune_weights()
    print("Weights: ", GRAPH.weights)
    print(np.sqrt(np.exp(GRAPH.weights)) * 3)

    unoptimized_trajectory = optimizer_to_map(
        GRAPH.vertices, GRAPH.unoptimized_graph)
    optimized_trajectory = optimizer_to_map(
        GRAPH.vertices, GRAPH.optimized_graph)

    print('Optimized chi2: ', GRAPH.optimized_graph.chi2())
    weights = np.reshape([], (0, 18))
    for _ in range(0):
        GRAPH.expectation_maximization_once()
        print(GRAPH.maximization_success)
        print(GRAPH.g2o_status)
        print(GRAPH.maximization_results.message)
        weights = np.vstack([weights, GRAPH.weights])
        print(GRAPH.weights)

    fig = plt.figure()
    axes = fig.add_subplot(111, projection='3d')
    tag_marker = '^'
    waypoint_marker = 's'
    location_marker = '.'
    axes.plot(unoptimized_trajectory['locations'][:, 0],
              unoptimized_trajectory['locations'][:, 1],
              unoptimized_trajectory['locations'][:, 2], location_marker,
              label='Uncorrected Path')
    axes.plot(optimized_trajectory['locations'][:, 0],
              optimized_trajectory['locations'][:, 1],
              optimized_trajectory['locations'][:, 2], location_marker,
              label='Corrected Path')

    axes.scatter(unoptimized_trajectory['tags'][:, 0],
                 unoptimized_trajectory['tags'][:, 1],
                 unoptimized_trajectory['tags'][:, 2], marker=tag_marker,
                 s=100, label='Uncorrected Tags')
    axes.scatter(optimized_trajectory['tags'][:, 0],
                 optimized_trajectory['tags'][:, 1],
                 optimized_trajectory['tags'][:, 2], marker=tag_marker,
                 s=100, label='Corrected Tags')

    axes.plot(unoptimized_trajectory['waypoints'][:, 0],
              unoptimized_trajectory['waypoints'][:, 1],
              unoptimized_trajectory['waypoints'][:, 2], waypoint_marker,
              label='Uncorrected Waypoints')
    axes.plot(optimized_trajectory['waypoints'][:, 0],
              optimized_trajectory['waypoints'][:, 1],
              optimized_trajectory['waypoints'][:, 2], waypoint_marker,
              label='Corrected Waypoints')

    axes.legend()
    plt.show()


if __name__ == '__main__':
    main()
