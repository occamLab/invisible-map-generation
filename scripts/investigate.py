#!/usr/bin/env python

import pickle
import numpy as np
import itertools
from graph import optimizer2map
import matplotlib.pyplot as plt
from convert_posegraph import convert

# with open('academic_center.pkl', 'rb') as data:
#     graph = convert(pickle.load(data, encoding='latin1'))

with open('graph.pkl', 'rb') as data:
    graph = pickle.load(data)

# with open('graph.pkl', 'wb') as data:
#     pickle.dump(graph, data)


def main():
    graph.optimizeGraph()
    graph.generateUnoptimizedGraph()
    graph.optimizeGraph()
    graph.generateMaximizationParams()
    graph.tuneWeights()
    print('Expectation: ', graph.g2oStatus)
    print('Maximization: ', graph.maximizationSuccess)

    unoptimizedTrajectory = optimizer2map(
        graph.vertices, graph.unoptimizedGraph)
    optimizedTrajectory = optimizer2map(graph.vertices, graph.optimizedGraph)

    print('Optimized chi2: ', graph.optimizedGraph.chi2())
    weights = np.reshape([], (0, 18))
    for _ in range(6):
        graph.emOnce()
        print(graph.maximizationSuccess)
        print(graph.g2oStatus)
        print(graph.maximizationResults.message)
        weights = np.vstack([weights, graph.weights])
        print(weights[-1])

    print(graph.maximizationResults)
    print('EM Maximization: ', graph.maximizationSuccess)
    print('EM chi2: ', graph.optimizedGraph.chi2())

    emTrajectory = optimizer2map(graph.vertices, graph.optimizedGraph)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tagMarker = '^'
    waypointMarker = 's'
    locationMarker = '.'
    ax.plot(unoptimizedTrajectory['locations'][:, 0], unoptimizedTrajectory['locations'][:, 1],
            unoptimizedTrajectory['locations'][:, 2], locationMarker,
            label='Uncorrected Path')
    ax.plot(optimizedTrajectory['locations'][:, 0], optimizedTrajectory['locations'][:, 1],
            optimizedTrajectory['locations'][:, 2], locationMarker,
            label='Corrected Path')
    ax.plot(emTrajectory['locations'][:, 0], emTrajectory['locations'][:, 1],
            emTrajectory['locations'][:, 2], locationMarker,
            label='EM Corrected Path')

    ax.plot(unoptimizedTrajectory['tags'][:, 0], unoptimizedTrajectory['tags'][:, 1],
            unoptimizedTrajectory['tags'][:, 2], tagMarker, label='Uncorrected Tags')
    ax.plot(optimizedTrajectory['tags'][:, 0], optimizedTrajectory['tags'][:, 1],
            optimizedTrajectory['tags'][:, 2], tagMarker, label='Corrected Tags')
    ax.plot(emTrajectory['tags'][:, 0], emTrajectory['tags'][:, 1],
            emTrajectory['tags'][:, 2], tagMarker, label='EM Corrected Tags')

    ax.plot(unoptimizedTrajectory['waypoints'][:, 0], unoptimizedTrajectory['waypoints'][:, 1],
            unoptimizedTrajectory['waypoints'][:, 2], waypointMarker,
            label='Uncorrected Waypoints')
    ax.plot(optimizedTrajectory['waypoints'][:, 0], optimizedTrajectory['waypoints'][:, 1],
            optimizedTrajectory['waypoints'][:, 2], waypointMarker,
            label='Corrected Waypoints')
    ax.plot(emTrajectory['waypoints'][:, 0], emTrajectory['waypoints'][:, 1],
            emTrajectory['waypoints'][:, 2], waypointMarker,
            label='EM Corrected Waypoints')

    ax.legend()

    weightsF, weightsAx = plt.subplots()
    weightmap = weightsAx.imshow(weights[:, :12])
    plt.colorbar(weightmap)

    plt.show()


def checkVals():
    graph.generateUnoptimizedGraph()
    graph.optimizeGraph()
    graph.generateMaximizationParams()
    errs = [[] for _ in range(18)]
    assignments = [np.where(i == 1)[0][0] for i in graph.observations]

    for i, assignment in enumerate(assignments):
        errs[assignment].append(graph.errors[i])

    variance = [(np.array(err)**2).sum() for err in errs]
    variance = np.array(variance) / [len(err) for err in errs]

    print(variance)
    graph.tuneWeights()
    print(np.exp(graph.weights))
    print(variance - np.exp(graph.weights))

    return variance


if __name__ == '__main__':
    main()
