#!/usr/bin/env python

import pickle
import numpy as np
from graph_utils import optimizer_to_map, connected_components, ordered_odometry_edges
from graph import VertexType
import matplotlib.pyplot as plt
from convert_posegraph import convert
from mpl_toolkits.mplot3d import Axes3D

# with open('data/straight_back_and_forth.pkl', 'rb') as data:
#     graph = convert(pickle.load(data, encoding='latin1'))

with open('converted-data/straight_back_and_forth.pkl', 'rb') as data:
    graph = pickle.load(data, encoding='latin1')

with open('straight_back_and_forth.pkl', 'wb') as data:
    pickle.dump(graph, data)


def checkVals1(graph):
    errs = [np.reshape([], [0, 6]) for _ in range(3)]
    for edge in graph.edges.values():
        endtype = graph.vertices[edge.enduid].mode
        if endtype == VertexType.DUMMY:
            errs[2] = np.vstack([errs[2],  edge.measurement[:6]])
        elif endtype == VertexType.ODOMETRY:
            errs[0] = np.vstack([errs[0], edge.measurement[:6]])
        elif endtype == VertexType.TAG:
            errs[1] = np.vstack([errs[1], edge.measurement[:6]])

    return np.nan_to_num(np.concatenate([np.log(np.var(x, 0)) for x in errs]))

def margin(x):
    return np.log((x / 3)**2)

print(connected_components(graph))
# with open('graph.pkl', 'wb') as data:
#     pickle.dump(graph, data)
errs = []

# graph.weights = np.array([36e4, 36e4, 36e4, 36e4, 36e4, 36e4, 0, 0, 0])

# graph.weights[6:9] = np.log(np.square(.02))
# graph.weights[9:12] = np.log(np.square(.02))
# graph.weights[0:3] = np.log(np.square(.05))
# graph.weights[3:6] = np.log(np.square(.05))

graph.weights = np.zeros(18)
graph.weights[:3] = margin(.8)
graph.weights[3:6] = margin(20)
graph.weights[6:9] = margin(.4)
graph.weights[9:12] = margin(5)
graph.update_edges()
print(graph.weights)

def main():
    graph.generate_unoptimized_graph()
    graph.optimize_graph()
    graph.generate_maximization_params()
    graph.tune_weights()
    print("Weights: ", graph.weights)
    print(np.sqrt(np.exp(graph.weights)) * 3)

    # print('Expectation: ', graph.g2o_status)
    # print('Maximization: ', graph.maximization_success)

    # global errs

    unoptimizedTrajectory = optimizer_to_map(
        graph.vertices, graph.unoptimized_graph)
    optimizedTrajectory = optimizer_to_map(
        graph.vertices, graph.optimized_graph)

    print('Optimized chi2: ', graph.optimized_graph.chi2())
    weights = np.reshape([], (0, 18))
    for _ in range(0):
        graph.expectation_maximization_once()
        print(graph.maximization_success)
        print(graph.g2o_status)
        print(graph.maximization_results.message)
        weights = np.vstack([weights, graph.weights])
        errs.append(checkVals(graph))
        print(graph.weights)

    # print(graph.maximization_results)
    # print('EM Maximization: ', graph.maximization_success)
    # print('EM chi2: ', graph.optimized_graph.chi2())

    # qxErrs = weights[:, 9]

    # for j in range(12):
    #     f, axs = plt.subplots(weights.shape[0] - 1, sharex=True)
    #     for i, ax in enumerate(axs):
    #         ax.hist(errs[i + 1][j])
    #         ax.set_title(str(j))

    # plt.show()

    # emTrajectory = optimizer_to_map(graph.vertices, graph.optimized_graph)

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
    # ax.plot(emTrajectory['locations'][:, 0], emTrajectory['locations'][:, 1],
    #         emTrajectory['locations'][:, 2], locationMarker,
    #         label='EM Corrected Path')

    ax.scatter(unoptimizedTrajectory['tags'][:, 0], unoptimizedTrajectory['tags'][:, 1],
            unoptimizedTrajectory['tags'][:, 2], marker=tagMarker, s=100, label='Uncorrected Tags')
    ax.scatter(optimizedTrajectory['tags'][:, 0], optimizedTrajectory['tags'][:, 1],
            optimizedTrajectory['tags'][:, 2], marker=tagMarker, s=100,label='Corrected Tags')
    # ax.plot(emTrajectory['tags'][:, 0], emTrajectory['tags'][:, 1],
    #         emTrajectory['tags'][:, 2], tagMarker, label='EM Corrected Tags')

    ax.plot(unoptimizedTrajectory['waypoints'][:, 0], unoptimizedTrajectory['waypoints'][:, 1],
            unoptimizedTrajectory['waypoints'][:, 2], waypointMarker,
            label='Uncorrected Waypoints')
    ax.plot(optimizedTrajectory['waypoints'][:, 0], optimizedTrajectory['waypoints'][:, 1],
            optimizedTrajectory['waypoints'][:, 2], waypointMarker,
            label='Corrected Waypoints')
    # ax.plot(emTrajectory['waypoints'][:, 0], emTrajectory['waypoints'][:, 1],
    #         emTrajectory['waypoints'][:, 2], waypointMarker,
    #         label='EM Corrected Waypoints')

    ax.legend()

    # weightsF, weightsAx = plt.subplots()
    # weightmap = weightsAx.imshow(weights[:, :12])
    # plt.colorbar(weightmap)

    plt.show()


def checkVals(graph):
    errs = [[] for _ in range(18)]
    assignments = [np.where(i == 1)[0][0] for i in graph.observations]

    for i, assignment in enumerate(assignments):
        errs[assignment].append(graph.errors[i])

    errs = [errs[i] for i in range(12)]
    return errs


# if __name__ == '__main__':
#     main()
