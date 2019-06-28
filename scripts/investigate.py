#!/usr/bin/env python

import pickle
import numpy as np
import itertools
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
    test = copy.deepcopy(graph)

    graph.plotMap()
    graph.plotErrors()

    rowsPerMap = 100
    numAxs = graph.observations.shape[0] // graph.observations.shape[1] // rowsPerMap
    heatf, heataxs = plt.subplots(1, numAxs)
    for i, ax in enumerate(heataxs):
        ax.imshow(graph.observations[i * rowsPerMap:(i+1)*rowsPerMap])

    plt.show()


if __name__ == '__main__':
    main()
