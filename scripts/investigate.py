from graph import VertexType, graph2Optimizer
import pickle
import g2o
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import maximization_model

with open('test.pkl', 'rb') as data:
    # graph = pickle.load(data, encoding='latin1')
    graph = pickle.load(data)
with open('test.pkl', 'wb') as data:
    pickle.dump(graph, data)


def main():
    graph.optimizeGraph()
    graph.generateUnoptimizedGraph()
    graph.optimizeGraph()

    fig = graph.plotMap()
    plt.show()


if __name__ == '__main__':
    main()
