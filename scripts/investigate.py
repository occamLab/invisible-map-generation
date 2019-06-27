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
    optimizer = graph2Optimizer(graph)

    uposes = np.array([optimizer.vertex(i).estimate().translation()
                       for i in optimizer.vertices()]) + 1

    print(optimizer.initialize_optimization())
    print(optimizer.optimize(20))

    poses = np.array([optimizer.vertex(i).estimate().translation()
                      for i in optimizer.vertices()]) + 1

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(poses[:, 0], poses[:, 1], poses[:, 2], '.', label='Corrected')
    ax.plot(uposes[:, 0], uposes[:, 1], uposes[:, 2], '.', label='Uncorrected')
    ax.legend()
    plt.show()
    print((poses-uposes).sum())
    return optimizer


if __name__ == '__main__':
    main()
# # weights = maxweights(graph.edges)
# x = g2o.VertexSE3()
# x.set_id(2)
