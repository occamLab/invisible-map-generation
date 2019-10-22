import pickle
import itertools
import numpy as np
import g2o
from graph_utils import optimizer_to_map, connected_components, ordered_odometry_edges
from graph import VertexType
import matplotlib.pyplot as plt
from convert_posegraph import convert
from mpl_toolkits.mplot3d import Axes3D

with open('converted-data/academic_center.pkl', 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
unoptimized_map = optimizer_to_map(graph.vertices, graph.unoptimized_graph)

graph.optimize_graph()
optimized_map = optimizer_to_map(graph.vertices, graph.optimized_graph)


def generate_transformations(positions):
    '''This may be difficult, as querying a transform a-> b in a
    dictionary with the transform b->a must return the inverse of
    b->a. It may be useful to make a separate clone of optimize_to_map
    which makes a hashmap of vertex uids.'''
    pass
    # pairs = list(itertools.combinations(positions, 2))

    # transformations
    # for pair in pairs:


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(unoptimized_map['locations'][:, 0], unoptimized_map['locations']
        [:, 1], unoptimized_map['locations'][:, 2])
plt.show()
