import graph
from convert_posegraph import convert
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


with open('../data/optimized_data/data_optimized.pkl', 'rb') as data:
    # with open('../data/raw_data/academic_center.pkl', 'rb') as data:
    testgraph = pickle.load(data)

graph = convert(testgraph)
trajectory, rotation = graph.uncorrectedTrajectories()[0]


# print(graph.vertices)
odomgraph = graph.odometryGraph().connectedComponents()[0]
positions = np.array(
    [odomgraph.vertices[vertex].value for vertex in list(odomgraph.vertices)])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1],
        trajectory[:, 2], '-o', label='Integrated')
ax.plot(positions[:, 0], positions[:, 1],
        positions[:, 2], '-o', label='Actual')

with open('test.pkl', 'wb') as data:
    pickle.dump(graph, data)

ax.legend()
plt.show()
