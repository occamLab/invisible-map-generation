from graph_utils import get_subgraph, optimizer_to_map, get_tags_all_position_estimate, ordered_odometry_edges, integrate_path
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = "converted-data/test_work.pkl"

with open(data, 'rb') as data:
    graph = pickle.load(data)

graph = get_subgraph(graph, 1000, 1500)
graph.generate_unoptimized_graph()
optimizer = graph.unoptimized_graph
graph_map = optimizer_to_map(graph.vertices, optimizer)


tags_estimate = get_tags_all_position_estimate(graph)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tagMarker = '^'
locationMarker = '.'

ax.scatter(graph_map['tags'][:, 0], graph_map['tags'][:, 1],
           graph_map['tags'][:, 2], marker=tagMarker, s=100, label='Tags')

ax.plot(graph_map['locations'][:, 0], graph_map['locations'][:, 1],
        graph_map['locations'][:, 2], '.', c='b',
        label='UnCorrected Path')

ax.plot(tags_estimate[:, 0], tags_estimate[:, 1],
        tags_estimate[:, 2], '.',
        label='Estimated Tags')


ax.legend()

plt.show()
