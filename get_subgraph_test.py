from graph_utils import  get_subgraph, optimizer_to_map, get_tags_all_position_estimate
import pickle
import matplotlib.pyplot as plt
from metrics import plot_tags_distance_diff_in_maps
from mpl_toolkits.mplot3d import Axes3D
#data = 'converted-data/tttest.pkl'
data = "converted-data/test1210.pkl"
#data = 'converted-data/academic_center.pkl'
with open(data, 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
optimizer = graph.unoptimized_graph
graph_map = optimizer_to_map(graph.vertices, optimizer)

subgraph = get_subgraph(graph, 588, 4000)
subgraph.generate_unoptimized_graph()
subgraph.optimize_graph()
optimized_graph = subgraph.optimized_graph
unoptimized_graph = subgraph.unoptimized_graph
subgraph_map_optimized = optimizer_to_map(subgraph.vertices, optimized_graph)
subgraph_map_unoptimized = optimizer_to_map(subgraph.vertices, unoptimized_graph)

tags_estimate = get_tags_all_position_estimate(graph, 588, 4000)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tagMarker = '^'
locationMarker = '.'

ax.plot(subgraph_map_optimized['locations'][:, 0], subgraph_map_optimized['locations'][:, 1],
        subgraph_map_optimized['locations'][:, 2], locationMarker, c = 'r',
        label='Corrected Path', markersize= 10)
ax.scatter(subgraph_map_optimized['tags'][:, 0], subgraph_map_optimized['tags'][:, 1],
           subgraph_map_optimized['tags'][:, 2], marker=tagMarker, c='r', s=100, label='Corrected Tags')

ax.plot(graph_map['locations'][:, 0], graph_map['locations'][:, 1],
        graph_map['locations'][:, 2], '.', c='b',
        label='UnCorrected Path')

ax.scatter(graph_map['tags'][:, 0], graph_map['tags'][:, 1],
           graph_map['tags'][:, 2], c = 'b', marker='^', s=50, label='UnCorrected Tags')

ax.scatter(tags_estimate[:, 0], tags_estimate[:, 1],
           tags_estimate[:, 2], marker=tagMarker, c='g', s=20, label='UnCorrected Tags')

ax.legend()

plt.show()

# plot_tags_distance_diff_in_maps(subgraph_map_optimized, subgraph_map_unoptimized)