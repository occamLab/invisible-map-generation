from graph_utils import  get_subgraph, optimizer_to_map
import pickle
import matplotlib.pyplot as plt
from metrics import plot_tags_distance_diff_in_maps
from mpl_toolkits.mplot3d import Axes3D

with open('converted-data/academic_center.pkl', 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
optimizer = graph.unoptimized_graph
graph_map = optimizer_to_map(graph.vertices, optimizer)

subgraph = get_subgraph(graph, 588,1500)
subgraph.generate_unoptimized_graph()
subgraph.optimize_graph()
optimized_graph = subgraph.optimized_graph
unoptimized_graph = subgraph.unoptimized_graph
subgraph_map_optimized = optimizer_to_map(subgraph.vertices, optimized_graph)
subgraph_map_unoptimized = optimizer_to_map(subgraph.vertices, unoptimized_graph)
plot_tags_distance_diff_in_maps(subgraph_map_optimized, subgraph_map_unoptimized)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# tagMarker = '^'
# locationMarker = '.'
#
# ax.plot(subgraph_map_optimized['locations'][:, 0], subgraph_map_optimized['locations'][:, 1],
#         subgraph_map_optimized['locations'][:, 2], locationMarker, c = 'r',
#         label='Corrected Path', markersize= 20)
# ax.scatter(subgraph_map_optimized['tags'][:, 0], subgraph_map_optimized['tags'][:, 1],
#            subgraph_map_optimized['tags'][:, 2], marker=tagMarker, c='r', s=300, label='Corrected Tags')
#
# ax.plot(graph_map['locations'][:, 0], graph_map['locations'][:, 1],
#         graph_map['locations'][:, 2], '.', c='b',
#         label='Corrected Path')
# ax.scatter(graph_map['tags'][:, 0], graph_map['tags'][:, 1],
#            graph_map['tags'][:, 2], c = 'b', marker='^', s=100, label='Corrected Tags')
#
# ax.legend()
#
# # weightsF, weightsAx = plt.subplots()
# # weightmap = weightsAx.imshow(weights[:, :12])
# # plt.colorbar(weightmap)
#
# plt.show()