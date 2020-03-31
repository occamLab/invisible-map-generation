from graph_utils import get_subgraph, optimizer_to_map, get_tags_all_position_estimate, ordered_odometry_edges, integrate_path
import pickle
import matplotlib.pyplot as plt
from metrics import plot_tags_distance_diff_in_maps
# from mpl_toolkits.mplot3d import Axes3D
#data = 'converted-data/tttest.pkl'
data = "converted-data/test_work.pkl"
#data = 'converted-data/academic_center.pkl'
with open(data, 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
optimizer = graph.unoptimized_graph
graph_map = optimizer_to_map(graph.vertices, optimizer)


tags_estimate = get_tags_all_position_estimate(graph)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tagMarker = '^'
locationMarker = '.'

ordered_edges = ordered_odometry_edges(graph)[0]
integrated_path = integrate_path(graph, ordered_edges)

# ax.plot(subgraph_map_optimized['locations'][:, 0], subgraph_map_optimized['locations'][:, 1],
#         subgraph_map_optimized['locations'][:, 2], locationMarker, c = 'r',
#         label='Corrected Path', markersize= 10)
ax.scatter(graph_map['tags'][:, 0], graph_map['tags'][:, 1],
           graph_map['tags'][:, 2], marker=tagMarker, s=100, label='Tags')

ax.plot(graph_map['locations'][:, 0], graph_map['locations'][:, 1],
        graph_map['locations'][:, 2], '.', c='b',
        label='UnCorrected Path')

ax.plot(tags_estimate[:, 0], tags_estimate[:, 1],
        tags_estimate[:, 2], '.',
        label='Estimated Tags')
# f, (ax1, ax2) = plt.subplots(2)
# ax.scatter(tags_estimate[:, 0], tags_estimate[:, 1], marker=tagMarker, c='g', s=20, label='Tags')
# ax2.scatter(tags_estimate[:, 0], tags_estimate[:, 1], marker=tagMarker, c='g', s=20, label='Tags')
# ax2.scatter(graph_map['tags'][:, 0], graph_map['tags'][:, 1], c='b', marker='^', s=50, label='Tag Anchor')

integrated_path[:, 0]
# ax.plot(integrated_path[:, 0], integrated_path[:, 1], integrated_path[:, 2])

ax.legend()

plt.show()

# plot_tags_distance_diff_in_maps(subgraph_map_optimized, subgraph_map_unoptimized)
