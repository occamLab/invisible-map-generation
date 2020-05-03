import json

import convert_json
import numpy as np
import graph_utils
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# Testing code
with open('data/round1.json', 'r') as f:
    x = json.load(f)

test_graph = convert_json.as_graph(x)

# with open('converted-data/straight_back_and_forth.pkl', 'rb') as data:
#     test_graph = pickle.load(data, encoding='latin1')


weights = np.array([
    # -1, -1, -1, -.5, -.5, -.5,
    0.,  0.,  0., 0.,  0.,  0.,
    # 0.,  0.,  0., 0.,  0.,  0.,
    # -1e16, -1e16, -1e16, -1e4, -1e4, -1e4,
    1e32, 1e32, 1e32, 1e32, 1e32, 1e32,
    0.,  0.,  0., 0.,  0.,  0.
])

test_graph.update_edges

test_graph.generate_unoptimized_graph()
# test_graph.optimized_graph = test_graph.unoptimized_graph
test_graph.optimize_graph()
test_graph.update_vertices()

resulting_map = graph_utils.optimizer_to_map(
    test_graph.vertices, test_graph.unoptimized_graph)
locations = resulting_map['locations']
tag_verts = resulting_map['tags']
# tag_verts = 
# tags = resulting_map['tags']

edges = graph_utils.ordered_odometry_edges(test_graph)[0]
path = graph_utils.integrate_path(test_graph, edges, [
    2.38298111e+01,  6.18518412e-01, - 2.23812237e+01,
    -1.15648886e-02, 1.37184479e-01,  7.07669616e-01, -6.93001000e-01
])

tags = graph_utils.get_tags_all_position_estimate(test_graph)

f = plt.figure()
f.add_subplot(111, projection='3d')
plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], '.', c='b', label='Odom Vertices')
plt.plot(tag_verts[:, 0], tag_verts[:, 1], tag_verts[:, 2], 'o', c='r', label='Tag Vertices')
plt.plot(tags[:, 0], tags[:, 1], tags[:, 2], '.', c='g', label='All Tag Edges')
plt.legend()
# plt.plot(path[::10, 0], path[::10, 1], path[::10, 2], '.')
plt.savefig('demo2.png')
plt.show()
