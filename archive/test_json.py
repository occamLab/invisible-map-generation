import json

import numpy as np
from map_processing import graph_utils, as_graph
import matplotlib.pyplot as plt

# from mpl_toolkits.mplot3d import Axes3D

# Testing code
with open("data/round1.json", "r") as f:
    x = json.load(f)

test_graph = as_graph.as_graph(x, prescaling_opt=as_graph.PrescalingOptEnum.FULL_COV)

test_graph.weights = np.array(
    [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -1e1,
        -1e1,
        1e1,
    ]
)

# Load these weights into the graph
test_graph.update_edges()

# Create the g2o object and optimize
test_graph.generate_unoptimized_graph()
test_graph.optimize_graph()

# Change vertex estimates based off the optimized graph
test_graph.update_vertices()

resulting_map = graph_utils.optimizer_to_map(
    test_graph.vertices, test_graph.optimized_graph
)
locations = resulting_map["locations"]
tag_verts = resulting_map["tags"]
# tag_verts =
# tags = resulting_map['tags']

edges = test_graph.get_ordered_odometry_edges()[0]
path = test_graph.integrate_path(
    edges,
    [
        2.38298111e01,
        6.18518412e-01,
        -2.23812237e01,
        -1.15648886e-02,
        1.37184479e-01,
        7.07669616e-01,
        -6.93001000e-01,
    ],
)

tags = test_graph.get_tags_all_position_estimate()

f = plt.figure()
f.add_subplot(111, projection="3d")
plt.plot(
    locations[:, 0], locations[:, 1], locations[:, 2], ".", c="b", label="Odom Vertices"
)
plt.plot(
    tag_verts[:, 0], tag_verts[:, 1], tag_verts[:, 2], "o", c="r", label="Tag Vertices"
)
plt.plot(tags[:, 0], tags[:, 1], tags[:, 2], ".", c="g", label="All Tag Edges")
plt.legend()
# plt.plot(path[::10, 0], path[::10, 1], path[::10, 2], '.')
plt.savefig("optimized.png")
plt.show()
