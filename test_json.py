import json

import convert_json
import graph_utils
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Testing code
with open('round1.json', 'r') as f:
    x = json.load(f)

y = convert_json.as_graph(x)

y.generate_unoptimized_graph()
resulting_map = graph_utils.optimizer_to_map(y.vertices, y.unoptimized_graph)
locations = resulting_map['locations']
# tags = resulting_map['tags']

edges = graph_utils.ordered_odometry_edges(y)[0]
path = graph_utils.integrate_path(y, edges, [ 2.38298111e+01,  6.18518412e-01, -2.23812237e+01, -1.15648886e-02,
        1.37184479e-01,  7.07669616e-01, -6.93001000e-01])

tags = graph_utils.get_tags_all_position_estimate(y)

f = plt.figure()
f.add_subplot(111, projection='3d')
plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], '.')
# plt.plot(tags[:, 0], tags[:, 1], tags[:, 2], '.')
plt.plot(tags[:, 0], tags[:, 1], tags[:, 2], '.')
plt.plot(path[::10, 0], path[::10, 1], path[::10, 2], '.')
plt.show()
