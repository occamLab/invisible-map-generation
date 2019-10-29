from graph_utils import  get_subgraph, optimizer_to_map
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

with open('converted-data/academic_center.pkl', 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
optimizer = graph.unoptimized_graph
graph_map = optimizer_to_map(graph.vertices, optimizer)


subgraph = get_subgraph(graph, 588,2000)
subgraph.generate_unoptimized_graph()


subgraph_map = optimizer_to_map(subgraph.vertices, subgraph.unoptimized_graph)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
tagMarker = '^'
locationMarker = '.'

ax.plot(subgraph_map['locations'][:, 0], subgraph_map['locations'][:, 1],
        subgraph_map['locations'][:, 2], locationMarker, c = 'r',
        label='Corrected Path', markersize= 20)
ax.scatter(subgraph_map['tags'][:, 0], subgraph_map['tags'][:, 1],
           subgraph_map['tags'][:, 2], marker=tagMarker, c='r', s=300, label='Corrected Tags')

ax.plot(graph_map['locations'][:, 0], graph_map['locations'][:, 1],
        graph_map['locations'][:, 2], '.', c='b',
        label='Corrected Path')
ax.scatter(graph_map['tags'][:, 0], graph_map['tags'][:, 1],
           graph_map['tags'][:, 2], c = 'b', marker='^', s=100, label='Corrected Tags')

ax.legend()

# weightsF, weightsAx = plt.subplots()
# weightmap = weightsAx.imshow(weights[:, :12])
# plt.colorbar(weightmap)

plt.show()