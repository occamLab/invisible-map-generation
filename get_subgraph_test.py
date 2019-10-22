from graph_utils import optimizer_to_map
import pickle

with open('converted-data/academic_center.pkl', 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
optimizer = graph.unoptimized_graph

# map = optimizer_to_map(graph.vertices, graph.unoptimized_graph)
