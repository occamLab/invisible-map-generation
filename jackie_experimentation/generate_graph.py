import plot_graph
import pprint as pp

class Node:
    def __init__(self, x, y, z):
        # data is in the form of [x,y,z] of odometry vertices
        self.data = [x,y,z]

class Graph:
    
    # create adjacency list
    g={}
    
    def addEdge(self,node,neighbour):
        # node -> neighbour  
        if node not in self.g:
            self.g[node]=[neighbour]
        else:
            self.g[node].append(neighbour)

        # neighbour -> node
        if neighbour not in self.g:
            self.g[neighbour] = [node]
        else:
            self.g[neighbour].append(node)
            
    def show_edges(self):
        for node in self.g:
            for neighbour in self.g[node]:
                print("(",node.data,", ",neighbour.data,")")

    def show_graph(self):
        pp.pprint(self.g)

def generate_mapping_graph(x, y, z):
    map_g = Graph()
    
    curr_node = Node(x[0], y[0], z[0])
    for i in range(1, len(x)):
        # Create node
        next_node = Node(x[i], y[i], z[i])
        map_g.addEdge(curr_node, next_node)
        curr_node = next_node

    # map_g.show_edges()
    map_g.show_graph()

if __name__ == "__main__":
    # x = [1, 2, 3, 4, 5, 6]
    # y = [1, 2, 3, 4, 5, 6]
    # z = [1, 2, 3, 4, 5, 6]

    x, y, z = plot_graph.parse_odometry()
    
    generate_mapping_graph(x, y, z)
