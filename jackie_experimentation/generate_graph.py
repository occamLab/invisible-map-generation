import parse_map
import pprint as pp
import json
import numpy as np
from shapely.geometry import LineString

class Node:
    def __init__(self, x, y, z, poseID):
        # data is in the form of [x,y,z] of odometry vertices
        self.data = [x,y,z]
        self.poseID = poseID

class Graph:
    
    # create adjacency list
    g={}
    
    def addEdge(self,node,neighbour):
        # node -> neighbour  
        if node.poseID not in self.g:
            self.g[node.poseID]=[neighbour.poseID]
        else:
            self.g[node.poseID].append(neighbour.poseID)

        # neighbour -> node
        if neighbour.poseID not in self.g:
            self.g[neighbour.poseID] = [node.poseID]
        else:
            self.g[neighbour.poseID].append(node.poseID)

    # def address2readable(self):
    #     self.new_g = {}
    #     for key in self.g:
    #         self.new_g[key.poseID] = []
    #         for neighbour in self.g[key]:
    #             self.new_g[key.poseID].append(neighbour)

    def show_edges(self):
        for node in self.g:
            for neighbour in self.g[node]:
                print("(",node.data,", ",neighbour.data,")")

    def show_graph(self):
        pp.pprint(self.g)

def generate_mapping_graph(x, y, z, poseID):
    map_g = Graph()
    
    curr_node = Node(x[0], y[0], z[0], poseID[0])
    for i in range(1, len(x)):
        # Create node
        next_node = Node(x[i], y[i], z[i], poseID[i])
        map_g.addEdge(curr_node, next_node)
        curr_node = next_node
    # map_g.show_edges()
    # map_g.address2readable()
    # map_g.show_graph()
    return map_g.g

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def modify_json(input_filename, output_filename, g):

    with open(input_filename, "r") as read_file:
        mapping_data = json.load(read_file)
        odometry_data = mapping_data["odometry_vertices"]

    for pt in odometry_data:
        pt["neighbors"] = g[pt["poseId"]]

    with open(output_filename, 'w') as outfile:
        json.dump(mapping_data, outfile, cls=NpEncoder)


def is_intersection(pt1, pt2):
    '''
    Returns whether there is an intersection between 2 line segments using x-z coordinates
    '''
    line2 = LineString([(0,2), (2,0)])
    line1 = LineString([(0,0.5), (0.5,0)])

    if str(line1.intersection(line2)) != "LINESTRING EMPTY":
        print(line1.intersection(line2))


def print_intersection_pt():
    for 
    pass


if __name__ == "__main__":
    # filepath = 'data_jackie_AC_3rd_floor_processed.json'

    # filepath = 'test_blocks_jackie.json'
    # map = parse_map.Map_Data(filepath)
    # x, y, z, poseID = map.parse_odometry()
    
    # g = generate_mapping_graph(x, y, z, poseID)
    # modify_json(filepath, 'test_blocks_jackie_graph.json', g)

    # line1 = LineString([(0,0), (2,2)])
    line2 = LineString([(0,2), (2,0)])
    line1 = LineString([(0,0.5), (0.5,0)])

    if str(line1.intersection(line2)) != "LINESTRING EMPTY":
        print(line1.intersection(line2))


