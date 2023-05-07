"""
Generate graph that connects intersecting nodes from odometry vertices. 
"""
import parse_map
import pprint as pp
import json
import numpy as np
from shapely.geometry import LineString


class Node:
    def __init__(self, x, y, z, poseID):
        # data is in the form of the [x,y,z] translation of odometry vertices
        self.data = [x, y, z]
        self.poseID = poseID


class Graph:
    def __init__(self):
        # create adjacency list
        self.g = {}
        self.edges_set = set()
        self.edges = []

    def addEdge(self, node, neighbour):
        # node -> neighbour
        if node.poseID not in self.g:
            self.g[node.poseID] = [neighbour.poseID]
        else:
            self.g[node.poseID].append(neighbour.poseID)

        # neighbour -> node
        if neighbour.poseID not in self.g:
            self.g[neighbour.poseID] = [node.poseID]
        else:
            self.g[neighbour.poseID].append(node.poseID)

        # add edge to set and list of edges
        if ((node, neighbour) not in self.edges_set) and (
            (neighbour, node) not in self.edges_set
        ):
            self.edges_set.add((node, neighbour))
            self.edges.append((node, neighbour))

    def show_edges(self):
        for node in self.g:
            for neighbour in self.g[node]:
                print("(", node.data, ", ", neighbour.data, ")")

    def show_graph(self):
        pp.pprint(self.g)

    def no_overlapping_endpoints(self, e1, e2):
        """Makes sure that the two edges are not intersecting"""
        endpoint_set = set(
            [
                e1[0].data[0],
                e1[0].data[2],
                e1[1].data[0],
                e1[1].data[2],
                e2[0].data[0],
                e2[0].data[2],
                e2[1].data[0],
                e2[1].data[2],
            ]
        )
        return len(endpoint_set) == 8

    def get_intersections(self):
        """
        Returns whether there is an intersection between 2 line segments using x-z coordinates
        """
        intersect_nodes = []  # Node objects
        self.intersect_vertices = []  # Same style as json objects
        _id = 0

        for e1 in self.edges:
            for e2 in self.edges:
                # check that there are no overlapping endpoints for intersection
                if e1 != e2 and self.no_overlapping_endpoints(e1, e2):
                    line1 = LineString(
                        [(e1[0].data[0], e1[0].data[2]), (e1[1].data[0], e1[1].data[2])]
                    )
                    line2 = LineString(
                        [(e2[0].data[0], e2[0].data[2]), (e2[1].data[0], e2[1].data[2])]
                    )

                    # find intersection
                    if (
                        str(line1.intersection(line2)) != "LINESTRING EMPTY"
                        and str(type(line1.intersection(line2)))
                        == "<class 'shapely.geometry.point.Point'>"
                    ):
                        # TODO: implement check for y position to make sure not different floors
                        intersect_pt = line1.intersection(line2)
                        print("INTERSECTION: ", intersect_pt)

                        # Create new node for intersection and add to list
                        intersect_node = Node(intersect_pt.x, 0, intersect_pt.y, _id)

                        # Create json formatted vertex
                        vertex = {}
                        vertex["translation"] = {}
                        vertex["translation"]["x"] = intersect_pt.x
                        vertex["translation"]["y"] = 0
                        vertex["translation"]["z"] = intersect_pt.y
                        vertex["id"] = _id

                        intersect_nodes.append(intersect_node)
                        self.intersect_vertices.append(vertex)
                        _id += 1


def generate_mapping_graph(x, y, z, poseID):
    map_g = Graph()

    curr_node = Node(x[0], y[0], z[0], poseID[0])
    for i in range(1, len(x)):
        # Create node
        next_node = Node(x[i], y[i], z[i], poseID[i])
        map_g.addEdge(curr_node, next_node)
        curr_node = next_node
    map_g.get_intersections()
    return map_g


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


def modify_json(input_filename, output_filename, graph):
    with open(input_filename, "r") as read_file:
        mapping_data = json.load(read_file)
        odometry_data = mapping_data["odometry_vertices"]

    # add neighbors to odometry vertices
    for pt in odometry_data:
        pt["neighbors"] = graph.g[pt["poseId"]]

    # add intersection vertices
    mapping_data["intersection_vertices"] = graph.intersect_vertices

    with open(output_filename, "w") as outfile:
        json.dump(mapping_data, outfile, cls=NpEncoder)


if __name__ == "__main__":
    # filepath = 'data_jackie_AC_3rd_floor_processed.json'

    filepath = "test_blocks_jackie.json"
    map = parse_map.Map_Data(filepath)
    x, y, z, poseID = map.parse_odometry()

    graph = generate_mapping_graph(x, y, z, poseID)
    modify_json(filepath, "test_blocks_jackie_graph_w_intersections.json", graph)
