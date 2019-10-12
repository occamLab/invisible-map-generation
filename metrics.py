import pickle
import itertools
import numpy as np
import g2o
from graph_utils import optimizer_to_map, connected_components, ordered_odometry_edges
from graph import VertexType
import matplotlib.pyplot as plt
from convert_posegraph import convert
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R

with open('converted-data/academic_center.pkl', 'rb') as data:
    graph = pickle.load(data)

graph.generate_unoptimized_graph()
unoptimized_map = optimizer_to_map(graph.vertices, graph.unoptimized_graph)

graph.optimize_graph()
optimized_map = optimizer_to_map(graph.vertices, graph.optimized_graph)

def compute_transformation(start, end):
    """
    compute the transformation from start position to end
    :param start: start position as a pose np array of [x, y, z, qx, qy, qz, 1]
    :param end: end position as a pose np array of [x, y, z, qx, qy, qz, 1]
    :return: numpy array of
    """
    translation = list(end[:3] - start[:3])
    start_rotation = R.from_quat(start[3:])
    end_rotation = R.from_quat(end[3:])
    rotation = end_rotation.as_dcm() * np.linalg.inv(start_rotation.as_dcm())
    rotation = list(R.from_dcm(rotation).as_quat())
    return np.array(translation + rotation)

def generate_transformations(positions):
    '''This may be difficult, as querying a transform a-> b in a
    dictionary with the transform b->a must return the inverse of
    b->a. It may be useful to make a separate clone of optimize_to_map
    which makes a hashmap of vertex uids.'''
    transformations = dict()
    pairs = list(itertools.combinations(positions, 2))
    # transformations
    for pair in pairs:
        key = (pair[0][7], pair[1][7])
        transformations[key] = compute_transformation(pair[0][:7], pair[1][:7])
    return transformations

def compute_tags_transformations_diff_between_maps(optimized_tags_transformations,unoptimized_tags_transformations, ):
    transformation_differences = dict()
    for key in optimized_tags_transformations.keys():
        if key in unoptimized_tags_transformations.keys():
            transformation_differences[key] = compute_transformation(optimized_tags_transformations[key],unoptimized_tags_transformations[key])
        else:
            unoptimized_tags_transformations = unoptimized_tags_transformations[(key[1],key[0])]
            inversed_unoptimized_tags_transformations = compute_transformation(unoptimized_tags_transformations, np.array([0,0,0,0,0,0,1]))
            transformation_differences[key] = compute_transformation(optimized_tags_transformations[key],
                                                                     inversed_unoptimized_tags_transformations)
    return transformation_differences

def plot_tags_distance_diff_in_maps(optimized_map, unoptimized_map):
    unoptimized_tags_transformations = generate_transformations(unoptimized_map['tags'])
    print("tag poses:", optimized_map['tags'])
    print("number of tags:", len(optimized_map['tags']))
    optimized_tags_transformations = generate_transformations(optimized_map['tags'])
    transformation_differences = compute_tags_transformations_diff_between_maps(optimized_tags_transformations, unoptimized_tags_transformations)
    distances = []
    error = []
    for key in optimized_tags_transformations.keys():
        distances.append(np.linalg.norm(optimized_tags_transformations[key][:3]))
        error.append(np.linalg.norm(transformation_differences[key][:3]))
    area = np.pi*3
    plt.scatter(distances, error, s=area)
    plt.xlabel('distance magnitude')
    plt.ylabel('error')
    plt.show()


plot_tags_distance_diff_in_maps(optimized_map, unoptimized_map)


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(unoptimized_map['locations'][:, 0], unoptimized_map['locations'][:, 1], unoptimized_map['locations'][:, 2])
# plt.show()
