"""Some helpful functions for visualizing and analyzing graphs.
"""
from typing import Union, List, Dict

import g2o
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from g2o import SE3Quat, EdgeProjectPSI2UV
from scipy.spatial.transform import Rotation as Rot

from graph_vertex_edge_classes import VertexType

# The camera axis used to get tag measurements are flipped  relative to the phone frame used for odom measurements
camera_to_odom_transform = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

# The ground truth tags for the 6-17-21 OCCAM Room
s = np.sin(np.pi / 4)
c = np.cos(np.pi / 4)
occam_room_tags = np.asarray([SE3Quat([0, 63.25 * 0.0254, 0, 0, 0, 0, 1]),
                              SE3Quat([269 * 0.0254, 48.5 * 0.0254, -31.25 * 0.0254, 0, 0, 0, 1]),
                              SE3Quat([350 * 0.0254, 58.25 * 0.0254, 86.25 * 0.0254, 0, c, 0, -s]),
                              SE3Quat([345.5 * 0.0254, 58 * 0.0254, 357.75 * 0.0254, 0, 1, 0, 0]),
                              SE3Quat([240 * 0.0254, 86 * 0.0254, 393 * 0.0254, 0, 1, 0, 0]),
                              SE3Quat([104 * 0.0254, 31.75 * 0.0254, 393 * 0.0254, 0, 1, 0, 0]),
                              SE3Quat([-76.75 * 0.0254, 56.5 * 0.0254, 316.75 * 0.0254, 0, c, 0, s]),
                              SE3Quat([-76.75 * 0.0254, 54 * 0.0254, 75 * 0.0254, 0, c, 0, s])])

def optimizer_to_map(vertices, optimizer: g2o.SparseOptimizer, is_sparse_bundle_adjustment=False) -> \
        Dict[str, Union[List, np.ndarray]]:
    """Convert a :class: g2o.SparseOptimizer to a dictionary containing locations of the phone, tags, and waypoints.

    Args:
        vertices: A dictionary of vertices. This is used to lookup the type of vertex pulled from the optimizer.
        optimizer: a :class: g2o.SparseOptimizer containing a map.
        is_sparse_bundle_adjustment: True if the optimizer is based on sparse bundle adjustment and False otherwise.

    Returns:
        A dictionary with fields 'locations', 'tags', and 'waypoints'. The 'locations' key covers a (n, 8) array
         containing x, y, z, qx, qy, qz, qw locations of the phone as well as the vertex uid at n points. The 'tags' and
        'waypoints' keys cover the locations of the tags and waypoints in the same format.
    """
    locations = np.reshape([], [0, 9])
    tagpoints = np.reshape([], [0, 3])
    tags = np.reshape([], [0, 8])
    waypoints = np.reshape([], [0, 8])
    waypoint_metadata = []
    exaggerate_tag_corners = True
    for i in optimizer.vertices():
        mode = vertices[i].mode
        if mode == VertexType.TAGPOINT:
            tag_vert = find_connected_tag_vert(optimizer, optimizer.vertex(i))

            if tag_vert is None:
                # TODO: double-check that the right way to handle this case is to continue
                continue

            location = optimizer.vertex(i).estimate()
            if exaggerate_tag_corners:
                location = location * np.array([10, 10, 1])

            tagpoints = np.vstack((tagpoints, tag_vert.estimate().inverse() * location))
        else:
            location = optimizer.vertex(i).estimate().translation()
            rotation = optimizer.vertex(i).estimate().rotation().coeffs()

            if mode == VertexType.ODOMETRY:
                pose = np.concatenate([location, rotation, [i], [vertices[i].meta_data['pose_id']]])
                locations = np.vstack([locations, pose])

            elif mode == VertexType.TAG:
                pose = np.concatenate([location, rotation, [i]])
                if is_sparse_bundle_adjustment:
                    # adjusts tag based on the position of the tag center
                    pose[:-1] = (SE3Quat([0, 0, 1, 0, 0, 0, 1]).inverse() * SE3Quat(vertices[i].estimate)).to_vector()
                if 'tag_id' in vertices[i].meta_data:
                    pose[-1] = vertices[i].meta_data['tag_id']
                tags = np.vstack([tags, pose])
            elif mode == VertexType.WAYPOINT:
                pose = np.concatenate([location, rotation, [i]])
                waypoints = np.vstack([waypoints, pose])
                waypoint_metadata.append(vertices[i].meta_data)

    # convert to array for sorting
    locations = np.array(locations)
    locations = locations[locations[:, -1].argsort()]
    return {'locations': locations, 'tags': np.array(tags), 'tagpoints': tagpoints,
            'waypoints': [waypoint_metadata, np.array(waypoints)]}


def optimizer_to_map_chi2(graph, optimizer: g2o.SparseOptimizer, is_sparse_bundle_adjustment=False) -> \
        Dict[str, Union[List, np.ndarray]]:
    """Convert a :class: g2o.SparseOptimizer to a dictionary containing locations of the phone, tags, waypoints, and
    per-odometry edge chi2 information.

    This function works by calling `optimizer_to_map` and adding a new entry that is a vector of the per-odometry edge
    chi2 information as calculated by the `map_odom_to_adj_chi2` method of the `Graph` class.

    Args:
        graph (Graph): A graph instance whose vertices attribute is passed as the first argument to `optimizer_to_map`
         and whose `map_odom_to_adj_chi2` method is used.
        optimizer: a :class: g2o.SparseOptimizer containing a map, which is passed as the second argument to
         `optimizer_to_map`.
        is_sparse_bundle_adjustment: True if the optimizer is based on sparse bundle adjustment and False otherwise;
         passed as the `is_sparse_bundle_adjustment` keyword argument to `optimizer_to_map`.

    Returns:
        A dictionary with fields 'locations', 'tags', 'waypoints', and 'locationsAdjChi2'. The 'locations' key covers a
        (n, 8) array  containing x, y, z, qx, qy, qz, qw locations of the phone as well as the vertex uid at n points.
        The 'tags' and 'waypoints' keys cover the locations of the tags and waypoints in the same format. Associated
        with each odometry node is a chi2 calculated from the `map_odom_to_adj_chi2` method of the `Graph` class, which
        is stored in the vector in the locationsAdjChi2 vector.
    """
    ret_map = optimizer_to_map(graph.vertices, optimizer, is_sparse_bundle_adjustment=is_sparse_bundle_adjustment)
    locations_shape = np.shape(ret_map["locations"])
    locations_adj_chi2 = np.zeros([locations_shape[0], 1])
    visible_tags_count = np.zeros([locations_shape[0], 1])

    for i, odom_node_vec in enumerate(ret_map["locations"]):
        uid = round(odom_node_vec[7])  # UID integer is stored as a floating point number, so cast it to an integer
        locations_adj_chi2[i], visible_tags_count[i] = graph.map_odom_to_adj_chi2(uid)

    ret_map["locationsAdjChi2"] = locations_adj_chi2
    ret_map["visibleTagsCount"] = visible_tags_count
    return ret_map


def find_connected_tag_vert(optimizer, location_vert):
    # TODO: it would be nice if we didn't have to scan the entire graph
    for edge in optimizer.edges():
        if type(edge) == EdgeProjectPSI2UV:
            if edge.vertex(0).id() == location_vert.id():
                return edge.vertex(2)
    return None


def measurement_to_matrix(measurement):
    transformation = np.eye(4)
    transformation[:3, 3] = measurement[:3]
    transformation[:3, :3] = Rot.from_quat(measurement[3:7]).as_matrix()
    return transformation


def pose_to_isometry(pose):
    """Convert a pose vector to a :class: g2o.Isometry3d instance.

    Args:
        pose: A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and qw respectively.
    Returns:
        A :class: g2o.Isometry3d instance encoding the same information as the input pose.
    """
    return g2o.Isometry3d(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])


def pose_to_se3quat(pose):
    """Convert a pose vector to a :class: g2o.Isometry3d instance.

    Args:
        pose: A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and qw respectively.
    Returns:
        A :class: g2o.Isometry3d instance encoding the same information as the input pose.
    """
    return g2o.SE3Quat(g2o.Quaternion(*np.roll(pose[3:7], 1)), pose[:3])


def isometry_to_pose(isometry):
    """Convert a :class: g2o.Isometry3d to a vector containing a pose.

    Args:
        isometry: A :class: g2o.Isometry3d instance.
    Returns:
        A 7 element 1-d numpy array encoding x, y, z, qx, qy, qz, and qw respectively.
    """
    return np.concatenate(
        [isometry.translation(), isometry.rotation().coeffs()])


def global_yaw_effect_basis(rotation, gravity_axis='z'):
    """Form a basis which describes the effect of a change in global yaw on a local measurement's qx, qy, and qz.

    Since the accelerometer measures gravitational acceleration, it can accurately measure the global z-azis but its
    measurement of the orthogonal axis are less reliable.

    Args:
        rotation: A :class: scipy.spatial.transform.Rotation encoding a local rotation.
        gravity_axis: A character specifying the gravity axis (e.g., 'z')

    Returns:
        A 3x3 numpy array where the columns are the new basis.
    """
    rotation1 = Rot.from_euler(gravity_axis, 0.05) * rotation
    change = rotation1.as_quat()[:3] - rotation.as_quat()[:3]
    return np.linalg.svd(change[:, np.newaxis])[0]


def locations_from_transforms(locations):
    for i in range(locations.shape[0]):
        locations[i, :7] = SE3Quat(locations[i, :7]).inverse().to_vector()
    return locations


def plot_metrics(sweep: np.ndarray, metrics: np.ndarray, log_scale: bool = False):
    to_plot = np.log(metrics) if log_scale else metrics
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(sweep, sweep.reshape(-1, 1), to_plot, cmap=cm.get_cmap('viridis'))
    ax.set_xlabel('Odometry Weights')
    ax.set_ylabel('April Tag Weights')
    ax.set_zlabel('Metric')
    fig.colorbar(surf)
    plt.show()


def weight_dict_from_array(array: Union[np.ndarray, List[int]]) -> Dict[str, np.ndarray]:
    weights = {'dummy': np.array([-1, 1e2, -1])}
    length = array.size if isinstance(array, np.ndarray) else len(array)
    if length == 2:
        weights['odometry'] = np.array([array[0]] * 6)
        weights['tag'] = np.array([array[1]] * 6)
        weights['tag_sba'] = np.array([array[1]] * 2)
    elif length == 8:
        weights['odometry'] = np.array(array[:6])
        weights['tag'] = np.array(array[6:] * 3)  # len of 8 should mean sba, so tag just needs to exist
        weights['tag_sba'] = np.array(array[6:])
    elif length == 12:
        weights['odometry'] = np.array(array[:6])
        weights['tag'] = np.array(array[6:])
        weights['tag_sba'] = np.array(array[6:8])
    elif length == 14:
        weights['odometry'] = np.array(array[:6])
        weights['tag'] = np.array(array[6:12])
        weights['tag_sba'] = np.array(array[12:])
    else:
        raise Exception('Given weights must be of length 2, 8, 12, or 14')
    return weights
