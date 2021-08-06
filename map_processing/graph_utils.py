"""Some helpful functions for visualizing and analyzing graphs.
"""
from enum import Enum
from typing import Union, List, Dict, Tuple, Any

import g2o
from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
from g2o import SE3Quat, EdgeProjectPSI2UV, Quaternion
from scipy.spatial.transform import Rotation as Rot
import shapely.geometry
from shapely.geometry import LineString

from map_processing.graph_vertex_edge_classes import VertexType

# The camera axis used to get tag measurements are flipped  relative to the phone frame used for odom measurements
camera_to_odom_transform = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

default_dummy_weights = np.array([-1, 1e2, -1])

assumed_focal_length = 1464

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


class MapInfo:
    """Container for identifying information for a graph (useful for caching process)

    Attributes:
        map_name (str): Specifies the child of the "maps" database reference to upload the optimized
         graph to; also passed as the map_name argument to the _cache_map method
        map_json (str): String corresponding to both the bucket blob name of the map and the path to cache the
         map relative to parent_folder
        map_dct (dict): String of json containing graph
    """

    def __init__(self, map_name: str, map_json: str, map_dct: Dict = None, uid: str = None):
        self.map_name: str = str(map_name)
        self.map_dct: Union[dict, str] = dict(map_dct) if map_dct is not None else {}
        self.map_json: str = str(map_json)
        self.uid = uid


def se3_quat_average(transforms):
    """TODO: documentation
    """
    translation_average = sum([t.translation() / len(transforms) for t in transforms])
    epsilons = np.ones(len(transforms), )
    converged = False
    quat_average = None
    while not converged:
        quat_sum = sum(np.array([t.orientation().x(), t.orientation().y(), t.orientation().z(), t.orientation().w()]) \
                       * epsilons[idx] for idx, t in enumerate(transforms))
        quat_average = quat_sum / np.linalg.norm(quat_sum)
        same_epsilon = [np.linalg.norm(epsilons[idx] * np.array([t.orientation().x(), t.orientation().y(),
                                                                 t.orientation().z(), t.orientation().w()]) - \
                                       quat_average) for idx, t in enumerate(transforms)]
        swap_epsilon = [np.linalg.norm(-epsilons[idx] * np.array([t.orientation().x(), t.orientation().y(),
                                                                  t.orientation().z(), t.orientation().w()]) - \
                                       quat_average) for idx, t in enumerate(transforms)]

        change_mask = np.greater(same_epsilon, swap_epsilon)
        epsilons[change_mask] = -epsilons[change_mask]
        converged = not np.any(change_mask)
    average_as_quat = Quaternion(quat_average[3], quat_average[0], quat_average[1], quat_average[2])
    return SE3Quat(average_as_quat, translation_average)


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


def plot_metrics(sweep: np.ndarray, metrics: np.ndarray, log_sweep: bool = False, log_metric: bool = False):
    filtered_metrics = metrics > -1
    sweep_plot = np.log(sweep) if log_sweep else sweep
    to_plot = np.log(metrics) if log_metric else metrics
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(sweep_plot, sweep_plot.transpose(), to_plot, cmap=cm.get_cmap('viridis'))
    ax.set_xlabel('Pose:Orientation')
    ax.set_ylabel('Odom:Tag')
    ax.set_zlabel('Metric')
    fig.colorbar(surf)
    plt.show()


def weight_dict_from_array(array: Union[np.ndarray, List[float]]) -> Dict[str, np.ndarray]:
    """
    Constructs a normalized weight dictionary from a given array of values
    """
    weights = {
        'dummy': np.array([-1, 1e2, -1]),
        'odometry': np.ones(6),
        'tag': np.ones(6),
        'tag_sba': np.ones(2),
        'odom_tag_ratio': 1
    }
    length = array.size if isinstance(array, np.ndarray) else len(array)
    half_len = length // 2
    has_ratio = length % 2 == 1
    if length == 1:  # ratio
        weights['odom_tag_ratio'] = array[0]
    elif length == 2: # tag/odom pose:rot/tag-sba x:y, ratio
        weights['odometry'] = np.array([array[0]] * 3 + [1] * 3)
        weights['tag'] = np.array([array[0]] * 3 + [1] * 3)
        weights['tag_sba'] = np.array([array[0], 1])
        weights['odom_tag_ratio'] = array[1]
    elif length == 3: # odom pose:rot, tag pose:rot/tag-sba x:y, ratio
        weights['odometry'] = np.array([array[0]] * 3 + [1] * 3)
        weights['tag'] = np.array([array[1]] * 3 + [1] * 3)
        weights['tag_sba'] = np.array([array[1], 1])
        weights['odom_tag_ratio'] = array[2]
    elif half_len == 2: # odom pose, odom rot, tag pose/tag-sba x, tag rot/tag-sba y, (ratio)
        weights['odometry'] = np.array([array[0]] * 3 + [array[1]] * 3)
        weights['tag'] = np.array([array[2]] * 3 + [array[3]] * 3)
        weights['tag_sba'] = np.array(array[2:])
        weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
    elif half_len == 3: # odom x y z qx qy, tag-sba x, (ratio)
        weights['odometry'] = np.array(array[:5])
        weights['tag_sba'] = np.array([array[5]])
        weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
    elif length == 4: # odom, tag-sba, (ratio)
        weights['odometry'] = np.array(array[:6])
        weights['tag_sba'] = np.array(array[6:])
        weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
    elif length == 5: # odom x y z qx qy, tag x y z qx qy, (ratio)
        weights['odometry'] = np.array(array[:5])
        weights['tag'] = np.array(array[5:])
        weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
    elif length == 6: # odom, tag, (ratio)
        weights['odometry'] = np.array(array[:6])
        weights['tag'] = np.array(array[6:])
        weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
    else:
        raise Exception(f'Weight length of {length} is not supported')
    return normalize_weights(weights)


def normalize_weights(weights: Dict[str, np.ndarray], is_sba: bool = False) -> Dict[str, np.ndarray]:
    """
    Normalizes the weights so that the resultant tag and odom weights in g2o will have a magnitude of 1, in ratio of
    weights['odom_tag_ratio'].

    If the provided array for each type is shorter than it should be, this will add elements to it until it is the
    right length. These elements will all be the same and chosen to get the correct overall magnitude for that weight
    set.

    Args:
        weights (dict): a dict mapping weight types to weight values, the set of weights to normalize.
        is_sba (bool): whether SBA is being used - if so, will scale tags so odom and tags are approx. the same units
    Returns:
        A new dict of weights where each value is normalized as said above, keeping dummy weights constant
    """
    odom_tag_ratio = max(0.00001, weights.get('odom_tag_ratio', 1))  # Put lower limit to prevent rounding causing division by 0
    odom_scale = 1 / (1 + 1 / odom_tag_ratio ** 2) ** 0.5
    tag_scale = 1 / (1 + odom_tag_ratio ** 2) ** 0.5
    if is_sba:
        tag_scale = tag_scale / assumed_focal_length
    normal_weights = {
        'dummy': weights['dummy'],
        'odom_tag_ratio': odom_tag_ratio
    }
    for weight_type in ('odometry', 'tag', 'tag_sba'):
        target_len = 2 if weight_type == 'tag_sba' else 6
        weight = weights.get(weight_type, np.ones(target_len))

        weight_mag = np.linalg.norm(np.exp(-weight))
        if weight.size < target_len and weight_mag >= 1:
            raise ValueError(f'Could not fill in weights of type {weight_type}, magnitude is already 1 or more ({weight_mag})')
        if weight.size > target_len:
            raise ValueError(f'{weight.size} weights for {weight_type} is too many - max is {target_len}')
        needed_weights = target_len - weight.size
        if needed_weights > 0:
            extra_weights = np.ones(needed_weights) * -0.5 * np.log((1 - weight_mag ** 2) / needed_weights)
            weight = np.hstack((weight, extra_weights))
            weight_mag = 1
        scale = odom_scale if weight_type == 'odometry' else tag_scale
        normal_weights[weight_type] = -(np.log(scale) - weight - np.log(weight_mag))
    return normal_weights


def weights_from_ratio(ratio: float) -> Dict[str, np.ndarray]:
    """
    Returns a weight dict with the given ratio between odom and tag weights
    """
    return weight_dict_from_array(np.array([ratio]))


class NeighborType(Enum):
    INTERSECTION = 0,
    CLOSE_DISTANCE = 1


def get_neighbors(vertices: np.ndarray, vertex_ids: Union[List[int], None] = None,
                  neighbor_type: NeighborType = NeighborType.INTERSECTION)\
        -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    nvertices = vertices.shape[0]
    if vertex_ids is None:
        vertex_ids = list(range(nvertices))
    neighbors = [[vertex_ids[1]]] + [[vertex_ids[i - 1], vertex_ids[i + 1]] for i in range(1, nvertices - 1)]\
                + [[vertex_ids[-2]]]
    curr_id = max(vertex_ids) + 1
    intersections = []

    for id1 in range(1, nvertices):
        for id2 in range(1, id1):
            if neighbor_type == NeighborType.INTERSECTION:
                intersection = _get_intersection(vertices, id1, id2, curr_id)
                if intersection is None:
                    continue
                intersections.append(intersection)
                neighbors[id1 - 1][-1] = curr_id
                neighbors[id1][0] = curr_id
                neighbors[id2 - 1][-1] = curr_id
                neighbors[id2][0] = curr_id
                curr_id += 1
            elif neighbor_type == NeighborType.CLOSE_DISTANCE and _is_close_enough(vertices, id1, id2):
                neighbors[id1].append(id2)
                neighbors[id2].append(id1)
                print(f'Point {id1} and {id2} are close enough, adding neighbors')

    return neighbors, intersections


def _is_close_enough(vertices, id1, id2):
    v1 = vertices[id1]
    v2 = vertices[id2]
    return abs(v1[1] - v2[1]) < 1 and ((v1[0] - v2[0]) ** 2 + (v1[2] - v2[2]) ** 2) ** 0.5 < 1


def _get_intersection(vertices, id1, id2, curr_id):
    line1_yval = (vertices[id1 - 1][1] + vertices[id1][1]) / 2
    line2_yval = (vertices[id2 - 1][1] + vertices[id2][1]) / 2
    if abs(line1_yval - line2_yval) > 1:
        return None

    line1 = LineString([(vertices[id1 - 1][0], vertices[id1 - 1][2]),
                        (vertices[id1][0], vertices[id1][2])])
    line2 = LineString([(vertices[id2 - 1][0], vertices[id2 - 1][2]),
                        (vertices[id2][0], vertices[id2][2])])

    intersect_pt = line1.intersection(line2)
    average = se3_quat_average([SE3Quat(vertices[id1 - 1]), SE3Quat(vertices[id1]),
                                SE3Quat(vertices[id2 - 1]), SE3Quat(vertices[id2])]).to_vector()
    if str(intersect_pt) == "LINESTRING EMPTY" or not isinstance(intersect_pt, shapely.geometry.point.Point):
        return None

    print(f'Intersection at {intersect_pt}, between {id1} and {id2}')
    return {
            'translation': {
                'x': intersect_pt.x,
                'y': average[1],
                'z': intersect_pt.y
            },
            'rotation': {
                'x': average[3],
                'y': average[4],
                'z': average[5],
                'w': average[6]
            },
            'poseId': curr_id,
            'neighbors': [id1 - 1, id1, id2 - 1, id2]
        }
