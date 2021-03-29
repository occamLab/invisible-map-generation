"""Some helpful functions for visualizing and analyzing graphs.
"""
import numpy as np
from graph_vertex_edge_classes import VertexType
from scipy.spatial.transform import Rotation as R
from g2o import SE3Quat, EdgeProjectPSI2UV
import g2o


def optimizer_to_map(vertices, optimizer: g2o.SparseOptimizer, is_sparse_bundle_adjustment=False):
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
    locationsAdjChi2 = np.reshape([], [0, 1])
    tagpoints = np.reshape([], [0, 3])
    tags = np.reshape([], [0, 8])
    waypoints = np.reshape([], [0, 8])
    waypoint_metadata = []
    exaggerate_tag_corners = True
    for i in optimizer.vertices():
        mode = vertices[i].mode
        if mode == VertexType.TAGPOINT:
            location = optimizer.vertex(i).estimate()
            if exaggerate_tag_corners:
                location = location * np.array([10, 10, 1])
            tag_vert = find_connected_tag_vert(optimizer, optimizer.vertex(i))
            tagpoints = np.vstack((tagpoints, tag_vert.estimate().inverse() * location))
        else:
            location = optimizer.vertex(i).estimate().translation()
            rotation = optimizer.vertex(i).estimate().rotation().coeffs()

            if mode == VertexType.ODOMETRY:
                pose = np.concatenate([location, rotation, [i], [vertices[i].meta_data['poseId']]])
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


def optimizer_to_map_chi2(graph, optimizer: g2o.SparseOptimizer, is_sparse_bundle_adjustment=False):
    """Convert a :class: g2o.SparseOptimizer to a dictionary containing locations of the phone, tags, and waypoints.

    Args:

        optimizer: a :class: g2o.SparseOptimizer containing a map.
        is_sparse_bundle_adjustment: True if the optimizer is based on sparse bundle adjustment and False otherwise.

    Returns:
        A dictionary with fields 'locations', 'tags', and 'waypoints'. The 'locations' key covers a (n, 8) array
         containing x, y, z, qx, qy, qz, qw locations of the phone as well as the vertex uid at n points. The 'tags' and
        'waypoints' keys cover the locations of the tags and waypoints in the same format.
    """
    locations = np.reshape([], [0, 9])
    locationsAdjChi2 = np.reshape([], [0, 1])
    tagpoints = np.reshape([], [0, 3])
    tags = np.reshape([], [0, 8])
    waypoints = np.reshape([], [0, 8])
    waypoint_metadata = []
    exaggerate_tag_corners = True
    vertices = graph.vertices
    uids = []

    for i in optimizer.vertices():
        mode = vertices[i].mode
        if mode == VertexType.TAGPOINT:
            location = optimizer.vertex(i).estimate()
            if exaggerate_tag_corners:
                location = location * np.array([10, 10, 1])
            tag_vert = find_connected_tag_vert(optimizer, optimizer.vertex(i))
            tagpoints = np.vstack((tagpoints, tag_vert.estimate().inverse() * location))
        else:
            location = optimizer.vertex(i).estimate().translation()
            rotation = optimizer.vertex(i).estimate().rotation().coeffs()

            if mode == VertexType.ODOMETRY:
                pose = np.concatenate([location, rotation, [i], [vertices[i].meta_data['poseId']]])
                locations = np.vstack([locations, pose])
                adjChi2Error = graph.map_odom_to_adj_chi2(i)
                locationsAdjChi2 = np.vstack([locationsAdjChi2, adjChi2Error])
                uids.append(i)

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
            'waypoints': [waypoint_metadata, np.array(waypoints)], 'locationsAdjChi2': locationsAdjChi2, 'uids': uids}


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
    transformation[:3, :3] = R.from_quat(measurement[3:7]).as_matrix()
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
    rotation1 = R.from_euler(gravity_axis, 0.05) * rotation
    change = rotation1.as_quat()[:3] - rotation.as_quat()[:3]
    return np.linalg.svd(change[:, np.newaxis])[0]


def locations_from_transforms(locations):
    """TODO: documentation

    """
    for i in range(locations.shape[0]):
        locations[i, :7] = SE3Quat(locations[i, :7]).inverse().to_vector()
    return locations


def pose2diffs(poses):
    """Convert an array of poses in the odom frame to an array of
    transformations from the last pose.

    Args:
      poses (np.ndarray): Pose or array of poses.
    Returns:
      An array of transformations
    """
    diffs = []
    for previous_pose, current_pose in zip(poses[:-1], poses[1:]):
        diffs.append(np.linalg.inv(previous_pose).dot(current_pose))
    diffs = np.array(diffs)
    return diffs


def matrix2measurement(pose, invert=False):
    """Convert a pose or array of poses in matrix form to [x, y, z,
    qx, qy, qz, qw].

    The output will have one fewer dimension than the input.

    Args:
      pose (np.ndarray): Pose or array of poses in matrix form.
        The poses are converted along the last two axes.
    Returns:
      Converted pose or array of poses.
    """
    translation = pose[..., :3, 3]
    rotation = R.from_matrix(pose[..., :3, :3]).as_quat()
    ret_val = np.concatenate([translation, rotation], axis=-1)
    if invert:
        ret_val = np.vstack(list(map(lambda measurement: SE3Quat(measurement).inverse().to_vector(), ret_val)))
    return ret_val
