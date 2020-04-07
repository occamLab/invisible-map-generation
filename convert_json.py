import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import graph


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


def matrix2measurement(pose):
    """ Convert a pose or array of poses in matrix form to [x, y, z,
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
    return np.concatenate([translation, rotation], axis=-1)


def as_graph(dct):
    """Convert a dictionary decoded from JSON into a graph.

    Args:
      dct (dict): The dictionary to convert to a graph.
    Returns:
      A graph derived from the input dictionary.
    """
    tag_data = np.array(dct['tag_data'])
    pose_data = np.array(dct['pose_data'])

    pose_matrices = pose_data[:, :16].reshape(-1, 4, 4).transpose(0, 2, 1)

    odom_vertex_estimates = matrix2measurement(pose_matrices)

    # The camera axis used to get tag measurements are flipped
    # relative to the phone frame used for odom measurements
    tag_to_odom_transform = np.eye(4)
    tag_to_odom_transform[[1,2], [1,2]] *= -1

    tag_edge_measurements_matrix = np.matmul(tag_to_odom_transform, tag_data[:, 1:17].reshape(-1, 4, 4))
    tag_edge_measurements = matrix2measurement(tag_edge_measurements_matrix)

    unique_tag_ids = np.unique(tag_data[:, 0]).astype('i')
    tag_vertex_id_by_tag_id = dict(
        zip(unique_tag_ids, range(unique_tag_ids.size)))

    # Enable lookup of tags by the frame they appear in
    tag_vertex_id_and_index_by_frame_id = {}

    for tag_index, (tag_id, tag_frame) in enumerate(tag_data[:, [0, 18]]):
        tag_vertex_id = tag_vertex_id_by_tag_id[tag_id]
        tag_vertex_id_and_index_by_frame_id[tag_frame] = tag_vertex_id_and_index_by_frame_id.get(
            tag_frame, [])
        tag_vertex_id_and_index_by_frame_id[tag_frame].append(
            (tag_vertex_id, tag_index))


    # Construct the dictionaries of vertices and edges
    vertices = {}
    edges = {}
    vertex_counter = unique_tag_ids.size
    edge_counter = 0

    previous_vertex = None
    previous_pose_matrix = None
    counted_tag_vertex_ids = set()
    first_odom_processed = False

    for i, odom_frame in enumerate(pose_data[:, 17]):
        current_odom_vertex_uid = vertex_counter
        vertices[current_odom_vertex_uid] = graph.Vertex(
            mode=graph.VertexType.ODOMETRY,
            estimate=odom_vertex_estimates[i],
            fixed=not first_odom_processed
        )
        first_odom_processed = True

        vertex_counter += 1

        # Connect odom to tag vertex
        for tag_vertex_id, tag_index in tag_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
            if tag_vertex_id not in counted_tag_vertex_ids:
                vertices[tag_vertex_id] = graph.Vertex(
                    mode=graph.VertexType.TAG,
                    estimate=matrix2measurement(pose_matrices[i].dot(tag_edge_measurements_matrix[tag_index])),
                    fixed=False
                )
                counted_tag_vertex_ids.add(tag_vertex_id)

            edges[edge_counter] = graph.Edge(
                startuid=current_odom_vertex_uid,
                enduid=tag_vertex_id,
                information=np.eye(6),
                measurement=tag_edge_measurements[tag_index]
                # measurement=np.array([0, 0, 0, 0, 0, 0, 1])
            )

            edge_counter += 1

        if previous_vertex:
            edges[edge_counter] = graph.Edge(
                startuid=previous_vertex,
                enduid=current_odom_vertex_uid,
                information=np.eye(6),
                measurement=matrix2measurement(np.linalg.inv(previous_pose_matrix).dot(pose_matrices[i]))
            )
            edge_counter += 1
        previous_vertex = current_odom_vertex_uid
        previous_pose_matrix = pose_matrices[i]

    resulting_graph = graph.Graph(vertices, edges)
    return resulting_graph
