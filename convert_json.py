import json
import numpy as np
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt

import graph


def pose2diffs(poses):
    diffs = []
    for previous_pose, current_pose in zip(poses[:-1], poses[1:]):
        diffs.append(np.linalg.inv(previous_pose).dot(current_pose))
    diffs = np.array(diffs)
    return diffs


def matrix2measurement(pose):
    translation = pose[..., :3, 3]
    rotation = R.from_matrix(pose[..., :3, :3]).as_quat()
    return np.concatenate([translation, rotation], axis=-1)


def as_graph(dct):
    tag_data = np.array(dct['tag_data'])
    pose_data = np.array(dct['pose_data'])
    map_id = dct['map_id']
    camera_intrinsics = dct['camera_intrinsics']

    pose_matrices = pose_data[:, :16].reshape(-1, 4, 4).transpose(0, 2, 1)

    odom_edge_measurements = matrix2measurement(pose2diffs(pose_matrices))
    odom_vertex_estimates = matrix2measurement(pose_matrices)

    tag_edge_measurements = matrix2measurement(
        tag_data[:, 1:17].reshape(-1, 4, 4))

    tags_by_frame = {}
    for i, tag_frame in enumerate(tag_data[:, 18]):
        tags_by_frame[tag_frame] = tags_by_frame.get(tag_frame, [])
        tags_by_frame[tag_frame].append(i)

    vertices = {}
    edges = {}
    vertex_counter = 0
    edge_counter = 0
    for i, odom_frame in enumerate(pose_data[:, 17]):
        odom_vertex = vertex_counter
        vertices[odom_vertex] = graph.Vertex(
            mode=graph.VertexType.ODOMETRY,
            estimate=odom_vertex_estimates[i],
            fixed=False
        )
        vertex_counter += 1
        for tag_idx in tags_by_frame[odom_frame]:
            vertices[vertex_counter] = graph.Vertex(
                mode=graph.VertexType.TAG,
                estimate=odom_vertex_estimates[i].dot(
                    tag_edge_measurements[tag_idx]),
                fixed=False
            )

            edges[edge_counter] = graph.Edge(
                startuid=odom_vertex,
                enduid=vertex_counter,
                information=np.eye(6),
                measurement=tag_edge_measurements[tag_idx]
            )
            vertex_counter += 1


with open('round1.json', 'r') as f:
    x = json.load(f)

y = as_graph(x)
