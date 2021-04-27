import itertools
import numpy as np
from as_graph import matrix2measurement
import graph


def as_graph(dct):
    """Convert a dictionary decoded from JSON into a graph.

    Args:
      dct (dict): The dictionary to convert to a graph.
    Returns:
      A graph derived from the input dictionary.
    """
    pose_data = np.array(dct['pose_data'])
    if not pose_data.size:
        pose_data = np.zeros((0, 18))
    pose_matrices = pose_data[:, :16].reshape(-1, 4, 4).transpose(0, 2, 1)
    odom_vertex_estimates = matrix2measurement(pose_matrices)

    # The camera axis used to get tag measurements are flipped
    # relative to the phone frame used for odom measurements
    camera_to_odom_transform = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    # flatten the data into individual numpy arrays that we can operate on
    if 'tag_data' in dct:
        tag_pose_flat = np.vstack([[x['tagPose'] for x in tagsFromFrame] for tagsFromFrame in dct['tag_data']])
        tag_ids = np.vstack(list(itertools.chain(*[[x['tagId'] for x in tagsFromFrame] for tagsFromFrame in \
                                                   dct['tag_data']])))
        pose_ids = np.vstack(list(itertools.chain(*[[x['poseId'] for x in tagsFromFrame] for tagsFromFrame in \
                                                    dct['tag_data']])))
        tag_joint_covar = np.vstack([[x['jointCovar'] for x in tagsFromFrame] for tagsFromFrame in
                                     dct['tag_data']])

        tag_position_variances = np.vstack([[x['tagPositionVariance'] for x in tagsFromFrame] for tagsFromFrame in \
                                            dct['tag_data']])
        tag_orientation_variances = np.vstack([[x['tagOrientationVariance'] for x in tagsFromFrame] for tagsFromFrame \
                                               in dct['tag_data']])
    else:
        tag_pose_flat = np.zeros((0, 16))
        tag_ids = np.zeros((0, 1), type=np.int)
        pose_ids = np.zeros((0, 1), type=np.int)
        tag_joint_covar = np.zeros((0, 49), type=np.double)

        tag_position_variances = np.zeros((0, 3), type=np.double)
        tag_orientation_variances = np.zeros((0, 4), type=np.double)

    tag_edge_measurements_matrix = np.matmul(camera_to_odom_transform, tag_pose_flat.reshape(-1, 4, 4))
    tag_edge_measurements = matrix2measurement(tag_edge_measurements_matrix)

    # Note that we are ignoring the variance deviation of qw since we use a compact quaternion parameterization of
    # orientation
    tag_joint_covar_matrices = tag_joint_covar.reshape((-1, 7, 7))
    # TODO: for some reason we have missing measurements (all zeros).  Throw those out
    tag_edge_prescaling = np.array([np.linalg.inv(covar[:-1, :-1]) if np.linalg.det(covar[:-1, :-1]) != 0 else \
                                        np.zeros((6, 6)) for covar in tag_joint_covar_matrices])

    # print("overwriting with diagonal covariances")
    # tag_edge_prescaling = 1./np.hstack((tag_position_variances, tag_orientation_variances[:,:-1]))
    print('resetting prescaling to identity')
    tag_edge_prescaling = np.ones(tag_edge_prescaling.shape)

    unique_tag_ids = np.unique(tag_ids)
    tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(unique_tag_ids.size)))
    tag_id_by_tag_vertex_id = dict(zip(tag_vertex_id_by_tag_id.values(), tag_vertex_id_by_tag_id.keys()))

    # Enable lookup of tags by the frame they appear in
    tag_vertex_id_and_index_by_frame_id = {}

    for tag_index, (tag_id, tag_frame) in enumerate(np.hstack((tag_ids, pose_ids))):
        tag_vertex_id = tag_vertex_id_by_tag_id[tag_id]
        tag_vertex_id_and_index_by_frame_id[tag_frame] = tag_vertex_id_and_index_by_frame_id.get(
            tag_frame, [])
        tag_vertex_id_and_index_by_frame_id[tag_frame].append(
            (tag_vertex_id, tag_index))

    waypoint_list_uniform = list(map(lambda x: np.asarray(x[:-1]).reshape((-1, 18)), dct.get('location_data', [])))
    waypoint_names = list(map(lambda x: x[-1], dct.get('location_data', [])))
    unique_waypoint_names = np.unique(waypoint_names)
    if waypoint_list_uniform:
        waypoint_data_uniform = np.concatenate(waypoint_list_uniform)
    else:
        waypoint_data_uniform = np.zeros((0, 18))
    waypoint_edge_measurements_matrix = waypoint_data_uniform[:, :16].reshape(-1, 4, 4)
    waypoint_edge_measurements = matrix2measurement(waypoint_edge_measurements_matrix)

    waypoint_vertex_id_by_name = dict(
        zip(unique_waypoint_names, range(unique_tag_ids.size, unique_tag_ids.size + unique_waypoint_names.size)))
    waypoint_name_by_vertex_id = dict(zip(waypoint_vertex_id_by_name.values(), waypoint_vertex_id_by_name.keys()))
    # Enable lookup of waypoints by the frame they appear in
    waypoint_vertex_id_and_index_by_frame_id = {}

    for waypoint_index, (waypoint_name, waypoint_frame) in enumerate(zip(waypoint_names, waypoint_data_uniform[:, 17])):
        waypoint_vertex_id = waypoint_vertex_id_by_name[waypoint_name]
        waypoint_vertex_id_and_index_by_frame_id[waypoint_frame] = waypoint_vertex_id_and_index_by_frame_id.get(
            waypoint_name, [])
        waypoint_vertex_id_and_index_by_frame_id[waypoint_frame].append(
            (waypoint_vertex_id, waypoint_index))

    # Construct the dictionaries of vertices and edges
    vertices = {}
    edges = {}
    vertex_counter = unique_tag_ids.size + unique_waypoint_names.size
    edge_counter = 0

    previous_vertex = None
    previous_pose_matrix = None
    counted_tag_vertex_ids = set()
    counted_waypoint_vertex_ids = set()
    first_odom_processed = False
    num_tag_edges = 0

    for i, odom_frame in enumerate(pose_data[:, 17]):
        current_odom_vertex_uid = vertex_counter
        vertices[current_odom_vertex_uid] = graph.Vertex(
            mode=graph.VertexType.ODOMETRY,
            estimate=odom_vertex_estimates[i],
            fixed=not first_odom_processed,
            meta_data={'poseId': odom_frame}
        )
        first_odom_processed = True
        vertex_counter += 1

        # Connect odom to tag vertex
        for tag_vertex_id, tag_index in tag_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
            if tag_vertex_id not in counted_tag_vertex_ids:
                vertices[tag_vertex_id] = graph.Vertex(
                    mode=graph.VertexType.TAG,
                    estimate=matrix2measurement(pose_matrices[i].dot(
                        tag_edge_measurements_matrix[tag_index])),
                    fixed=False
                )
                vertices[tag_vertex_id].meta_data['tag_id'] = tag_id_by_tag_vertex_id[tag_vertex_id]
                counted_tag_vertex_ids.add(tag_vertex_id)
            edges[edge_counter] = graph.Edge(
                startuid=current_odom_vertex_uid,
                enduid=tag_vertex_id,
                information=np.eye(6),
                information_prescaling=tag_edge_prescaling[tag_index],
                measurement=tag_edge_measurements[tag_index],
                corner_ids=None,
                camera_intrinsics=None
            )
            num_tag_edges += 1

            edge_counter += 1

        # Connect odom to waypoint vertex
        for waypoint_vertex_id, waypoint_index in waypoint_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
            if waypoint_vertex_id not in counted_waypoint_vertex_ids:
                vertices[waypoint_vertex_id] = graph.Vertex(
                    mode=graph.VertexType.WAYPOINT,
                    estimate=matrix2measurement(pose_matrices[i].dot(
                        waypoint_edge_measurements_matrix[waypoint_index])),
                    fixed=False
                )
                vertices[waypoint_vertex_id].meta_data['name'] = waypoint_name_by_vertex_id[waypoint_vertex_id]
                counted_waypoint_vertex_ids.add(waypoint_vertex_id)

            edges[edge_counter] = graph.Edge(
                startuid=current_odom_vertex_uid,
                enduid=waypoint_vertex_id,
                corner_ids=None,
                information=np.eye(6),
                information_prescaling=None,
                camera_intrinsics=None,
                measurement=waypoint_edge_measurements[waypoint_index]
            )

            edge_counter += 1

        if previous_vertex:
            # TODO: might want to consider prescaling based on the magnitude of the change
            edges[edge_counter] = graph.Edge(
                startuid=previous_vertex,
                enduid=current_odom_vertex_uid,
                information=np.eye(6),
                information_prescaling=None,
                measurement=matrix2measurement(np.linalg.inv(
                    previous_pose_matrix).dot(pose_matrices[i])),
                corner_ids=None,
                camera_intrinsics=None
            )
            edge_counter += 1

        # make dummy node
        dummy_node_uid = vertex_counter
        vertices[dummy_node_uid] = graph.Vertex(
            mode=graph.VertexType.DUMMY,
            estimate=np.hstack((np.zeros(3, ), odom_vertex_estimates[i][3:])),
            fixed=True
        )
        vertex_counter += 1

        # connect odometry to dummy node
        edges[edge_counter] = graph.Edge(
            startuid=current_odom_vertex_uid,
            enduid=dummy_node_uid,
            information=np.eye(6),
            information_prescaling=None,
            measurement=np.array([0, 0, 0, 0, 0, 0, 1]),
            corner_ids=None,
            camera_intrinsics=None
        )
        edge_counter += 1

        previous_vertex = current_odom_vertex_uid
        previous_pose_matrix = pose_matrices[i]

    resulting_graph = graph.Graph(vertices, edges, gravity_axis='y', damping_status=True)
    return resulting_graph
