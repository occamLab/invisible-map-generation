import itertools
from collections import defaultdict
import numpy as np
from g2o import SE3Quat, CameraParameters
from map_processing.as_graph import matrix2measurement, se3_quat_average
from map_processing import graph


def as_graph(dct, fix_tag_vertices=False):
    """Convert a dictionary decoded from JSON into a graph.

    Args:
      dct (dict): The dictionary to convert to a graph.
    Returns:
      A graph derived from the input dictionary.
    """
    pose_data = np.array(dct["pose_data"])
    if not pose_data.size:
        pose_data = np.zeros((0, 18))
    pose_matrices = pose_data[:, :16].reshape(-1, 4, 4).transpose(0, 2, 1)
    odom_vertex_estimates = matrix2measurement(pose_matrices, invert=True)
    tag_size = 0.173  # TODO: need to send this with the tag detection
    true_3d_points = np.array(
        [
            [-tag_size / 2, -tag_size / 2, 1],
            [tag_size / 2, -tag_size / 2, 1],
            [tag_size / 2, tag_size / 2, 1],
            [-tag_size / 2, tag_size / 2, 1],
        ]
    )
    true_3d_tag_center = np.array([0, 0, 1])
    # The camera axis used to get tag measurements are flipped
    # relative to the phone frame used for odom measurements
    camera_to_odom_transform = np.array(
        [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    )
    # flatten the data into individual numpy arrays that we can operate on
    if "tag_data" in dct and len(dct["tag_data"]) > 0:
        tag_pose_flat = np.vstack(
            [[x["tagPose"] for x in tagsFromFrame] for tagsFromFrame in dct["tag_data"]]
        )
        camera_intrinsics_for_tag = np.vstack(
            [
                [x["cameraIntrinsics"] for x in tagsFromFrame]
                for tagsFromFrame in dct["tag_data"]
            ]
        )
        tag_corners = np.vstack(
            [
                [x["tagCornersPixelCoordinates"] for x in tagsFromFrame]
                for tagsFromFrame in dct["tag_data"]
            ]
        )
        tag_ids = np.vstack(
            list(
                itertools.chain(
                    *[
                        [x["tagId"] for x in tagsFromFrame]
                        for tagsFromFrame in dct["tag_data"]
                    ]
                )
            )
        )
        pose_ids = np.vstack(
            list(
                itertools.chain(
                    *[
                        [x["poseId"] for x in tagsFromFrame]
                        for tagsFromFrame in dct["tag_data"]
                    ]
                )
            )
        )
    else:
        tag_pose_flat = np.zeros((0, 16))
        camera_intrinsics_for_tag = np.zeros((0, 4))
        tag_corners = np.zeros((0, 8))
        tag_ids = np.zeros((0, 1), dtype=np.int)
        pose_ids = np.zeros((0, 1), dtype=np.int)

    tag_edge_measurements_matrix = np.matmul(
        camera_to_odom_transform, tag_pose_flat.reshape(-1, 4, 4)
    )
    tag_edge_measurements = matrix2measurement(tag_edge_measurements_matrix)

    unique_tag_ids = np.unique(tag_ids)
    tag_vertex_id_by_tag_id = dict(
        zip(unique_tag_ids, range(0, unique_tag_ids.size * 5, 5))
    )
    tag_id_by_tag_vertex_id = dict(
        zip(tag_vertex_id_by_tag_id.values(), tag_vertex_id_by_tag_id.keys())
    )
    tag_corner_ids_by_tag_vertex_id = dict(
        zip(
            tag_id_by_tag_vertex_id.keys(),
            map(
                lambda tag_vertex_id: list(range(tag_vertex_id + 1, tag_vertex_id + 5)),
                tag_id_by_tag_vertex_id.keys(),
            ),
        )
    )

    # Enable lookup of tags by the frame they appear in
    tag_vertex_id_and_index_by_frame_id = {}

    for tag_index, (tag_id, tag_frame) in enumerate(np.hstack((tag_ids, pose_ids))):
        tag_vertex_id = tag_vertex_id_by_tag_id[tag_id]
        tag_vertex_id_and_index_by_frame_id[
            tag_frame
        ] = tag_vertex_id_and_index_by_frame_id.get(tag_frame, [])
        tag_vertex_id_and_index_by_frame_id[tag_frame].append(
            (tag_vertex_id, tag_index)
        )

    waypoint_list_uniform = list(
        map(
            lambda x: np.asarray(x[:-1]).reshape((-1, 18)), dct.get("location_data", [])
        )
    )
    waypoint_names = list(map(lambda x: x[-1], dct.get("location_data", [])))
    unique_waypoint_names = np.unique(waypoint_names)
    if waypoint_list_uniform:
        waypoint_data_uniform = np.concatenate(waypoint_list_uniform)
    else:
        waypoint_data_uniform = np.zeros((0, 18))
    waypoint_edge_measurements_matrix = waypoint_data_uniform[:, :16].reshape(-1, 4, 4)
    waypoint_edge_measurements = matrix2measurement(waypoint_edge_measurements_matrix)

    waypoint_vertex_id_by_name = dict(
        zip(
            unique_waypoint_names,
            range(
                unique_tag_ids.size * 5,
                unique_tag_ids.size * 5 + unique_waypoint_names.size,
            ),
        )
    )
    waypoint_name_by_vertex_id = dict(
        zip(waypoint_vertex_id_by_name.values(), waypoint_vertex_id_by_name.keys())
    )
    # Enable lookup of waypoints by the frame they appear in
    waypoint_vertex_id_and_index_by_frame_id = {}

    for waypoint_index, (waypoint_name, waypoint_frame) in enumerate(
        zip(waypoint_names, waypoint_data_uniform[:, 17])
    ):
        waypoint_vertex_id = waypoint_vertex_id_by_name[waypoint_name]
        waypoint_vertex_id_and_index_by_frame_id[
            waypoint_frame
        ] = waypoint_vertex_id_and_index_by_frame_id.get(waypoint_name, [])
        waypoint_vertex_id_and_index_by_frame_id[waypoint_frame].append(
            (waypoint_vertex_id, waypoint_index)
        )

    # Construct the dictionaries of vertices and edges
    vertices = {}
    edges = {}
    vertex_counter = unique_tag_ids.size * 5 + unique_waypoint_names.size
    edge_counter = 0

    previous_vertex = None
    counted_tag_vertex_ids = set()
    counted_waypoint_vertex_ids = set()
    first_odom_processed = False
    num_tag_edges = 0
    # DEBUG: this appears to be counterproductive
    initialize_with_averages = False
    tag_transform_estimates = defaultdict(lambda: [])

    for i, odom_frame in enumerate(pose_data[:, 17]):
        current_odom_vertex_uid = vertex_counter
        vertices[current_odom_vertex_uid] = graph.Vertex(
            mode=graph.VertexType.ODOMETRY,
            estimate=odom_vertex_estimates[i],
            fixed=not first_odom_processed,
            meta_data={"poseId": odom_frame},
        )
        first_odom_processed = True
        vertex_counter += 1

        # Connect odom to tag vertex
        for tag_vertex_id, tag_index in tag_vertex_id_and_index_by_frame_id.get(
            int(odom_frame), []
        ):
            current_tag_transform_estimate = (
                SE3Quat(np.hstack((true_3d_tag_center, [0, 0, 0, 1])))
                * SE3Quat(tag_edge_measurements[tag_index]).inverse()
                * SE3Quat(vertices[current_odom_vertex_uid].estimate)
            )
            # keep track of estimates in case we want to average them to initialize the graph
            tag_transform_estimates[tag_vertex_id].append(
                current_tag_transform_estimate
            )
            if tag_vertex_id not in counted_tag_vertex_ids:
                vertices[tag_vertex_id] = graph.Vertex(
                    mode=graph.VertexType.TAG,
                    estimate=current_tag_transform_estimate.to_vector(),
                    fixed=fix_tag_vertices,
                )
                vertices[tag_vertex_id].meta_data["tag_id"] = tag_id_by_tag_vertex_id[
                    tag_vertex_id
                ]
                for idx, true_point_3d in enumerate(true_3d_points):
                    vertices[
                        tag_corner_ids_by_tag_vertex_id[tag_vertex_id][idx]
                    ] = graph.Vertex(
                        mode=graph.VertexType.TAGPOINT,
                        estimate=np.hstack((true_point_3d, [0, 0, 0, 1])),
                        fixed=True,
                    )
                counted_tag_vertex_ids.add(tag_vertex_id)
            # adjust the x-coordinates of the detections to account for differences in coordinate systems induced by
            # the camera_to_odom_transform
            tag_corners[tag_index][::2] = (
                2 * camera_intrinsics_for_tag[tag_index][2]
                - tag_corners[tag_index][::2]
            )
            # TODO: create proper subclasses
            for k, point in enumerate(true_3d_points):
                SE3Quat(tag_edge_measurements[tag_index]) * (
                    point - np.array([0, 0, 1])
                )
                CameraParameters(
                    camera_intrinsics_for_tag[tag_index][0],
                    camera_intrinsics_for_tag[tag_index][2:],
                    0,
                )
                # print("chi2", np.sum(np.square(tag_corners[tag_index][2*k : 2*k + 2] -
                #                                cam.cam_map(point_in_camera_frame))))
            edges[edge_counter] = graph.Edge(
                startuid=current_odom_vertex_uid,
                enduid=tag_vertex_id,
                corner_ids=tag_corner_ids_by_tag_vertex_id[tag_vertex_id],
                information=np.eye(2),
                information_prescaling=None,
                camera_intrinsics=camera_intrinsics_for_tag[tag_index],
                measurement=tag_corners[tag_index],
            )
            num_tag_edges += 1

            edge_counter += 1

        # Connect odom to waypoint vertex
        for (
            waypoint_vertex_id,
            waypoint_index,
        ) in waypoint_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
            if waypoint_vertex_id not in counted_waypoint_vertex_ids:
                vertices[waypoint_vertex_id] = graph.Vertex(
                    mode=graph.VertexType.WAYPOINT,
                    estimate=(
                        SE3Quat(vertices[current_odom_vertex_uid].estimate).inverse()
                        * SE3Quat(waypoint_edge_measurements[waypoint_index])
                    ).to_vector(),
                    fixed=False,
                )
                vertices[waypoint_vertex_id].meta_data[
                    "name"
                ] = waypoint_name_by_vertex_id[waypoint_vertex_id]
                counted_waypoint_vertex_ids.add(waypoint_vertex_id)

            edges[edge_counter] = graph.Edge(
                startuid=current_odom_vertex_uid,
                enduid=waypoint_vertex_id,
                corner_ids=None,
                information=np.eye(6),
                information_prescaling=None,
                camera_intrinsics=None,
                measurement=(
                    SE3Quat(vertices[waypoint_vertex_id].estimate)
                    * SE3Quat(vertices[current_odom_vertex_uid].estimate).inverse()
                ).to_vector(),
            )

            edge_counter += 1

        if previous_vertex:
            # TODO: might want to consider prescaling based on the magnitude of the change
            edges[edge_counter] = graph.Edge(
                startuid=previous_vertex,
                enduid=current_odom_vertex_uid,
                corner_ids=None,
                information=np.eye(6),
                information_prescaling=None,
                camera_intrinsics=None,
                measurement=(
                    SE3Quat(vertices[current_odom_vertex_uid].estimate)
                    * SE3Quat(vertices[previous_vertex].estimate).inverse()
                ).to_vector(),
            )
            edge_counter += 1
        dummy_node_uid = vertex_counter
        vertices[dummy_node_uid] = graph.Vertex(
            mode=graph.VertexType.DUMMY,
            estimate=np.hstack(
                (
                    np.zeros(
                        3,
                    ),
                    odom_vertex_estimates[i][3:],
                )
            ),
            fixed=True,
        )
        vertex_counter += 1
        edges[edge_counter] = graph.Edge(
            startuid=current_odom_vertex_uid,
            enduid=dummy_node_uid,
            corner_ids=None,
            information=np.eye(6),
            information_prescaling=None,
            camera_intrinsics=None,
            measurement=np.array([0, 0, 0, 0, 0, 0, 1]),
        )
        edge_counter += 1

        previous_vertex = current_odom_vertex_uid
    if initialize_with_averages:
        for vertex_id, transforms in tag_transform_estimates.items():
            vertices[vertex_id].estimate = se3_quat_average(transforms).to_vector()

    # TODO: Huber delta should probably scale with pixels rather than error
    resulting_graph = graph.Graph(
        vertices,
        edges,
        gravity_axis="y",
        is_sparse_bundle_adjustment=True,
        use_huber=False,
        huber_delta=None,
        damping_status=True,
    )
    return resulting_graph
