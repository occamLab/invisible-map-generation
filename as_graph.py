import itertools
from collections import defaultdict
from enum import Enum
from typing import Union

import numpy as np
from g2o import SE3Quat, Quaternion
from scipy.spatial.transform import Rotation as Rot

import graph
from graph_utils import camera_to_odom_transform


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
        invert (bool): If inverted, then the return enum_value will be inverted
    Returns:
      Converted pose or array of poses.
    """
    translation = pose[..., :3, 3]
    if pose.shape[0] != 0:
        rotation = Rot.from_matrix(pose[..., :3, :3]).as_quat()
    else:
        rotation = np.zeros([0, 4])
    ret_val = np.concatenate([translation, rotation], axis=-1)
    if invert:
        ret_val = np.vstack(list(map(lambda measurement: SE3Quat(measurement).inverse().to_vector(), ret_val)))
    return ret_val


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


class PrescalingOptEnum(Enum):
    """Enum used in the as_graph method to select which approach is taken

    Class attributes:
        USE_SBA: Do not use sparse bundle adjustment
        FULL_COV: When creating the tag edge prescaling matrix, compute it from the covariance matrix calculated
         to account for the reliability of the tag pose estimate.
        DIAG_COV: Same as `FULL_COV`, except only the matrix diagonal is used.
        ONES: Prescaling matrix is set to a matrix of 1s.
    """
    USE_SBA = 0
    FULL_COV = 1
    DIAG_COV = 2
    ONES = 3

    @staticmethod
    def get_by_value(enum_value: int):
        return PrescalingOptEnum._value2member_map_[enum_value]


def make_sba_tag_arrays():
    tag_size = 0.173  # TODO: need to send this with the tag detection
    pos_tag_sz_div_2 = tag_size / 2
    neg_tag_sz_div_2 = - pos_tag_sz_div_2
    true_3d_points = np.array(
        [[neg_tag_sz_div_2, neg_tag_sz_div_2, 1],
         [pos_tag_sz_div_2, neg_tag_sz_div_2, 1],
         [pos_tag_sz_div_2, pos_tag_sz_div_2, 1],
         [neg_tag_sz_div_2, pos_tag_sz_div_2, 1]])
    true_3d_tag_center = np.array([0, 0, 1])
    return true_3d_points, true_3d_tag_center


def as_graph(dct, fix_tag_vertices: bool = False, prescaling_opt: PrescalingOptEnum = PrescalingOptEnum.USE_SBA):
    """Convert a dictionary decoded from JSON into a graph.

    This function was created by combining the as_graph functions from convert_json.py and convert_json_sba.py. Because
    the two implementations shared a lot of code but also deviated in a few important ways, the entire functionality of
    each was preserved in this function by using the prescaling_opt argument to toggle on/off logical branches
    according to the implementation in convert_json.py and the implementation in convert_json_sba.py.

    Args:
        dct (dict): The dictionary to convert to a graph.
        fix_tag_vertices (bool): Passed as the `fixed` keyword argument to the Vertex constructor when constructing
         vertices labeled as Tag vertices.
        prescaling_opt (PrescalingOptEnum): Selects which logical branches to use. If it is equal to
        `PrescalingOptEnum.USE_SBA`,  then sparse bundle adjustment is used; otherwise, the the outcome only differs
         between the remaining enum values by how the tag edge prescaling matrix is selected. Read the PrescalingOptEnum
         class documentation for more information.

    Returns: A graph derived from the input dictionary.

    Raises:
        An exception if prescaling_opt is a enum_value that is not handled.
    """
    # The following variables instantiated to None are optionally used depending on the enum_value of prescaling_opt
    tag_joint_covar = None
    tag_position_variances = None
    tag_orientation_variances = None
    true_3d_tag_center: Union[None, np.ndarray] = None
    true_3d_points: Union[None, np.ndarray] = None
    tag_transform_estimates = None
    tag_corner_ids_by_tag_vertex_id = None
    camera_intrinsics_for_tag: Union[np.ndarray, None] = None
    tag_corners = None
    tag_edge_prescaling = None
    previous_pose_matrix = None
    initialize_with_averages = None

    # Pull out this equality from the enum (this equality is checked many times)
    use_sba = prescaling_opt == PrescalingOptEnum.USE_SBA

    if use_sba:
        true_3d_points, true_3d_tag_center = make_sba_tag_arrays()

    frame_ids = [pose['id'] for pose in dct['pose_data']]
    pose_matrices = np.zeros((0, 16))
    if len(dct['pose_data']) > 0:
        pose_matrices = np.array([pose['pose'] for pose in dct['pose_data']]).reshape((-1, 4, 4)).transpose(0, 2, 1)
    odom_vertex_estimates = matrix2measurement(pose_matrices, invert=use_sba)

    if 'tag_data' in dct and len(dct['tag_data']) > 0:
        good_tag_detections = dct['tag_data']
        # good_tag_detections = list(filter(lambda l: len(l) > 0,
        #                              [[tag_data for tag_data in tags_from_frame
        #                                if np.linalg.norm(np.asarray([tag_data['tag_pose'][i] for i in (3, 7, 11)])) < 1
        #                                and tag_data['tag_pose'][10] < 0.7] for tags_from_frame in dct['tag_data']]))
        tag_pose_flat = np.vstack([[x['tag_pose'] for x in tags_from_frame] for tags_from_frame in good_tag_detections])

        if use_sba:
            camera_intrinsics_for_tag = np.vstack([[x['camera_intrinsics'] for x in tags_from_frame] for tags_from_frame
                                                   in good_tag_detections])
            tag_corners = np.vstack([[x['tag_corners_pixel_coordinates'] for x in tags_from_frame] for tags_from_frame
                                     in good_tag_detections])
        else:
            tag_joint_covar = np.vstack([[x['joint_covar'] for x in tags_from_frame] for tags_from_frame in
                                         good_tag_detections])
            tag_position_variances = np.vstack([[x['tag_position_variance'] for x in tags_from_frame] for
                                                tags_from_frame in good_tag_detections])
            tag_orientation_variances = np.vstack([[x['tag_orientation_variance'] for x in tags_from_frame] for
                                                   tags_from_frame in dct['tag_data']])

        tag_ids = np.vstack(list(itertools.chain(*[[x['tag_id'] for x in tags_from_frame] for tags_from_frame in
                                                   good_tag_detections])))
        pose_ids = np.vstack(list(itertools.chain(*[[x['pose_id'] for x in tags_from_frame] for tags_from_frame in
                                                    good_tag_detections])))
    else:
        tag_pose_flat = np.zeros((0, 16))
        tag_ids = np.zeros((0, 1), dtype=np.int64)
        pose_ids = np.zeros((0, 1), dtype=np.int64)

        if use_sba:
            camera_intrinsics_for_tag = np.zeros((0, 4))
            tag_corners = np.zeros((0, 8))
        else:
            tag_joint_covar = np.zeros((0, 49), dtype=np.double)
            tag_position_variances = np.zeros((0, 3), dtype=np.double)
            tag_orientation_variances = np.zeros((0, 4), dtype=np.double)
    unique_tag_ids = np.unique(tag_ids)
    if use_sba:
        tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(0, unique_tag_ids.size * 5, 5)))
    else:
        tag_vertex_id_by_tag_id = dict(zip(unique_tag_ids, range(unique_tag_ids.size)))

    tag_edge_measurements_matrix = np.matmul(camera_to_odom_transform, tag_pose_flat.reshape([-1, 4, 4]))
    tag_edge_measurements = matrix2measurement(tag_edge_measurements_matrix)
    n_pose_ids = pose_ids.shape[0]

    if not use_sba:
        if prescaling_opt == PrescalingOptEnum.FULL_COV:
            # Note that we are ignoring the variance deviation of qw since we use a compact quaternion parameterization
            # of orientation
            tag_joint_covar_matrices = tag_joint_covar.reshape((-1, 7, 7))

            # TODO: for some reason we have missing measurements (all zeros).  Throw those out
            tag_edge_prescaling = np.array([np.linalg.inv(covar[:-1, :-1]) if np.linalg.det(covar[:-1, :-1]) != 0 else \
                                            np.zeros((6, 6)) for covar in tag_joint_covar_matrices])
        elif prescaling_opt == PrescalingOptEnum.DIAG_COV:
            tag_edge_prescaling = 1. / np.hstack((tag_position_variances, tag_orientation_variances[:, :-1]))
        elif prescaling_opt == PrescalingOptEnum.ONES:
            tag_edge_prescaling = np.ones((n_pose_ids, 6, 6))
        else:
            raise Exception("{} is not yet handled".format(str(prescaling_opt)))

    tag_id_by_tag_vertex_id = dict(zip(tag_vertex_id_by_tag_id.values(), tag_vertex_id_by_tag_id.keys()))
    if use_sba:
        tag_corner_ids_by_tag_vertex_id = dict(
            zip(tag_id_by_tag_vertex_id.keys(),
                map(lambda tag_vertex_id_x: list(range(tag_vertex_id_x + 1, tag_vertex_id_x + 5)),
                    tag_id_by_tag_vertex_id.keys())))

    tag_vertex_id_and_index_by_frame_id = {}  # Enable lookup of tags by the frame they appear in
    for tag_index, (tag_id, tag_frame) in enumerate(np.hstack((tag_ids, pose_ids))):
        tag_vertex_id = tag_vertex_id_by_tag_id[tag_id]
        tag_vertex_id_and_index_by_frame_id[tag_frame] = tag_vertex_id_and_index_by_frame_id.get(tag_frame, [])
        tag_vertex_id_and_index_by_frame_id[tag_frame].append((tag_vertex_id, tag_index))

    waypoint_names = [location_data['name'] for location_data in dct['location_data']]
    unique_waypoint_names = np.unique(waypoint_names)
    num_unique_waypoint_names = unique_waypoint_names.size

    waypoint_edge_measurements_matrix = np.zeros((0, 4, 4))
    if len(dct['location_data']) > 0:
        waypoint_edge_measurements_matrix = np.concatenate([np.asarray(location_data['transform']).reshape((-1, 4, 4))
                                                            for location_data in dct['location_data']])
    waypoint_edge_measurements = matrix2measurement(waypoint_edge_measurements_matrix)
    waypoint_frame_ids = [location_data['pose_id'] for location_data in dct['location_data']]

    if use_sba:
        waypoint_vertex_id_by_name = dict(
            zip(unique_waypoint_names,
                range(unique_tag_ids.size * 5, unique_tag_ids.size * 5 + num_unique_waypoint_names)))
    else:
        waypoint_vertex_id_by_name = dict(
            zip(unique_waypoint_names, range(unique_tag_ids.size, unique_tag_ids.size + num_unique_waypoint_names)))

    waypoint_name_by_vertex_id = dict(zip(waypoint_vertex_id_by_name.values(), waypoint_vertex_id_by_name.keys()))
    waypoint_vertex_id_and_index_by_frame_id = {}  # Enable lookup of waypoints by the frame they appear in

    for waypoint_index, (waypoint_name, waypoint_frame) in enumerate(zip(waypoint_names, waypoint_frame_ids)):
        waypoint_vertex_id = waypoint_vertex_id_by_name[waypoint_name]
        waypoint_vertex_id_and_index_by_frame_id[waypoint_frame] = waypoint_vertex_id_and_index_by_frame_id.get(
            waypoint_name, [])
        waypoint_vertex_id_and_index_by_frame_id[waypoint_frame].append((waypoint_vertex_id, waypoint_index))

    num_tag_edges = edge_counter = 0
    vertices = {}
    edges = {}
    counted_tag_vertex_ids = set()
    counted_waypoint_vertex_ids = set()
    previous_vertex = None
    first_odom_processed = False
    if use_sba:
        vertex_counter = unique_tag_ids.size * 5 + num_unique_waypoint_names
        # TODO: debug; this appears to be counterproductive
        initialize_with_averages = False
        tag_transform_estimates = defaultdict(lambda: [])
    else:
        vertex_counter = unique_tag_ids.size + num_unique_waypoint_names
    for i, odom_frame in enumerate(frame_ids):
        current_odom_vertex_uid = vertex_counter
        vertices[current_odom_vertex_uid] = graph.Vertex(
            mode=graph.VertexType.ODOMETRY,
            estimate=odom_vertex_estimates[i],
            fixed=not first_odom_processed,
            meta_data={'pose_id': odom_frame})
        first_odom_processed = True
        vertex_counter += 1

        # Connect odom to tag vertex
        for tag_vertex_id, tag_index in tag_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
            if use_sba:
                current_tag_transform_estimate = SE3Quat(np.hstack((true_3d_tag_center, [0, 0, 0, 1]))) * SE3Quat(
                    tag_edge_measurements[tag_index]).inverse() * SE3Quat(vertices[current_odom_vertex_uid].estimate)
                # if(tag_vertex_id == 5):
                #     print(current_tag_transform_estimate.to_homogeneous_matrix())
                # keep track of estimates in case we want to average them to initialize the graph
                tag_transform_estimates[tag_vertex_id].append(current_tag_transform_estimate)
                if tag_vertex_id not in counted_tag_vertex_ids:
                    vertices[tag_vertex_id] = graph.Vertex(
                        mode=graph.VertexType.TAG,
                        estimate=current_tag_transform_estimate.to_vector(),
                        fixed=fix_tag_vertices,
                        meta_data={'tag_id': tag_id_by_tag_vertex_id[tag_vertex_id]})

                    for idx, true_point_3d in enumerate(true_3d_points):
                        vertices[tag_corner_ids_by_tag_vertex_id[tag_vertex_id][idx]] = graph.Vertex(
                            mode=graph.VertexType.TAGPOINT,
                            estimate=np.hstack((true_point_3d, [0, 0, 0, 1])),
                            fixed=True)
                    counted_tag_vertex_ids.add(tag_vertex_id)
                # adjust the x-coordinates of the detections to account for differences in coordinate systems induced by
                # the camera_to_odom_transform
                tag_corners[tag_index][::2] = 2 * camera_intrinsics_for_tag[tag_index][2] - tag_corners[tag_index][::2]

                # Commented-out (unused):
                # TODO: create proper subclasses
                # for k, point in enumerate(true_3d_points):
                #     point_in_camera_frame = SE3Quat(tag_edge_measurements[tag_index]) * (point - np.array([0, 0, 1]))
                #     cam = CameraParameters(camera_intrinsics_for_tag[tag_index][0],
                #                            camera_intrinsics_for_tag[tag_index][2:], 0)
                #     print("chi2", np.sum(np.square(tag_corners[tag_index][2*k : 2*k + 2] -
                #                                    cam.cam_map(point_in_camera_frame))))

                edges[edge_counter] = graph.Edge(
                    startuid=current_odom_vertex_uid,
                    enduid=tag_vertex_id,
                    corner_ids=tag_corner_ids_by_tag_vertex_id[tag_vertex_id],
                    information=np.eye(2),
                    information_prescaling=None,
                    camera_intrinsics=camera_intrinsics_for_tag[tag_index],
                    measurement=tag_corners[tag_index]
                )
            else:
                if tag_vertex_id not in counted_tag_vertex_ids:
                    vertices[tag_vertex_id] = graph.Vertex(
                        mode=graph.VertexType.TAG,
                        estimate=matrix2measurement(pose_matrices[i].dot(
                            tag_edge_measurements_matrix[tag_index])),
                        fixed=fix_tag_vertices,
                        meta_data={'tag_id': tag_id_by_tag_vertex_id[tag_vertex_id]})
                    counted_tag_vertex_ids.add(tag_vertex_id)
                edges[edge_counter] = graph.Edge(
                    startuid=current_odom_vertex_uid,
                    enduid=tag_vertex_id,
                    information=np.eye(6),
                    information_prescaling=tag_edge_prescaling[tag_index],
                    measurement=tag_edge_measurements[tag_index],
                    corner_ids=None,
                    camera_intrinsics=None)

            num_tag_edges += 1
            edge_counter += 1

        # Connect odom to waypoint vertex
        for waypoint_vertex_id, waypoint_index in waypoint_vertex_id_and_index_by_frame_id.get(int(odom_frame), []):
            if waypoint_vertex_id not in counted_waypoint_vertex_ids:
                if use_sba:
                    estimate_arg = (SE3Quat(vertices[current_odom_vertex_uid].estimate).inverse() * SE3Quat(
                        waypoint_edge_measurements[waypoint_index])).to_vector()
                else:
                    estimate_arg = matrix2measurement(pose_matrices[i].dot(waypoint_edge_measurements_matrix[
                                                                               waypoint_index]))
                vertices[waypoint_vertex_id] = graph.Vertex(
                    mode=graph.VertexType.WAYPOINT,
                    estimate=estimate_arg,
                    fixed=False,
                    meta_data={'name': waypoint_name_by_vertex_id[waypoint_vertex_id]})
                counted_waypoint_vertex_ids.add(waypoint_vertex_id)

            if use_sba:
                measurement_arg = (SE3Quat(vertices[waypoint_vertex_id].estimate) * SE3Quat(
                    vertices[current_odom_vertex_uid].estimate).inverse()).to_vector()
            else:
                measurement_arg = waypoint_edge_measurements[waypoint_index]
            edges[edge_counter] = graph.Edge(
                startuid=current_odom_vertex_uid,
                enduid=waypoint_vertex_id,
                corner_ids=None,
                information=np.eye(6),
                information_prescaling=None,
                camera_intrinsics=None,
                measurement=measurement_arg)
            edge_counter += 1

        if previous_vertex:
            if use_sba:
                measurement_arg = (SE3Quat(vertices[current_odom_vertex_uid].estimate) * SE3Quat(
                    vertices[previous_vertex].estimate).inverse()).to_vector()
            else:
                # TODO: might want to consider prescaling based on the magnitude of the change
                measurement_arg = matrix2measurement(np.linalg.inv(previous_pose_matrix).dot(pose_matrices[i]))

            edges[edge_counter] = graph.Edge(
                startuid=previous_vertex,
                enduid=current_odom_vertex_uid,
                corner_ids=None,
                information=np.eye(6),
                information_prescaling=None,
                camera_intrinsics=None,
                measurement=measurement_arg)
            edge_counter += 1

        # Make dummy node
        dummy_node_uid = vertex_counter
        vertices[dummy_node_uid] = graph.Vertex(
            mode=graph.VertexType.DUMMY,
            estimate=np.hstack((np.zeros(3, ), odom_vertex_estimates[i][3:])),
            fixed=True)
        vertex_counter += 1

        # Connect odometry to dummy node
        edges[edge_counter] = graph.Edge(
            startuid=current_odom_vertex_uid,
            enduid=dummy_node_uid,
            information=np.eye(6),
            information_prescaling=None,
            measurement=np.array([0, 0, 0, 0, 0, 0, 1]),
            corner_ids=None,
            camera_intrinsics=None)
        edge_counter += 1
        previous_vertex = current_odom_vertex_uid

        if not use_sba:
            previous_pose_matrix = pose_matrices[i]

    if use_sba:
        if initialize_with_averages:
            for vertex_id, transforms in tag_transform_estimates.items():
                vertices[vertex_id].estimate = se3_quat_average(transforms).to_vector()

    # TODO: Huber delta should probably scale with pixels rather than error
    resulting_graph = graph.Graph(vertices, edges, gravity_axis='y', is_sparse_bundle_adjustment=use_sba,
                                  use_huber=False, huber_delta=None, damping_status=True)
    return resulting_graph
