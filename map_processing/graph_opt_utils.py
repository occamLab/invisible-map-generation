"""
Utility functions for graph optimization.
"""

import math
from typing import Union, List, Optional, Set, Type, Tuple

import g2o
import numpy as np
# noinspection PyUnresolvedReferences
from g2o import EdgeSE3Gravity
from g2o import SE3Quat, EdgeProjectPSI2UV, EdgeSE3Expmap, EdgeSE3, VertexSE3, VertexSE3Expmap

from . import graph_util_get_neighbors, VertexType
from .data_models import PGTranslation, PGRotation, PGTagVertex, PGOdomVertex, PGWaypointVertex, PGDataSet, \
    OG2oOptimizer
from .transform_utils import transform_gt_to_have_common_reference


def optimizer_to_map(vertices, optimizer: g2o.SparseOptimizer, is_sba=False) -> OG2oOptimizer:
    """Convert a :class: g2o.SparseOptimizer to a dictionary containing locations of the phone, tags, and waypoints.

    Args:
        vertices: A dictionary of vertices. This is used to look up the type of vertex pulled from the optimizer.
        optimizer: a :class: g2o.SparseOptimizer containing a map.
        is_sba: Set to True if the optimizer is based on sparse bundle adjustment and False
         otherwise. If true, the odometry locations and the tag vertices' poses are inverted. In the case of the tag
         vertices, the poses are first transformed by a -1 translation (applied on the LHS of the pose) before
         inversion.

    Returns:
        A dictionary with fields 'locations', 'tags', and 'waypoints'. The 'locations' key covers a (n, 8) array
         containing x, y, z, qx, qy, qz, qw locations of the phone as well as the vertex uid at n points. The 'tags' and
        'waypoints' keys cover the locations of the tags and waypoints in the same format.
    """
    locations = []
    tagpoints = []
    tags = []
    waypoints = []
    waypoint_metadata = []
    exaggerate_tag_corners = True
    for i in optimizer.vertices():
        mode = vertices[i].mode
        if mode == VertexType.TAGPOINT:
            tag_vert = optimizer_find_connected_tag_vert(optimizer, optimizer.vertex(i))
            if tag_vert is None:
                # TODO: double-check that the right way to handle this case is to continue
                continue
            location = optimizer.vertex(i).estimate()
            if exaggerate_tag_corners:
                location = location * np.array([10, 10, 1])
            tagpoints.append(tag_vert.estimate().inverse() * location)
        else:
            location = optimizer.vertex(i).estimate().translation()
            rotation = optimizer.vertex(i).estimate().rotation().coeffs()
            pose = np.concatenate([location, rotation])

            if mode == VertexType.ODOMETRY:
                if is_sba:
                    pose = SE3Quat(pose).inverse().to_vector()
                pose_with_metadata = np.concatenate([pose, [i], [vertices[i].meta_data['pose_id']]])
                locations.append(pose_with_metadata)
            elif mode == VertexType.TAG:
                pose_with_metadata = np.concatenate([pose, [i]])
                if is_sba:
                    # Adjust tag based on the position of the tag center
                    pose_with_metadata[:-1] = (SE3Quat([0, 0, -1, 0, 0, 0, 1]) * SE3Quat(pose)).inverse().to_vector()
                if 'tag_id' in vertices[i].meta_data:
                    pose_with_metadata[-1] = vertices[i].meta_data['tag_id']
                tags.append(pose_with_metadata)
            elif mode == VertexType.WAYPOINT:
                pose_with_metadata = np.concatenate([pose, [i]])
                waypoints.append(pose_with_metadata)
                waypoint_metadata.append(vertices[i].meta_data)
    locations_arr = np.array(locations)
    locations_arr = locations_arr[locations_arr[:, -1].argsort()] if len(locations) > 0 else np.zeros((0, 9))
    tags_arr = np.array(tags) if len(tags) > 0 else np.zeros((0, 8))
    tagpoints_arr = np.array(tagpoints) if len(tagpoints) > 0 else np.zeros((0, 3))
    waypoints_arr = np.array(waypoints) if len(waypoints) > 0 else np.zeros((0, 8))
    return OG2oOptimizer(locations=locations_arr, tags=tags_arr, tagpoints=tagpoints_arr, waypoints_arr=waypoints_arr,
                         waypoints_metadata=waypoint_metadata)


def optimizer_to_map_chi2(graph, optimizer: g2o.SparseOptimizer, is_sba=False) -> OG2oOptimizer:
    """Convert a :class: g2o.SparseOptimizer to a dictionary containing locations of the phone, tags, waypoints, and
    per-odometry edge chi2 information.

    This function works by calling `optimizer_to_map` and adding a new entry that is a vector of the per-odometry edge
    chi2 information as calculated by the `map_odom_to_adj_chi2` method of the `Graph` class.

    Args:
        graph (Graph): A graph instance whose vertices attribute is passed as the first argument to `optimizer_to_map`
         and whose `map_odom_to_adj_chi2` method is used.
        optimizer: a :class: g2o.SparseOptimizer containing a map, which is passed as the second argument to
         `optimizer_to_map`.
        is_sba: True if the optimizer is based on sparse bundle adjustment and False otherwise;
         passed as the `is_sba` keyword argument to `optimizer_to_map`.
    """
    ret_map = optimizer_to_map(graph.vertices, optimizer, is_sba=is_sba)
    locations_shape = np.shape(ret_map.locations)
    locations_adj_chi2 = np.zeros([locations_shape[0], 1])
    visible_tags_count = np.zeros([locations_shape[0], 1])

    for i, odom_node_vec in enumerate(ret_map.locations):
        uid = round(odom_node_vec[7])  # UID integer is stored as a floating point number, so cast it to an integer
        locations_adj_chi2[i], visible_tags_count[i] = graph.map_odom_to_adj_chi2(uid)

    ret_map.locationsAdjChi2 = locations_adj_chi2
    ret_map.visibleTagsCount = visible_tags_count
    return ret_map


def optimizer_find_connected_tag_vert(optimizer: g2o.SparseOptimizer, location_vert):
    """TODO: documentation
    """
    # TODO: it would be nice if we didn't have to scan the entire graph
    for edge in optimizer.edges():
        if type(edge) == EdgeProjectPSI2UV:
            if edge.vertex(0).id() == location_vert.id():
                return edge.vertex(2)
    return None


def get_chi2_of_edge(
        edge: Union[EdgeProjectPSI2UV, EdgeSE3Expmap, EdgeSE3, EdgeSE3Gravity],
        start_vert: Optional[Union[VertexSE3, VertexSE3Expmap]] = None) -> Tuple[float, float]:
    """Computes the chi2 and log-normalized fitness values associated with the provided edge.

    Arguments:
        edge: A g2o edge of type EdgeProjectPSI2UV, EdgeSE3Expmap, EdgeSE3, or EdgeSE3Gravity
        start_vert: The start vertex associated with this edge (only used if the edge is of type EdgeSE3Gravity;
         otherwise, it is ignored.)

    Returns:
        Chi2 value associated with the provided edge.

    Notes:
        TODO: documentation on the alternative fitness metric.

    Raises:
        ValueError - If an edge is encountered that is not handled (handled edges are EdgeProjectPSI2UV,
         EdgeSE3Expmap, EdgeSE3, and EdgeSE3Gravity).
        ValueError - If the edge is of type EdgeSE3Gravity and no start vertex was provided.
        ValueError - If the resulting chi2 value ends up being NaN.

    Notes:
        TODO: Explain the log normalization stuff.
    """
    chi2: float
    information: np.ndarray = edge.information()

    if isinstance(edge, EdgeProjectPSI2UV):
        cam = edge.parameter(0)
        camera_coords = edge.vertex(1).estimate() * edge.vertex(2).estimate().inverse() * edge.vertex(0).estimate()
        pixel_coords = cam.cam_map(camera_coords)
        error = edge.measurement() - pixel_coords
        chi2 = error.dot(information).dot(error)
    elif isinstance(edge, EdgeSE3Expmap):
        error = edge.vertex(1).estimate().inverse() * edge.measurement() * edge.vertex(0).estimate()
        chi2 = error.log().T.dot(information).dot(error.log())
    elif isinstance(edge, EdgeSE3):
        delta = edge.measurement().inverse() * edge.vertex(0).estimate().inverse() * edge.vertex(1).estimate()
        error = np.hstack((delta.translation(), delta.orientation().coeffs()[:-1]))
        chi2 = error.dot(information).dot(error)
    elif isinstance(edge, EdgeSE3Gravity):
        if start_vert is None:
            raise ValueError("No start vertex provided for edge of type EdgeSE3Gravity")
        direction = edge.measurement()[:3]
        measurement = edge.measurement()[3:]
        if isinstance(start_vert, VertexSE3):
            rot_mat = start_vert.estimate().Quaternion().inverse().R
        else:  # start_vert is a VertexSE3Expmap, so don't invert the rotation
            rot_mat = start_vert.estimate().Quaternion().R
        estimate = np.matmul(rot_mat, direction)
        error = estimate - measurement
        chi2 = error.dot(information).dot(error)
    else:
        raise ValueError(f"Unhandled edge type for chi2 calculation: {type(edge)}")

    if math.isnan(chi2):
        raise ValueError(f"chi2 is NaN for: {edge}")

    k: int = information.shape[0]
    c: float = -np.log(np.power(2 * np.pi, -0.5 * k))
    alpha: float = c - np.log(np.sqrt(np.linalg.det(information))) + 0.5 * chi2
    return chi2, alpha


def sum_optimizer_edges_chi2(
        optimizer: g2o.SparseOptimizer,
        edge_type_filter: Optional[Set[Type[Union[EdgeProjectPSI2UV, EdgeSE3Expmap, EdgeSE3Gravity]]]] = None) \
        -> Tuple[float, float]:
    """Iterates through edges in the g2o sparse optimizer object and sums the chi2 values for all the edges.

    Args:
        optimizer: A SparseOptimizer object
        edge_type_filter: A set providing an inclusive filter of the edge types to sum. If no set is provided or an
         empty set is provided, then no edges are filtered.

    Returns:
        Sum of the chi2 and alpha values associated with each edge. See `get_chi2_of_edge` function for more information
        on these values.
    """
    if edge_type_filter is None:
        edge_type_filter = set()

    total_chi2 = 0.0
    total_alpha = 0.0
    for edge in optimizer.edges():
        if len(edge_type_filter) == 0 or type(edge) in edge_type_filter:
            fitness_metrics = get_chi2_of_edge(edge, edge.vertices()[0])
            total_chi2 += fitness_metrics[0]
            total_alpha += fitness_metrics[1]
    return total_chi2, total_alpha


def ground_truth_metric(optimized_tag_verts: np.ndarray, ground_truth_tags: np.ndarray) \
        -> float:
    """Error metric for tag pose accuracy.

    Calculates the transforms from the anchor tag to each other tag for the optimized and the ground truth tags,
    then compares the transforms and finds the difference in the translation components.

    Args:
        optimized_tag_verts: A n-by-7 numpy array containing length-7 pose vectors.
        ground_truth_tags: A n-by-7 numpy array containing length-7 pose vectors.

    Returns:
        A float representing the average difference in tag positions (translation only) in meters.
    """
    num_tags = optimized_tag_verts.shape[0]
    sum_trans_diffs = np.zeros((num_tags,))
    ground_truth_as_se3 = [SE3Quat(tag_pose) for tag_pose in ground_truth_tags]

    for anchor_tag in range(num_tags):
        anchor_tag_se3quat = SE3Quat(optimized_tag_verts[anchor_tag])
        world_frame_ground_truth = transform_gt_to_have_common_reference(IM_anchor_pose=anchor_tag_se3quat, GT_anchor_pose=ground_truth_as_se3[anchor_tag],\
            ground_truth_tags=ground_truth_as_se3)[:, :3]
        sum_trans_diffs += np.linalg.norm(world_frame_ground_truth - optimized_tag_verts[:, :3], axis=1)
    avg_trans_diffs = sum_trans_diffs / num_tags
    avg = float(np.mean(avg_trans_diffs))
    return avg


def make_processed_map_json(opt_result: OG2oOptimizer, calculate_intersections: bool = False) \
        -> str:
    """Serializes the result of an optimization into a JSON that is of an acceptable format for uploading to the
    database.

    Args:
        opt_result: A dictionary containing the tag locations, odometry locations, waypoint locations,
         odometry-adjacent chi2 array, and per-odometry-node visible tags count array in the keys 'tags', 'locations',
         'waypoints', 'locationsAdjChi2', and 'visibleTagsCount', respectively. This is the format of dictionary that is
         produced by the `map_processing.graph_opt_utils.optimizer_to_map_chi2` function and, subsequently, the
         `GraphManager.optimize_graph` method.
        calculate_intersections: If true, graph_util_get_neighbors.get_neighbors is called with the odometry nodes
         as the argument. The results are appended to the resulting tag vertex map under the 'neighbors' key.

    Returns:
        Json string containing the serialized results.

    Raises:
        ValueError - If both the `visible_tags_count` and `adj_chi2_arr` arguments are None or not None.
        pydantic.ValidationError - If the input arguments are not of the correct format to be parsed into any of the
         relevant data set models in the `data_set_models` module.
    """
    tag_locations = opt_result.tags
    odom_locations = opt_result.locations
    adj_chi2_arr = opt_result.locationsAdjChi2
    visible_tags_count = opt_result.visibleTagsCount
    waypoint_locations = (opt_result.waypoints_metadata, opt_result.waypoints_arr)

    if (visible_tags_count is None) ^ (adj_chi2_arr is None):
        raise ValueError("'visible_tags_count' and 'adj_chi2_arr' arguments must both be None or non-None")

    tag_vertex_list: List[PGTagVertex] = []
    for curr_tag in tag_locations:
        tag_vertex_list.append(
            PGTagVertex(
                translation=PGTranslation(x=curr_tag[0], y=curr_tag[1], z=curr_tag[2]),
                rotation=PGRotation(x=curr_tag[3], y=curr_tag[4], z=curr_tag[5], w=curr_tag[6]),
                id=int(curr_tag[7])
            )
        )

    odom_vertex_list: List[PGOdomVertex] = []
    for curr_odom in odom_locations:
        odom_vertex_list.append(
            PGOdomVertex(
                translation=PGTranslation(x=curr_odom[0], y=curr_odom[1], z=curr_odom[2]),
                rotation=PGRotation(x=curr_odom[3], y=curr_odom[4], z=curr_odom[5], w=curr_odom[6]),
                poseId=int(curr_odom[8]),
            )
        )

    if adj_chi2_arr is not None:
        odom_locations_with_chi2_and_viz_tags = np.concatenate(
            [odom_locations, adj_chi2_arr, visible_tags_count], axis=1)
        for odom_idx, curr_odom in enumerate(odom_locations_with_chi2_and_viz_tags):
            odom_vertex_list[odom_idx].adjChi2 = curr_odom[9]
            odom_vertex_list[odom_idx].vizTags = curr_odom[10]

    if calculate_intersections:
        neighbors_list, intersections = graph_util_get_neighbors.get_neighbors(odom_locations[:, :7])
        for index, neighbors in enumerate(neighbors_list):
            odom_vertex_list[index].neighbors = neighbors
        for intersection in intersections:
            odom_vertex_list.append(PGOdomVertex(**intersection))

    waypoint_vertex_list: List[PGWaypointVertex] = []
    for idx in range(len(waypoint_locations[0])):
        waypoint_vertex_list.append(
            PGWaypointVertex(
                translation=PGTranslation(x=waypoint_locations[1][idx][0], y=waypoint_locations[1][idx][1],
                                          z=waypoint_locations[1][idx][2]),
                rotation=PGRotation(x=waypoint_locations[1][idx][3], y=waypoint_locations[1][idx][4],
                                    z=waypoint_locations[1][idx][5], w=waypoint_locations[1][idx][6]),
                id=waypoint_locations[0][idx]["name"]
            )
        )

    processed_data_set = PGDataSet(
        tag_vertices=tag_vertex_list, odometry_vertices=odom_vertex_list, waypoints_vertices=waypoint_vertex_list)
    return processed_data_set.json(indent=2)


def compare_std_dev(all_tags, all_tags_original):
    """TODO: documentation
    """
    return {int(tag_id): (np.std(all_tags_original[all_tags_original[:, -1] == tag_id, :-1], axis=0),
                          np.std(all_tags[all_tags[:, -1] == tag_id, :-1], axis=0)) for tag_id in
            np.unique(all_tags[:, -1])}
