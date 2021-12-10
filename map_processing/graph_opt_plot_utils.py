"""
Plotting utilities for graph optimization
"""

from typing import *

import numpy as np
from g2o import Quaternion, SE3Quat
from matplotlib import pyplot as plt, cm

from . import graph_opt_utils, transform_utils


def plot_metrics(sweep: np.ndarray, metrics: np.ndarray, log_sweep: bool = False, log_metric: bool = False):
    sweep_plot = np.log(sweep) if log_sweep else sweep
    to_plot = np.log(metrics) if log_metric else metrics
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(sweep_plot, sweep_plot.transpose(), to_plot, cmap=cm.get_cmap('viridis'))
    ax.set_xlabel('Pose:Orientation')
    ax.set_ylabel('Odom:Tag')
    ax.set_zlabel('Metric')
    fig.colorbar(surf)
    plt.show()


def plot_optimization_result(
        locations: np.ndarray,
        prior_locations: np.ndarray,
        tag_verts: np.ndarray,
        tagpoint_positions: np.ndarray,
        waypoint_verts: Tuple[List, np.ndarray],
        original_tag_verts: Optional[np.ndarray] = None,
        ground_truth_tags: Optional[np.ndarray] = None,
        plot_title: Union[str, None] = None,
        is_sba: bool = False
) -> None:
    """Visualization used during the optimization routine.
    """
    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.6, 0.75], projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(120, -90)

    plt.plot(prior_locations[:, 0], prior_locations[:, 1], prior_locations[:, 2], "-", c="g",
             label="Prior Odom Vertices")
    plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], "-", c="b", label="Odom Vertices")

    if original_tag_verts is not None:
        original_tag_verts = np.array(original_tag_verts)  # Copy to avoid modifying input
        if is_sba:
            transform_utils.apply_z_translation_to_lhs_of_se3_vectors(original_tag_verts)
        plt.plot(original_tag_verts[:, 0], original_tag_verts[:, 1], original_tag_verts[:, 2], "o", c="c",
                 label="Tag Vertices Original")

    # Fix the 1-meter offset on the tag anchors
    if is_sba:
        tag_verts = np.array(tag_verts)  # Copy to avoid modifying input
        transform_utils.apply_z_translation_to_lhs_of_se3_vectors(tag_verts)

    if ground_truth_tags is not None:
        # noinspection PyTypeChecker
        tag_list: List = tag_verts.tolist()
        tag_list.sort(key=lambda x: x[-1])
        ordered_tags = np.asarray([tag[0:-1] for tag in tag_list])

        anchor_tag = 0
        anchor_tag_se3quat = SE3Quat(ordered_tags[anchor_tag])
        to_world = anchor_tag_se3quat * ground_truth_tags[anchor_tag].inverse()
        world_frame_ground_truth = np.asarray([(to_world * tag).to_vector() for tag in ground_truth_tags])
        print("\nAverage translation difference:", graph_opt_utils.ground_truth_metric(ordered_tags, ground_truth_tags,
                                                                                       True))
        plt.plot(world_frame_ground_truth[:, 0], world_frame_ground_truth[:, 1], world_frame_ground_truth[:, 2],
                 'o', c='k', label=f'Actual Tags')
        for i, tag in enumerate(world_frame_ground_truth):
            ax.text(tag[0], tag[1], tag[2], str(i), c='k')

    plt.plot(tag_verts[:, 0], tag_verts[:, 1], tag_verts[:, 2], "o", c="r", label="Tag Vertices")
    for tag_vert in tag_verts:
        R = Quaternion(tag_vert[3:-1]).rotation_matrix()
        axis_to_color = ["r", "g", "b"]
        for axis_id in range(3):
            ax.quiver(tag_vert[0], tag_vert[1], tag_vert[2], R[0, axis_id], R[1, axis_id],
                      R[2, axis_id], length=1, color=axis_to_color[axis_id])

    plt.plot(tagpoint_positions[:, 0], tagpoint_positions[:, 1], tagpoint_positions[:, 2], ".", c="m",
             label="Tag Corners")

    for vert in tag_verts:
        ax.text(vert[0], vert[1], vert[2], str(int(vert[-1])), color="black")

    plt.plot(waypoint_verts[1][:, 0], waypoint_verts[1][:, 1], waypoint_verts[1][:, 2], "o", c="y",
             label="Waypoint Vertices")
    for vert_idx in range(len(waypoint_verts[0])):
        vert = waypoint_verts[1][vert_idx]
        waypoint_name = waypoint_verts[0][vert_idx]["name"]
        ax.text(vert[0], vert[1], vert[2], waypoint_name, color="black")

    # Archive of plotting commands:
    # plt.plot(all_tags[:, 0], all_tags[:, 1], all_tags[:, 2], '.', c='g', label='All Tag Edges')
    # plt.plot(all_tags_original[:, 0], all_tags_original[:, 1], all_tags_original[:, 2], '.', c='m',
    #          label='All Tag Edges Original')
    # all_tags = graph_utils.get_tags_all_position_estimate(graph)
    # tag_edge_std_dev_before_and_after = compare_std_dev(all_tags, all_tags_original)
    # tag_vertex_shift = original_tag_verts - tag_verts
    # print("tag_vertex_shift", tag_vertex_shift)

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
    axis_equal(ax)
    plt.gcf().set_dpi(300)
    if isinstance(plot_title, str):
        plt.title(plot_title, wrap=True)
    plt.show()


def axis_equal(ax: plt.Axes):
    """Create cubic bounding box to simulate equal aspect ratio

    Args:
        ax: Matplotlib Axes object
    """
    axis_range_from_limits = lambda limits: limits[1] - limits[0]
    max_range = np.max(np.array([axis_range_from_limits(ax.get_xlim()), axis_range_from_limits(ax.get_ylim()),
                                 axis_range_from_limits(ax.get_zlim())]))
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * \
        (ax.get_xlim()[1] + ax.get_xlim()[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * \
        (ax.get_ylim()[1] + ax.get_ylim()[0])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * \
        (ax.get_zlim()[1] + ax.get_zlim()[0])

    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")


def plot_adj_chi2(map_from_opt: Dict, plot_title: Union[str, None] = None) -> None:
    """TODO: Documentation

    Args:
        map_from_opt:
        plot_title:
    """
    locations_chi2_viz_tags = []
    locations_shape = np.shape(map_from_opt["locations"])
    for i in range(locations_shape[0]):
        locations_chi2_viz_tags.append((map_from_opt["locations"][i], map_from_opt["locationsAdjChi2"][i],
                                        map_from_opt["visibleTagsCount"][i]))
    locations_chi2_viz_tags.sort(key=lambda x: x[0][7])  # Sorts by UID, which is at the 7th index

    chi2_values = np.zeros([locations_shape[0], 1])  # Contains adjacent chi2 values
    viz_tags = np.zeros([locations_shape[0], 3])
    odom_uids = np.zeros([locations_shape[0], 1])  # Contains UIDs
    for idx in range(locations_shape[0]):
        chi2_values[idx] = locations_chi2_viz_tags[idx][1]
        odom_uids[idx] = int(locations_chi2_viz_tags[idx][0][7])

        # Each row: UID, chi2_value, and num. viz. tags (only if != 0). As of now, the number of visible tags is
        # ignored when plotting (plot only shows boolean value: whether at least 1 tag vertex is visible)
        num_tag_verts = locations_chi2_viz_tags[idx][2]
        if num_tag_verts != 0:
            viz_tags[idx, :] = np.array([odom_uids[idx], chi2_values[idx], num_tag_verts]).flatten()

    odom_uids.flatten()
    chi2_values.flatten()

    f = plt.figure()
    ax: plt.Axes = f.add_axes([0.1, 0.1, 0.8, 0.7])
    ax.plot(odom_uids, chi2_values)
    ax.scatter(viz_tags[:, 0], viz_tags[:, 1], marker="o", color="red")
    ax.set_xlim(min(odom_uids), max(odom_uids))
    ax.legend(["chi2 value", ">=1 tag vertex visible"])
    ax.set_xlabel("Odometry vertex UID")

    plt.xlabel("Odometry vertex UID")
    if plot_title is not None:
        plt.title(plot_title, wrap=True)
    plt.show()
