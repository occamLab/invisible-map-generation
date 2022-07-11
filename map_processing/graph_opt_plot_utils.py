"""
Plotting utilities for graph optimization.
"""

import pdb
from typing import *

import numpy as np
from g2o import SE3Quat
from matplotlib import pyplot as plt, cm

from map_processing.data_models import OG2oOptimizer
from map_processing.transform_utils import transform_vector_to_matrix, transform_gt_to_have_common_reference

# Arrays for use in drawing reference frames
X_HAT_1X3 = np.array(((1, 0, 0),))
Y_HAT_1X3 = np.array(((0, 1, 0),))
Z_HAT_1X3 = np.array(((0, 0, 1),))
BASES_COLOR_CODES = ("r", "g", "b")


def draw_frames(poses: np.ndarray, plt_axes: plt.Axes, colors: Tuple[str, str, str] = BASES_COLOR_CODES) -> None:
    """Draw an arbitrary number (N) of reference frames at given translation offsets.

    Args:
        poses: Pose(s) whose basis vectors are plotted. Can be provided in one of a few formats: (1) A Nx7 vector of N
         SE3Quat vectors, (2) a length-7 1-dimensional SE3Quat vector, (3) a Nx4x4 array of N homogenous transform
         matrices, or (4) a 4x4 2-dimensional transform matrix.
        plt_axes: Matplotlib axes to plot on
        colors: Tuple of color codes to use for the first, second, and third dimensions' basis vector arrows,
         respectively.

    Raises:
        ValueError: if the input array shapes are not as expected
    """
    rot_mats: Optional[np.ndarray] = None
    offsets: Optional[np.ndarray] = None

    if len(poses.shape) == 1 and poses.shape[0] == 7:
        pose_mat = transform_vector_to_matrix(poses)
        rot_mats = np.expand_dims(pose_mat[:3, :3], 1)
        offsets = pose_mat[:3, 3].transpose()
    elif len(poses.shape) == 2:
        if poses.shape[1] == 7:
            pose_mats = np.zeros([poses.shape[0], 4, 4])
            for frame_idx in range(poses.shape[0]):
                pose_mats[frame_idx, :, :] = transform_vector_to_matrix(poses[frame_idx, :])
            rot_mats = pose_mats[:, :3, :3]
            offsets = pose_mats[:, :3, 3].transpose()
        elif poses.shape == (4, 4):
            rot_mats = np.expand_dims(poses[:3, :3], 0)
            offsets = np.expand_dims(poses[:3, 3].transpose(), 1)
    elif len(poses.shape) == 3 and poses.shape[1:] == (4, 4):
        rot_mats = poses[:, :3, :3]
        offsets = poses[:, :3, 3].transpose()

    if rot_mats is None or offsets is None:
        raise ValueError(f"poses argument was of an invalid shape: {poses.shape}")

    for b in range(3):
        basis_vecs = (rot_mats[:, :, b]).transpose()

        plt_axes.quiver(
            offsets[0, :],
            offsets[1, :],
            offsets[2, :],
            # For each basis vector, dot it with the corresponding basis vector of the reference frame it is within
            np.matmul(X_HAT_1X3, basis_vecs),
            np.matmul(Y_HAT_1X3, basis_vecs),
            np.matmul(Z_HAT_1X3, basis_vecs),
            length=0.5,
            arrow_length_ratio=0.3,
            normalize=True,
            color=colors[b],
        )


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
        opt_odometry: np.ndarray,
        orig_odometry: np.ndarray,
        opt_tag_verts: np.ndarray,
        opt_tag_corners: np.ndarray,
        opt_waypoint_verts: Tuple[List, np.ndarray],
        orig_tag_verts: Optional[np.ndarray] = None,
        ground_truth_tags: Optional[List[SE3Quat]] = None,
        plot_title: Union[str, None] = None
) -> None:
    """Visualization used during the optimization routine.

    Notes:
        Assumes that `ground_truth_tags` has been sorted such that the tags poses are in ascending order according to
        their tag IDs.
    """
    f = plt.figure()
    ax = f.add_axes([0.1, 0.1, 0.6, 0.75], projection="3d")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(120, -90)

    plt.plot(orig_odometry[:, 0], orig_odometry[:, 1], orig_odometry[:, 2], "-", c="g", label="Prior Odom Vertices")
    plt.plot(opt_odometry[:, 0], opt_odometry[:, 1], opt_odometry[:, 2], "-", c="b", label="Odom Vertices")
    plt.plot(opt_tag_corners[:, 0], opt_tag_corners[:, 1], opt_tag_corners[:, 2], ".", c="m", label="Tag Corners")

    # Plot optimized tag vertices, their reference frames, and their labels
    draw_frames(opt_tag_verts[:, :7], plt_axes=ax)
    plt.plot(opt_tag_verts[:, 0], opt_tag_verts[:, 1], opt_tag_verts[:, 2], "o", c="#ff8000",
             label="Tag Vertices Optimized")

    draw_frames(orig_tag_verts[:, :-1], plt_axes=ax)
    for vert in opt_tag_verts:
        ax.text(vert[0], vert[1], vert[2], str(int(vert[-1])), color="#663300")

    # Plot original tag vertices and their labels
    if orig_tag_verts is not None:
        plt.plot(orig_tag_verts[:, 0], orig_tag_verts[:, 1], orig_tag_verts[:, 2], "o", c="c",
                 label="Tag Vertices Original")
        for vert in orig_tag_verts:
            ax.text(vert[0], vert[1], vert[2], str(int(vert[-1])), color="#006666")

    # Plot ground truth vertices and their labels
    if ground_truth_tags is not None:
        # pdb.set_trace()
        # noinspection PyTypeChecker
        opt_tag_list: List = opt_tag_verts.tolist()
        tag_ids = [int(tag[-1]) for tag in opt_tag_list]
        opt_tag_dict = {}
        for se3_quat in opt_tag_list:
            opt_tag_dict[se3_quat[-1]] = se3_quat[:-1]

        matching_ground_truth_tags = {}
        for tag_id in tag_ids:
            if tag_id in ground_truth_tags.as_dict_of_se3_arrays.keys():
                matching_ground_truth_tags[tag_id] = SE3Quat(ground_truth_tags.as_dict_of_se3_arrays[tag_id])
        # ordered_opt_tags_array = np.asarray([tag[0:-1] for tag in opt_tag_list])

        anchor_tag_id = 16  # Select arbitrarily
        print("Here")
        world_frame_ground_truth = transform_gt_to_have_common_reference(
            anchor_pose=SE3Quat(opt_tag_dict[anchor_tag_id]),
            anchor_id=anchor_tag_id, ground_truth_tags=matching_ground_truth_tags)

        tag_poses_to_plot = []
        for (_, tag_pose) in world_frame_ground_truth.items():
            tag_poses_to_plot.append(tag_pose)
        tag_poses_to_plot = np.array(tag_poses_to_plot)
        
        plt.plot(tag_poses_to_plot[:, 0], tag_poses_to_plot[:, 1], tag_poses_to_plot[:, 2],
                 "o", c="k", label=f"Ground Truth Tags (anchor id={anchor_tag_id})")
        draw_frames(tag_poses_to_plot, plt_axes=ax)
        # for i, tag in enumerate(world_frame_ground_truth):
        #     ax.text(tag[0], tag[1], tag[2], str(tag_ids[i]), c='k')

    # Plot waypoint vertices and their labels
    plt.plot(opt_waypoint_verts[1][:, 0], opt_waypoint_verts[1][:, 1], opt_waypoint_verts[1][:, 2], "o", c="y",
             label="Waypoint Vertices")
    for vert_idx in range(len(opt_waypoint_verts[0])):
        vert = opt_waypoint_verts[1][vert_idx]
        waypoint_name = opt_waypoint_verts[0][vert_idx]["name"]
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


def plot_adj_chi2(map_from_opt: OG2oOptimizer, plot_title: Union[str, None] = None) -> None:
    """TODO: Documentation

    Args:
        map_from_opt:
        plot_title:
    """
    locations_chi2_viz_tags = []
    locations_shape = np.shape(map_from_opt.locations)
    for i in range(locations_shape[0]):
        locations_chi2_viz_tags.append((map_from_opt.locations[i], map_from_opt.locationsAdjChi2[i],
                                        map_from_opt.visibleTagsCount[i]))
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
