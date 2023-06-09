import os
import pdb
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from run import run_sweep
from map_processing.cache_manager import CacheManagerSingleton
from firebase_admin import credentials
from map_processing.data_models import OSweepResults
from map_processing.transform_utils import transform_vector_to_matrix
import numpy as np
import matplotlib.pyplot as plt


# The shift metric will be tested by utilizing two different
# maps. The optimization mapped will be optimized with a set
# of parameters in order to obtain global positions of each
# tag. The trajectory map will be used as a "test" map that
# will be iterated across to determine the shift metric.
TRAJECTORY_MAP_NAME = "generated_23-01-09-21-07-32.json"
OPTIMIZATION_MAP_NAME = "r1-single-straight-3round*"
SBA = True

CAMERA_POSE_FLIPPER = np.array(
    [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
)


class TagDetection:
    tag_pose: np.ndarray = np.ones(shape=[16, 1])
    tag_id: int = 0
    odom_pose: np.ndarray = np.ones(shape=[16, 1])
    odom_id: int = 0
    opt_tag_pose: np.ndarray = np.ones(shape=[])

    @property
    def tag_pose_homogeneous(self):
        return np.reshape(np.array(self.tag_pose), (4, 4), order="C")

    @property
    def odom_pose_homogeneous(self):
        return np.reshape(np.array(self.odom_pose), (4, 4), order="F")

    @property
    def tag_pose_global(self):
        return (
            self.odom_pose_homogeneous @ CAMERA_POSE_FLIPPER @ self.tag_pose_homogeneous
        )

    @property
    def opt_tag_pose_homogeneous(self):
        return transform_vector_to_matrix(self.opt_tag_pose)


class DetectionPair:
    def __init__(self, detection_one: TagDetection, detection_two: TagDetection):
        self.detection_one = detection_one
        self.detection_two = detection_two

    @property
    def relative_tag_transform(self):
        return self.detection_two.tag_pose_global @ np.linalg.inv(
            self.detection_one.tag_pose_global
        )

    @property
    def relative_tag_transform_trans(self):
        return self.relative_tag_transform[0:3, 3]

    @property
    def opt_tag_one_on_tag_one(self):
        return (
            self.detection_one.opt_tag_pose_homogeneous
            @ self.detection_one.tag_pose_global
            @ np.linalg.inv(self.detection_one.opt_tag_pose_homogeneous)
        )

    @property
    def opt_tag_two_on_tag_one(self):
        return (
            self.detection_two.opt_tag_pose_homogeneous
            @ self.detection_one.tag_pose_global
            @ np.linalg.inv(self.detection_one.opt_tag_pose_homogeneous)
        )

    @property
    def relative_opt_tag_transform(self):
        return self.opt_tag_two_on_tag_one @ np.linalg.inv(self.opt_tag_one_on_tag_one)

    @property
    def opt_tag_shift(self):
        return self.relative_tag_transform @ np.linalg.inv(
            self.relative_opt_tag_transform
        )

    @property
    def opt_tag_shift_pos(self):
        return np.abs(self.opt_tag_shift[0:3, 3])


def find_maps_and_data():
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    cms = CacheManagerSingleton(
        firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
    )

    # Assumes only one map exists in the .cache folder for the optimization and trajectory
    optimization_map_info = cms.find_maps(
        OPTIMIZATION_MAP_NAME, search_restriction=2
    ).pop()
    trajectory_map_info = cms.find_maps(TRAJECTORY_MAP_NAME, search_restriction=2).pop()

    # Both maps should have the same ground truth
    gt_data = cms.find_ground_truth_data_from_map_info(optimization_map_info)

    return optimization_map_info, trajectory_map_info, gt_data


def run_optimization():
    opt_results = run_sweep(OPTIMIZATION_MAP_NAME, 0 if SBA else 1)
    return opt_results


def calculate_shift_metric(opt_results, trajectory_map_dct):
    trajectory_tag_data = trajectory_map_dct["tag_data"]

    shift_metric = []

    for oresult_index, oresult in enumerate(opt_results):
        shift_metric.append(np.zeros(4))
        opt_tag_pos_dct = {}
        for tag in oresult.map_opt.tags:
            opt_tag_pos_dct[int(tag[-1])] = tag[:-1]

        tag_pairs = []

        for i in range(len(trajectory_tag_data) - 1):
            if (
                trajectory_tag_data[i][0]["tag_id"]
                == trajectory_tag_data[i + 1][0]["tag_id"]
            ):
                continue
            detection_one = TagDetection()
            detection_one.tag_id = trajectory_tag_data[i][0]["tag_id"]
            detection_one.tag_pose = trajectory_tag_data[i][0]["tag_pose"]
            detection_one.odom_id = trajectory_tag_data[i][0]["pose_id"]
            detection_one.odom_pose = trajectory_map_dct["pose_data"][
                detection_one.odom_id
            ]["pose"]
            detection_one.opt_tag_pose = opt_tag_pos_dct[detection_one.tag_id]

            detection_two = TagDetection()
            detection_two.tag_id = trajectory_tag_data[i + 1][0]["tag_id"]
            detection_two.tag_pose = trajectory_tag_data[i + 1][0]["tag_pose"]
            detection_two.odom_id = trajectory_tag_data[i + 1][0]["pose_id"]
            detection_two.odom_pose = trajectory_map_dct["pose_data"][
                detection_two.odom_id
            ]["pose"]
            detection_two.opt_tag_pose = opt_tag_pos_dct[detection_two.tag_id]

            curr_tag_pair = DetectionPair(
                detection_one=detection_one, detection_two=detection_two
            )
            tag_pairs.append(curr_tag_pair)

            shift_metric[oresult_index][
                :3
            ] += curr_tag_pair.opt_tag_shift_pos / np.linalg.norm(
                curr_tag_pair.relative_tag_transform_trans
            )
            shift_metric[oresult_index][3] += np.linalg.norm(
                curr_tag_pair.opt_tag_shift_pos
            ) / np.linalg.norm(curr_tag_pair.relative_tag_transform_trans)

        shift_metric[oresult_index] /= len(tag_pairs)

    return np.array(shift_metric)


def visualize_results(shift_metrics, opt_results):
    shift_metric_list = shift_metrics[:, 3]
    gt_metric_list = opt_results.gt_results_list
    plt.plot(shift_metric_list, gt_metric_list, "bo")
    plt.plot(
        [min(shift_metric_list), max(shift_metric_list)],
        [opt_results.gt_metric_pre, opt_results.gt_metric_pre],
        "r-",
    )
    plt.legend(["Optimization Result", "Pre-Optimization GT"])
    plt.xlabel("Shift Metric")
    plt.ylabel("GT Metric")
    plt.title(f"Shift Metric Test on Parameter Sweep of {OPTIMIZATION_MAP_NAME}")
    plt.figtext(
        0.5,
        0.01,
        "This test was done on two distinct generated datasets based on {OPTIMIZATION_MAP_NAME}. Each were constructed with 10 obs noise, 0.1 odom pos noise, and 0.0025 odom rot noise",
        wrap=True,
        horizontalalignment="center",
        fontsize=12,
    )

    plt.show()


def run_shift_metric_test():
    optimization_map_info, trajectory_map_info, gt_data = find_maps_and_data()
    opt_results = run_optimization()
    shift_metrics = calculate_shift_metric(
        opt_results.oresults_list, trajectory_map_info.map_dct
    )
    visualize_results(shift_metrics, opt_results)
    pdb.set_trace()
    # print("----------------------------------")
    # print(f"X Shift: {shift_metric[0]:.3f}")
    # print(f"Y Shift: {shift_metric[1]:.3f}")
    # print(f"Z Shift: {shift_metric[2]:.3f}")
    # print(f"Overall Shift: {shift_metric[3]:.3f}")


def visualize_results_one_dataset(opt_results: OSweepResults):
    shift_metric_list = opt_results.shift_metric_list
    gt_metric_list = opt_results.gt_results_list
    plt.plot(shift_metric_list, gt_metric_list, "bo")
    plt.plot(
        [min(shift_metric_list), max(shift_metric_list)],
        [opt_results.gt_metric_pre, opt_results.gt_metric_pre],
        "r-",
    )
    plt.legend(["Optimization Result", "Pre-Optimization GT"])
    plt.xlabel("Shift Metric")
    plt.ylabel("GT Metric")
    plt.title(f"Shift Metric Test on Parameter Sweep of {OPTIMIZATION_MAP_NAME}")
    plt.figtext(
        0.5,
        0.01,
        f"This test was done on the real recording of {OPTIMIZATION_MAP_NAME} w/ SBA. The shift metric was calculated by comparing the optimized tags to unoptimized detections.\nPre-Opt GT: {opt_results.gt_metric_pre}    Min GT: {opt_results.min_gt}    Min Shift GT: {opt_results.min_shift_gt}",
        wrap=True,
        horizontalalignment="center",
        fontsize=10,
    )

    plt.show()


def run_one_dataset_shift_metric_test():
    opt_results: OSweepResults = run_optimization()
    visualize_results_one_dataset(opt_results)


if __name__ == "__main__":
    # run_shift_metric_test()
    run_one_dataset_shift_metric_test()
