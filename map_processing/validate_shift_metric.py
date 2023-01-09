import os
import pdb
import sys
repository_root = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from map_processing.graph_opt_hl_interface import (
    holistic_optimize,
    WEIGHTS_DICT,
    WeightSpecifier,
)
from map_processing.cache_manager import CacheManagerSingleton
from firebase_admin import credentials
from map_processing.data_models import OComputeInfParams, GTDataSet, OConfig, OResult
from map_processing import PrescalingOptEnum
from map_processing.transform_utils import transform_vector_to_matrix
import numpy as np


# The shift metric will be tested by utilizing two different
# maps. The optimization mapped will be optimized with a set
# of parameters in order to obtain global positions of each
# tag. The trajectory map will be used as a "test" map that
# will be iterated across to determine the shift metric.
TRAJECTORY_MAP_NAME = "generated_23-01-07-22-14-40.json"
OPTIMIZATION_MAP_NAME = "generated_23-01-08-18-48-10.json"
SBA = False
WEIGHTS = WEIGHTS_DICT[WeightSpecifier(5)]

CAMERA_POSE_FLIPPER = np.array(
    [
        [1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, -1, 0], 
        [0, 0, 0, 1]
    ]
)

class TagDetection():
    tag_pose: np.ndarray = np.ones(shape=[16,1])
    tag_id: int = 0
    odom_pose: np.ndarray = np.ones(shape=[16,1])
    odom_id: int = 0
    opt_tag_pose: np.ndarray = np.ones(shape=[])

    @property
    def tag_pose_homogeneous(self):
        return np.reshape(np.array(self.tag_pose), (4,4), order="C")

    @property
    def odom_pose_homogeneous(self):
        return np.reshape(np.array(self.odom_pose), (4,4), order="F")

    @property
    def tag_pose_global(self):
        return self.odom_pose_homogeneous@CAMERA_POSE_FLIPPER@self.tag_pose_homogeneous

    @property
    def opt_tag_pose_homogeneous(self):
        return transform_vector_to_matrix(self.opt_tag_pose)


class DetectionPair():
    def __init__(self, detection_one: TagDetection, detection_two: TagDetection):
        self.detection_one = detection_one
        self.detection_two = detection_two

    @property
    def relative_tag_transform(self):
        return self.detection_two.tag_pose_global@np.linalg.inv(self.detection_one.tag_pose_global)

    @property
    def opt_tag_one_on_tag_one(self):
        return self.detection_one.opt_tag_pose_homogeneous@self.detection_one.tag_pose_global@np.linalg.inv(self.detection_one.opt_tag_pose_homogeneous)

    @property
    def opt_tag_two_on_tag_one(self):
        return self.detection_two.opt_tag_pose_homogeneous@self.detection_one.tag_pose_global@np.linalg.inv(self.detection_one.opt_tag_pose_homogeneous)

    @property
    def relative_opt_tag_transform(self):
        return self.opt_tag_two_on_tag_one@np.linalg.inv(self.opt_tag_one_on_tag_one)

    @property
    def opt_tag_shift(self):
        return self.relative_tag_transform@np.linalg.inv(self.relative_opt_tag_transform)

    @property
    def opt_tag_shift_pos(self):
        return np.abs(self.opt_tag_shift[0:3,3])

def find_maps_and_data():
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    cms = CacheManagerSingleton(
        firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
    )

    # Assumes only one map exists in the .cache folder for the optimization and trajectory
    optimization_map_info = cms.find_maps(OPTIMIZATION_MAP_NAME, search_restriction=2).pop()
    trajectory_map_info = cms.find_maps(TRAJECTORY_MAP_NAME, search_restriction=2).pop()

    # Both maps should have the same ground truth
    gt_data = cms.find_ground_truth_data_from_map_info(optimization_map_info)

    return optimization_map_info, trajectory_map_info, gt_data

def run_optimization(optimization_map_info, gt_data):
    compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3) * 1.0,
        tag_var=1.0,
        ang_vel_var=1.0
    )

    oconfig = OConfig(
        is_sba=SBA,
        weights=WEIGHTS,
        compute_inf_params=compute_inf_params
    )
    opt_result = holistic_optimize(
        map_info=optimization_map_info,
        pso=PrescalingOptEnum(0 if SBA else 1),
        oconfig=oconfig,
        verbose=True,
        visualize=False,
        compare=False,
        upload=False,
        gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data)
    )
    return opt_result

def calculate_shift_metric(opt_result_opt_tags, trajectory_map_dct):
    trajectory_tag_data = trajectory_map_dct["tag_data"]
    tag_pairs = []

    opt_tag_pos_dct = {}
    for tag in opt_result_opt_tags:
        opt_tag_pos_dct[int(tag[-1])] = tag[:-1]

    shift_metric = np.zeros([4])

    for i in range(len(trajectory_tag_data)-1):
        if trajectory_tag_data[i][0]["tag_id"] == trajectory_tag_data[i+1][0]["tag_id"]:
            continue
        detection_one = TagDetection()
        detection_one.tag_id = trajectory_tag_data[i][0]["tag_id"]
        detection_one.tag_pose = trajectory_tag_data[i][0]["tag_pose"]
        detection_one.odom_id = trajectory_tag_data[i][0]["pose_id"]
        detection_one.odom_pose = trajectory_map_dct["pose_data"][detection_one.odom_id]["pose"]
        detection_one.opt_tag_pose = opt_tag_pos_dct[detection_one.tag_id]

        detection_two = TagDetection()
        detection_two.tag_id = trajectory_tag_data[i+1][0]["tag_id"]
        detection_two.tag_pose = trajectory_tag_data[i+1][0]["tag_pose"]
        detection_two.odom_id = trajectory_tag_data[i+1][0]["pose_id"]
        detection_two.odom_pose = trajectory_map_dct["pose_data"][detection_two.odom_id]["pose"]
        detection_two.opt_tag_pose = opt_tag_pos_dct[detection_two.tag_id]

        curr_tag_pair = DetectionPair(
            detection_one=detection_one,
            detection_two=detection_two
        )
        tag_pairs.append(curr_tag_pair)

        shift_metric[:3] += curr_tag_pair.opt_tag_shift_pos
        shift_metric[3] += np.linalg.norm(curr_tag_pair.opt_tag_shift_pos)

    return shift_metric/len(tag_pairs)

def run_shift_metric_test():
    optimization_map_info, trajectory_map_info, gt_data = find_maps_and_data()
    opt_result = run_optimization(optimization_map_info, gt_data)
    # shift_metric = calculate_shift_metric(opt_result.map_opt.tags, trajectory_map_info.map_dct)
    shift_metric = opt_result.shift_metric
    print("----------------------------------")
    print(f"X Shift: {shift_metric[0]:.3f}")
    print(f"Y Shift: {shift_metric[1]:.3f}")
    print(f"Z Shift: {shift_metric[2]:.3f}")
    print(f"Overall Shift: {shift_metric[3]:.3f}")


if __name__ == "__main__":
    run_shift_metric_test()