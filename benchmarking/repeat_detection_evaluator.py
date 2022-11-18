"""
This file takes a path of an unprocessed map, maps out the repeated detections of a tag,
then calculates the pixel difference of the corners if those two tags were projected onto
the first detection of the tag. 

The purpose of this test to identify the effect odometry drift/tag detection error has on
errors that we see in the invisible map algorithm. If there was no odometry drift and no
tag detection error, we would see no difference in the corners of two different detections
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import json
import numpy as np
import pandas as pd
import benchmarking.benchmarking_utils as B
import map_processing.throw_out_bad_tags as tag_filter
from scipy.spatial.transform import Rotation as R
import pdb
import argparse

VISUALIZE = False
FIX_IT = False

def make_parser():
    """
    Creates a ArgumentParser object for CLI.
    """
    p = argparse.ArgumentParser(
        description="Visualize and analyze error from oblique/straight tag observations")
    
    p.add_argument(
        "-n", help="name of the test in the config file")
    
    p.add_argument(
        "-t", help="throw out bad tags", action = "store_true")
    
    p.add_argument(
        "-v", help="visualize data", action = "store_true")
    
    p.add_argument(
        "-i", help="print result data", action = "store_true")
    

    return p


def calc_ratio(variances) -> list:
    """
    Compute the importance matrix ratios by average all of the variances.

    Args:
        variances (dataframe): dataframe with variance qx, qy, qz, x, y, z, id

    Returns:
        list: a list of two values [angle variance, position variance]
    """
    transposed = variances.transpose()
    
    ratios = []
    for col in transposed.columns:
        rot_avg = (transposed[col]["qx"] + transposed[col]["qy"] + transposed[col]["qz"]) /3
        trans_avg = (transposed[col]["x"] + transposed[col]["y"] + transposed[col]["z"]) /3
        
        ratios.append(rot_avg/trans_avg)
        
    print(ratios)
        
    
def calc_variance(quats_and_trans):
    variances = []
    [variances.append(np.var(row)) for row in quats_and_trans]
    return variances

def create_quat_from_simd_4x4(simd_4x4_matrix):
    """
    create the corresponding se3quat from a SIMD 4x4 matrix

    Args:
        simd_4x4_matrix (_type_): _description_
    
    Ret:
        quat: quaternion
    """
    
    simd_as_R = R.from_matrix(simd_4x4_matrix[:3, :3])
    quat = simd_as_R.as_quat()

    t = np.transpose([simd_4x4_matrix[:, 3]])
    
    return quat[:3], t

def calculate_tag_detection_variance(tag_idxs, first_observation, next_observations):
    """
    Calculate the variance of tag detections.
    - We collect the consecutive observations of a tag

    Args:
        tag_idxs (_type_): _description_
        first_observation (_type_): _description_
        next_observations (_type_): _description_
    """
    # pdb.set_trace()
    all_observations = [first_observation]
    [all_observations.append(observation) for observation in next_observations]
    prev_idx = tag_idxs[0]
    cons_obs_l = [] # list of sets of consecutive observations
    cons_obs = []
    
    # iterate through indexes, group by consecutive observations.
    for ls_idx, tag_idx in enumerate(tag_idxs):
        
        if tag_idx != prev_idx + 1:
            cons_obs_l.append(cons_obs)
            cons_obs = []
            
        cons_obs.append(ls_idx)
        prev_idx = tag_idx
    
    cons_obs_l.append(cons_obs)
    cons_obs_l.pop(0)
    
    qx = []
    qy = []
    qz = []
    x = []
    y = []
    z = []
    
    variance = []
    num_obs = []
    for cons_observations in cons_obs_l:
        # if there's only one observation, no variance can be calculated.
        if len(cons_observations) == 1:
            continue
        
        # for each observation in the consecutive observations of a tag
        for obs in cons_observations:
            obs_pose = all_observations[obs]
            # pdb.set_trace()
            quat, t = create_quat_from_simd_4x4(obs_pose)
            qx.append(quat[0])
            qy.append(quat[1])
            qz.append(quat[2])
            x.append(t[0])
            y.append(t[1])
            z.append(t[2])
        
        # Find the variance amongst those consecutive observations
        num_obs.append(len(cons_observations))
        variance.append(calc_variance([qx, qy, qz, x, y , z]))
        
    return variance, num_obs
    
def create_observations_dict(instances, unprocessed_map_data):
    """
    Create a dictionary with all the tools necessary to complete the loop closure overlay evaluator

    Args:
        instances (list): A list of indexes corresponding to the detection of a certain tag.
        unprocessed_map_data (dict): dictionary of unprocessed map data

    Returns:
        dict as follows 

        "id":
        {
            "camera_pose":[
                [

                ],
                [

                ],
            ]
            "tag_pose":[
                [

                ],
                [

                ]
            ]
            "corner_pixels":[
                [

                ],
                [

                ]
            ]
        }
    """
    # print(instances)
    results = {}
    for idx in instances:
        results[idx] = {}
        if idx == instances[0]:
            tag_filter_pixel_corners = B.compute_corner_pixels(
                idx, unprocessed_map_data).tolist()
        # prettified_corner_pixels = [tag_filter_pixel_corners[:2,0],tag_filter_pixel_corners[:2,1],tag_filter_pixel_corners[:2,2],tag_filter_pixel_corners[:2,3]]
        # results[idx]["corner_pixels"] = [pixel_pair.tolist() for pixel_pair in prettified_corner_pixels]
        results[idx]["corner_pixels"] = tag_filter_pixel_corners
        results[idx]["tag_pose"] = (B.compute_tag_pose(
            idx, unprocessed_map_data).tolist())
        results[idx]["camera_pose"] = (
            B.compute_camera_pose(idx, unprocessed_map_data).tolist())

    return results

def overlay_tags(tags_to_overlay, unprocessed_map_data, tag_id, visualize, print_info, all_variances):
    """
    Overlays the first observation of the tag on top of subsequent observations to see
    how much error has been accumulated.7

    Args:
        tags_to_overlay (list)): a list of tags to overlay, where each "tag" is a dictionary of 3 things (corner pixels, camera pose and tag pose) with a key of the index.
        We only use the corner pixels if it's the first detection of a tag, because that's what is overlayed on everything
        unprocessed_map_data (_type_): _description_
    """
    errors = []
    all_pixels = []
    subsequent_detection_poses = []

    tag_idxs = list(tags_to_overlay.keys())
    first_detection = tags_to_overlay[tag_idxs[0]]
    first_detection_pose = np.array(first_detection["tag_pose"])
    first_detection_pixels = np.array(first_detection["corner_pixels"])
    all_pixels.append(first_detection_pixels)
    for idx in tag_idxs[1:]:
        subsequent_detection = tags_to_overlay[idx]
        init_observation_new_coord_frame = B.CAMERA_POSE_FLIPPER@np.linalg.inv(
            subsequent_detection["camera_pose"])@first_detection["camera_pose"]@B.CAMERA_POSE_FLIPPER@first_detection["tag_pose"]
        #init_observation_new_coord_frame = init_observation_new_coord_frame
        subsequent_detection_poses.append(init_observation_new_coord_frame)
        all_pixels.append(B.compute_corner_pixels(
            tag_idxs[1], unprocessed_map_data, tag_pose=init_observation_new_coord_frame))
    
    variances, weighting = calculate_tag_detection_variance(tag_idxs,init_observation_new_coord_frame, subsequent_detection_poses)
    
    for variance in variances:
        variance.append(tag_id)
        # pdb.set_trace()
        all_variances.loc[len(all_variances.index)] =  variance
        
        
    # pdb.set_trace()
    
    # A FEW PRINT STATEMENTS FOR METRICS
    if print_info:
        print("\n")
        print(f"TAG_ID: {tag_id}")
        print(f"NUMBER OF TIMES OBSERVED: {len(tag_idxs)} at {tag_idxs}")
        print("\nRMS ERROR:")
        for i, set_of_pixels in enumerate(all_pixels[1:]):
            errors.append(B.compute_RMS_error(first_detection_pixels, set_of_pixels)[0])
            print(f"error for observation {tag_idxs[i+1]} is {B.compute_RMS_error(first_detection_pixels, set_of_pixels)[0]} pixels")
            print(
                f"that's roughly {B.compute_RMS_error(first_detection_pose, subsequent_detection_poses[i])[0]} meters")
            
        print(f" \n tag_pose for tag {tag_idxs[i]} is \n {init_observation_new_coord_frame} \n \n tag {tag_idxs[i+1]} is \n {subsequent_detection_poses[0]}")
        print(f" \n tag variances for tag {tag_id} is \n {calculate_tag_detection_variance(tag_idxs,init_observation_new_coord_frame, subsequent_detection_poses)}")

    
   
    if visualize:
        B.visualizing_corner_pixel_differences(all_pixels, tag_id, "LCD")
    
    return errors, all_variances

def create_matching_tags_dict(path, visualize = False, print_info = False):
    """
    Main pipeline of the loop closure workflow

    Args:
        path (_type_): _description_
    """
    errors = []
    with open(path) as data_file:
        unprocessed_map_data = json.load(data_file)

    seen_tags = []
    matching_tags_data = {}
    tags = unprocessed_map_data["tag_data"]
    for i in range(len(tags)):
        # print(seen_tags)
        if tags[i][0]["tag_id"] in seen_tags:
            matched_tags = ([idx for idx, tag_id in enumerate(
                seen_tags) if tag_id == tags[i][0]["tag_id"]])
            matched_tags.append(i)

            # Append the occurences to the tag in question.
            matching_tags_data[tags[i][0]["tag_id"]] = (
                create_observations_dict(matched_tags, unprocessed_map_data))

        seen_tags.append(tags[i][0]["tag_id"])
    # To check and make sure data looks good
    with open("loop_closure_comparison.json", "w") as write_file:
        json.dump(matching_tags_data, write_file, indent=4, sort_keys=True, )
        
    all_variances = pd.DataFrame(columns = ["qx","qy","qz","x","y","z", "id"])
    for key in matching_tags_data:
        # pdb.set_trace()
        # print(all_variances)
        error, all_variances = overlay_tags(matching_tags_data[key], unprocessed_map_data, key, visualize, print_info, all_variances)
        errors.extend(error)
        
    calc_ratio(all_variances)
    all_variances.to_csv(f"{NAME_OF_TEST}_variance.csv")
        
    if print_info:
        print(f"mean error is {np.array(errors).mean()}")
        

    return matching_tags_data

if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    
    parser = make_parser()
    args = parser.parse_args()
    
    NAME_OF_TEST = args.n.lower()
    
    with open("benchmarking_config.json") as config_file:
        config = json.load(config_file)["overlay_tags"][NAME_OF_TEST]
        print(f"your configuration: {config}")
    
    if args.t:
        tag_filter.throw_out_bad_tags(config["UNPROCESSED_PATH"])

    VISUALIZE = False
    if args.v:
        VISUALIZE = True
    
    PRINT_INFO = False
    if args.i:
        PRINT_INFO = True
    create_matching_tags_dict(config["UNPROCESSED_PATH"], visualize = VISUALIZE, print_info= PRINT_INFO)
