"""
Kind of similar to the repeat detection script, this script maps the optimized
tag onto the camera
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import json
import pdb
import numpy as np
import repeat_detection_evaluator as rde

import map_processing
from map_processing import benchmarking_utils as B
import pprint

UP_PATH = "datasets/floor_2_obright_cleaned.json"
P_PATH = "datasets/sweeped/floor_2_obright_processed.json"
SHOW_INDIVIDUAL_COORDS = False
VISUALIZE = False


def overlay_tags(up_path,p_path):
    """
    Overlay an optimized tag with other occurences of that tag.

    Args:
        up_path (str): path to the unprocessed(unoptimized) map data
        p_path (str): path to the processed(optimized) map data

    Returns:
        dict: a dictionary containing the poses for the optimized tags
    """
    
    errors = []
    tag_simd_dict = {}
    ids = []
    
    with open(p_path,"r") as data_dict:
        p_map_data = json.load(data_dict)
        
    with open(up_path,"r") as data_dict:
        up_map_data = json.load(data_dict)
    
    tags_to_overlay = B.create_dict_of_observations_and_poses(up_path)
    
    for tag in p_map_data["tag_vertices"]:
        
        try:
            next_tag_id = tags_to_overlay[tag["id"]]
        except KeyError:
            print(f"tag_id: {tag['id']} was not found in the unprocessed map \n")
            continue
        
        tag_trans = np.array(list(tag["translation"].values()))
        tag_quat = list(tag["rotation"].values())
        opt_tag_pose = B.create_simd_4x4_from_se3_quat(tag_trans,tag_quat)
        
        print(f"tag {tag['id']}")
        
        # for every observation of the next tag id
        for observation in next_tag_id:
            
            associated_camera_id = up_map_data["tag_data"][observation][0]["pose_id"]
            opt_cam_trans_component = list(p_map_data["odometry_vertices"][associated_camera_id]["translation"].values())
            opt_cam_rot_component = list(p_map_data["odometry_vertices"][associated_camera_id]["rotation"].values())
            
            # TODO: not sure if this is actually correct
            # correct_quat_order = opt_cam_rot_component[1:] + [opt_cam_rot_component[0]]
            
            opt_cam_pose = B.create_simd_4x4_from_se3_quat(opt_cam_trans_component, opt_cam_rot_component)
            
            opt_tag_in_cam_frame = B.CAMERA_POSE_FLIPPER@np.linalg.inv(opt_cam_pose)@opt_tag_pose
            
            unopt_pixels = B.compute_corner_pixels(observation,up_map_data)
            opt_tag_in_unopt_frame_pixels = B.compute_corner_pixels(observation,up_map_data,opt_tag_in_cam_frame)
            
            if VISUALIZE:
                B.visualizing_corner_pixel_differences([unopt_pixels,opt_tag_in_unopt_frame_pixels], tag["id"], "OT")
            
            # A few print statements to gain insight on the data
            print(f"observation {observation}")
            print(f"we see an error of {B.compute_RMS_error(unopt_pixels,opt_tag_in_unopt_frame_pixels)[0]}")

            # errors.append(B.compute_RMS_error(next_tag_id[observation]["tag_pose"],opt_tag_in_cam_frame))
            errors.append(B.compute_RMS_error(unopt_pixels,opt_tag_in_unopt_frame_pixels)[0])
            
        print("\n")
        ids.append(tag["id"])
        tag_simd_dict[tag["id"]] = opt_tag_pose.tolist()
        
    print(f"average error: {np.array(errors).mean()}")
    print(f"tags seen: {ids}")
    # pprint.pprint(tag_simd_dict)
    return tag_simd_dict
    
if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    overlay_tags(UP_PATH,P_PATH)

 






    
    
    
    
    
