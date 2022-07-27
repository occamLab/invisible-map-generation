"""
Utilities used while benchmarking Invisible Maps. 
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R

import benchmarking.repeat_detection_evaluator as rde

TAG_SIZE = 0.152
MATRIX_SIZE_CONVERTER = np.array(
    [
        [1, 0, 0, 0], 
        [0, 1, 0, 0], 
        [0, 0, 1, 0]
        ]
)
CAMERA_POSE_FLIPPER = np.array(
    [
        [1, 0, 0, 0], 
        [0, -1, 0, 0], 
        [0, 0, -1, 0], 
        [0, 0, 0, 1]
    ]
)
ERROR_THRESHOLD = 20

SHOW_INDIVIDUAL_COORDS = False
VISUALIZE = False
FIX_IT = False


# error_analysis_quat.py

def rotational_difference_calculation (mat1, mat2):
    """
    Calculates the rpy difference between the first and second rotation matrix. 

    Args:
        mat1 (list): a list of 16 numbers describing the rotation matrix referring to the non-LIDAR observation (either oblique or straight on)
        mat1 (list): a list of 16 numbers describing the rotation matrix referrring to the LIDAR observations (treated as truth)
    
    Returns:
        net_rpy (list): list of roll, pitch, yaw of the angle
    """
    
    # data collected with/without LIDAR
    without_LIDAR_array= np.array(mat1).reshape((4,4))
    LIDAR_array = np.array(mat2).reshape((4,4))

    net_rotation = (np.linalg.inv(LIDAR_array)@without_LIDAR_array)[0:3,0:3]
    # print(net_rotation)
    net_quat = R.from_matrix(net_rotation).as_quat()
    
    # [value == value/net_quat[3] for value in net_quat]
    
    for value in net_quat:
        value = value/net_quat[3]

    # data collected with/without LIDAR
    t_x = without_LIDAR_array[0,3] - LIDAR_array[0,3]
    t_y = without_LIDAR_array[1,3] - LIDAR_array[1,3]
    t_z = without_LIDAR_array[2,3] - LIDAR_array[2,3]
    
    net_translation = (t_x, t_y, t_z)
    # print(net_translation)
    
    return net_quat[0:3], net_translation

def data_information (data, names = None):
    """
    Give basic information about a dataset -- median, mean, range, std, etc.

    Args:
        qxyzs(list): a list of lists containing data you'd like information about
    """
    
    if names == None:
        names = ["oblique-rolls","oblique-pitches","oblique-yaws",
                "straight-rolls","straight-pitches","straight-yaws"]
    
    for i, dataset in enumerate(data):
        # print(dataset)
        print(names[i])
        print(f"Median: {np.median(dataset)}")
        print(f"Mean: {np.mean(dataset)}")
        print(f"Range: {max(dataset)-min(dataset)}")
        print(f"Standard Deviation: {np.std(dataset)}")
        print("\n")

def compute_variances(all_pose_data):
    """
    compute the variances of all of the individual values in all of the pose data (rotationally + translationally)
    
    Args:
        all_pose_data (list): a list of qx,qy,qz,x,y,zs
        
    """
    
    variances = []
    for list in all_pose_data:
        variances.append(np.var(list))
        
    return variances

def visualize_data_spread(qxyzs, xyzs, test_name):
    """
    Visualize both datasets

    Args:
        o_qx (list): list of rolls from oblique observations
        o_qy (list): list of pitches from oblique observations
        o_qz (list): list of yaws from oblique observations
        s_qx (list): list of rolls from straight observations
        s_qy (list): list of pitches from straight observations
        s_qz (list): list of yaws from straight observations
        test_name (str): name of the test (for titling/saving purposes)

    Returns:
        True: when complete!
    """
    all_pose_data = (qxyzs + xyzs)
    titles = ["oblique-qx","oblique-qy","oblique-qz","straight-qx","straight-qy","straight-qz",
              "o_x","o_y","o_z","s_x","s_y","s_z"]
    colors = ["red","green","blue"]
    plt.figure(figsize= (15,10))
    plt.suptitle(test_name, fontsize = 15)
    
    for i in range(12):
        plt.subplot(4,3,i+1, title = titles[i])
        plt.hist(all_pose_data[i], bins = 25, color = colors[i%3])
        
    plt.show()

# sba_evaluator.py

def compute_camera_intrinsics(tag_idx, unprocessed_map_data):
    """
    Camera intrisics allow us to convert 3D information into a 2D space
    by using the pinhole camera model: https://en.wikipedia.org/wiki/Pinhole_camera_model
    It's confusing, but it's essentially a matrix we multiply onto everything.

    Args:
        tag_idx (int): the index of the tag to investigate
        unprocessed_map_data (dict): a dictionary containing the unprocessed map data

    Returns:
        camera_intrinsics: 3x3 matrix expressing the camera intrinsics.
    """
    [fx,fy,Cx,Cy] = unprocessed_map_data["tag_data"][int(tag_idx)][0]["camera_intrinsics"]
    camera_intrinsics = np.array([[fx,0,Cx],[0,fy,Cy],[0,0,1]])
    
    return camera_intrinsics

def compute_tag_pose(tag_idx, unprocessed_map_data):
    """
    Returns the tag_pose at 4x4 matrix instead of a list of 16 numbers

    Args:
        tag_idx (int): the index of the tag to investigate

    Returns:
        pose: 4x4 pose matrix
    """
    poses = unprocessed_map_data["tag_data"][tag_idx][0]["tag_pose"]
    pose = np.reshape(poses, [4, 4])

    return pose

def set_corner_pixels_tag_frame():
    """
    Return the location of the corner pixels of a tag, as compared to the
    origin of the tag. Note that the third value is always 0 because we just
    assume that the corners are directly in line with the tag itself. 

    Returns:
        pixels: A 4x4 array where the columns are the coordinates of the corners of a
        tag
    """
    top_left_pixel = np.array([[-TAG_SIZE/2],[TAG_SIZE/2],[0],[1]])
    top_right_pixel = np.array([[TAG_SIZE/2],[TAG_SIZE/2],[0],[1]])
    bottom_left_pixel = np.array([[-TAG_SIZE/2],[-TAG_SIZE/2],[0],[1]])
    bottom_right_pixel = np.array([[TAG_SIZE/2],[-TAG_SIZE/2],[0],[1]])
    pixels = np.hstack((bottom_left_pixel, bottom_right_pixel, top_right_pixel, top_left_pixel))
    return pixels

def visualizing_corner_pixel_differences(pixels, tag_id, visualization_type):
    """
    Visualizing the corner pixels of different tag detections

    Args:
        pixels (list): A list of pixels to graph. 
        tag_id (int/str): the tag being observed (for naming purposes)
        visualization_type (str): a "code" indicating what kind of visualization (for legend)
    """

    plt.axis("equal")
    pixels = [np.matrix.transpose(pixel) for pixel in pixels]

    if visualization_type == "CO":
        # Without LIDAR (calculated with SBA) v.s. LIDAR (observed)
        legend = ["calculated", "observed"]

    if visualization_type == "LCD":
        # loop closure detection -- laying multiple observations of the same
        # tag while recording to see drift
        
        legend = [f"{i+1} observation" if i !=
                  0 else "initial" for i in range(len(pixels))]

    if visualization_type == "OT":
        # overlay tags -- testing optimized tag on top of unoptimized observations
        legend = ["initial observation","optimized"]
        
    # The initial dataset will always be first_detection
    plt.scatter(pixels[0][:, 0], pixels[0][:, 1], color="blue")

    colors = ["orange", "red", "green", "black", "yellow", "cyan",
              "maroon", "chocolate", "rebeccapurple", "indianred"]
    for i in range(len(pixels)-1):
        plt.scatter(pixels[i+1][:, 0], pixels[i+1][:, 1], color=colors[i])

    plt.legend(legend)
    plt.title(tag_id)
    plt.show()
    
def compute_RMS_error(pixels1, pixels2):
    """
    Not certain what the best error metric is for this sort of thing. 
    We decided to take the root mean squared value of the distances
    between each corner

    Args:
        calculated_pixels (2x4 matrix): pixel locations calculated by this script
        observed_pixels (2x4 matrix): pixel locations from measurement

    Returns:
        RMS_error: A float corresponding to the RMS error between corners
        throw (bool): True if the tag surpasses the error threshhold, false if not.
    """
    
    throw = False
    distance_between_points = np.linalg.norm(pixels1 - pixels2, axis = 0)
    RMS_error = np.sqrt(np.square(distance_between_points)).mean()
    
    if RMS_error >= ERROR_THRESHOLD:
        throw = True
    
    # RMS_error = distance_between_points

    return RMS_error, throw


# loop_closure_evaluator.py
def compute_camera_pose(tag_idx, unprocessed_map_data):
    camera_pose_id = unprocessed_map_data["tag_data"][tag_idx][0]["pose_id"]
    poses = unprocessed_map_data["pose_data"][camera_pose_id]["pose"]

    return np.reshape(poses, [4, 4], order='F')

def compute_corner_pixels(tag_idx, unprocessed_map_data, tag_pose=None):
    camera_intrinsics = compute_camera_intrinsics(
        tag_idx, unprocessed_map_data)

    # If a new tag_pose is not added as an argument, then compute it.
    if tag_pose is None:
        tag_pose = compute_tag_pose(tag_idx, unprocessed_map_data)

    corner_pixel_poses = set_corner_pixels_tag_frame()

    # This equation, whiteboarded out, to convert from the tag frame's corner pixels to the
    # corner pixels we see on the phone.
    sba_pixel_corners = camera_intrinsics@MATRIX_SIZE_CONVERTER@tag_pose@corner_pixel_poses

    for i in range(4):
        for j in range(3):
            sba_pixel_corners[j, i] = sba_pixel_corners[j,
                                                        i]/sba_pixel_corners[2, i]

    # pdb.set_trace()
    sba_pixel_corners = sba_pixel_corners[0:-1]

    if SHOW_INDIVIDUAL_COORDS:

        print("bottom left:")
        print(sba_pixel_corners[:2, 0])

        print("bottom right:")
        print(sba_pixel_corners[:2, 1])

        print("top right:")

        print(sba_pixel_corners[:2, 2])

        print("top left:")
        print(sba_pixel_corners[:2, 3])

        print("\n")

    return sba_pixel_corners

# overlay_tags.py
def create_simd_4x4_from_se3_quat(translation, quaternion):
    """
    Create the corresponding SIMD 4x4 matrix from an se3quat. 

    Args:
        translation (array): x,y,z
        quaternion (array): qx, qy, qz, qw

    Returns:
        _type_: _description_
    """
    
    final_simd_4x4 = np.zeros((4,4))
    
    quat_as_R = R.from_quat(np.array(quaternion))
    r_matrix = quat_as_R.as_matrix()
    
    final_simd_4x4[0:3,0:3] = r_matrix
    final_simd_4x4[0:3,3] = np.transpose(translation)
    final_simd_4x4[3,3] = 1
    
    return final_simd_4x4

def create_dict_of_observations_and_poses(up_path):
    
    with open(up_path,"r") as data_dict:
        up_map_data = json.load(data_dict)
        
    all_tags_observations = rde.create_matching_tags_dict(up_path)
    
    for i in range(len(up_map_data["tag_data"])):
        if up_map_data["tag_data"][i][0]["tag_id"] not in all_tags_observations:
            all_tags_observations[up_map_data["tag_data"][i][0]["tag_id"]]=(rde.create_observations_dict([i],up_map_data))
    
    return all_tags_observations

if __name__ == "__main__":
    
    # TRUE
    print("blame ayush if it doesn't work")