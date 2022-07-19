"""
Evaluating the effectiveness of the SBA algorithm by manually calculating it.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import pdb

TAG_SIZE = 0.152
MATRIX_SIZE_CONVERTER = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])
CAMERA_POSE_FLIPPER = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])

ERROR_THRESHOLD = 20
SHOW_INDIVIDUAL_COORDS = False
VISUALIZE = False



def compute_camera_intrinsics(tag_idx, unprocessed_map_data):
    """
    Camera intrisics allow us to convert 3D information into a 2D space
    by using the pinhole camera model: https://en.wikipedia.org/wiki/Pinhole_camera_model
    It's confusing, but it's essentially a matrix we multiply onto everything.

    Args:
        tag_idx (int): the tag to investigate

    Returns:
        camera_intrinsics: 3x3 matrix expressing the camera intrinsics.
    """
    [fx,fy,Cx,Cy] = unprocessed_map_data["tag_data"][tag_idx][0]["camera_intrinsics"]
    camera_intrinsics = np.array([[fx,0,Cx],[0,fy,Cy],[0,0,1]])
    
    return camera_intrinsics

def compute_camera_pose(camera_pose_id, unprocessed_map_data):
    poses = unprocessed_map_data["pose_data"][camera_pose_id+1]["pose"]
    
    return np.reshape(poses, [4, 4], order='F')

def compute_tag_pose(tag_idx, unprocessed_map_data):
    """
    Returns the tag_pose at 4x4 matrix instead of a list of 16 numbers

    Args:
        tag_idx (int): the tag to investigate

    Returns:
        pose: 4x4 pose matrix
    """
    poses = unprocessed_map_data["tag_data"][tag_idx][0]["tag_pose"]
    pose = np.reshape(poses, [4, 4])

    return pose

def compute_corner_pixels():
    """
    Return the location of the corner pixels of a tag, as compared to the
    origin of the tag. Note that the third value is always 0 because we just
    assume that the corners are directly in line with the 

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

def visualizing_difference(calculated_pixels, observed_pixels, tag_id):
    
    calculated_pixels = np.matrix.transpose(calculated_pixels)
    observed_pixels = np.matrix.transpose(observed_pixels)

    plt.axis('equal') 
    
    plt.scatter(calculated_pixels[:, 0],calculated_pixels[:, 1], color = "r")
    plt.scatter(observed_pixels[:, 0],observed_pixels[:, 1], color = "g")
    plt.legend(["calculated","observed"])
    plt.title(tag_id)
    plt.show()

def sba_error_metric(calculated_pixels, observed_pixels):
    """
    Not certain what the best error metric is for this sort of thing. 
    We decided to take the root mean squared value of the distances
    between each corner

    Args:
        calculated_pixels (2x4 matrix): pixel locations calculated by this script
        observed_pixels (2x4 matrix): pixel locations from measurement

    Returns:
        sba_error: A float corresponding to the RMS error between corners
    """
    
    throw = False
    distance_between_points = np.linalg.norm(calculated_pixels - observed_pixels, axis = 0)
    sba_error = np.sqrt(np.square(distance_between_points)).mean()
    
    if sba_error >= ERROR_THRESHOLD:
        throw = True

    return sba_error, throw

def run(tag_idx, unprocessed_map_data):  
    """
    Run the SBA evaluator. 
    """  
    
    camera_intrinsics = compute_camera_intrinsics(tag_idx,unprocessed_map_data)
    observed_pixels = np.reshape(unprocessed_map_data["tag_data"][tag_idx][0]["tag_corners_pixel_coordinates"],[2,4],order = 'F')
    tag_pose = compute_tag_pose(tag_idx,unprocessed_map_data)
    corner_pixel_poses = compute_corner_pixels()
    
    # This equation, whiteboarded out, to convert from the tag frame's corner pixels to the
    # corner pixels we see on the phone. 
    sba_pixel_corners = camera_intrinsics@MATRIX_SIZE_CONVERTER@tag_pose@corner_pixel_poses
 
    for i in range(4):
        for j in range(3):
            sba_pixel_corners[j, i] = sba_pixel_corners[j, i]/sba_pixel_corners[2,i]
            
    # pdb.set_trace()
    sba_pixel_corners = sba_pixel_corners[0:-1] 
    
    if SHOW_INDIVIDUAL_COORDS:
        
        print("bottom left:")
        print(sba_pixel_corners[:2,0])
        
        print("bottom right:")
        print(sba_pixel_corners[:2,1])
        
        print("top right:")
        
        print(sba_pixel_corners[:2,2])
        
        print("top left:")
        print(sba_pixel_corners[:2,3])
        
        print("\n")
    
    relevant_tag = unprocessed_map_data["tag_data"][tag_idx][0]["tag_id"]   
    RMS_error, throw = sba_error_metric(sba_pixel_corners, observed_pixels)
    # print(f"tag {relevant_tag} RMS error: {RMS_error}")
    
    if VISUALIZE and not throw: 
        visualizing_difference(sba_pixel_corners, observed_pixels, relevant_tag)
    
    return RMS_error, throw, relevant_tag

def throw_out_bad_tags(data_path):
    with open(data_path) as data_file:
        unprocessed_map_data = json.load(data_file)
    
    throws = []
    throws_indeces = []
    errors = []
    
    for i in range(len(unprocessed_map_data["tag_data"])):
        sba_rms_error, throw, relevant_tag = run(i, unprocessed_map_data)
        
        if throw: 
            throws.append(relevant_tag)
            throws_indeces.append(i)
            continue
        
        errors.append(sba_rms_error)

    unprocessed_map_data["tag_data"] = [i for j, i in enumerate(unprocessed_map_data["tag_data"]) if j not in throws_indeces]
        
    with open(data_path, "w") as f:
        json.dump(unprocessed_map_data, f, indent = 2)

    percent_thrown = 100 * (len(throws)/len(errors))
    
    return (f"average error: {np.mean(errors)}" + "\n" + 
            f"list of tags thrown: {throws}" + "\n" +
            f"percentage of tags thrown: {percent_thrown:.2f}%")

   
    # print(f"least error: {min(errors)} at tag {tag_id_of_least_error}")
    # print(f"most error: {max(errors)} at tag {tag_id_of_most_error}")
    
if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    print(throw_out_bad_tags("../error_analysis/datasets/floor_2_obleft.json"))
    
