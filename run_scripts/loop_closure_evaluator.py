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
FIX_IT = False

from sba_evaluator import sba_error_metric, throw_out_bad_tags


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

def compute_camera_pose(tag_idx, unprocessed_map_data):
    camera_pose_id = unprocessed_map_data["tag_data"][tag_idx][0]["pose_id"]
    poses = unprocessed_map_data["pose_data"][camera_pose_id]["pose"]
    
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

def set_corner_pixels_tag_frame():
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

def compute_corner_pixels_no_tag_pose(tag_idx, unprocessed_map_data):
        
    camera_intrinsics = compute_camera_intrinsics(tag_idx,unprocessed_map_data)
    tag_pose = compute_tag_pose(tag_idx,unprocessed_map_data)
    corner_pixel_poses = set_corner_pixels_tag_frame()
    
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
    
    
    return sba_pixel_corners
  
def compute_corner_pixels(tag_idx, unprocessed_map_data, tag_pose):
        
    camera_intrinsics = compute_camera_intrinsics(tag_idx,unprocessed_map_data)
    #tag_pose = compute_tag_pose(tag_idx,unprocessed_map_data)
    corner_pixel_poses = set_corner_pixels_tag_frame()
    
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
    
    
    return sba_pixel_corners
       
def visualizing_difference(pixels, tag_id, visualization_type):
    """
    Visualize the corner pixels of both the calculated and observed pixels
    """
    
    plt.axis("equal")
    pixels = [np.matrix.transpose(pixel) for pixel in pixels]

    if visualization_type == "CO":
        legend = ["calculated","observed"]
        
    if visualization_type == "LCD":
        # LOOP CLOSURE DETECTION
        legend = [f"{i+1} observation" if i!= 0 else "initial" for i in range(len(pixels))]
    
    # The initial dataset will always be first_detection
    plt.scatter(pixels[0][:, 0],pixels[0][:, 1], color = "blue")
    
    colors = ["orange", "red", "green", "black", "yellow", "cyan", "maroon","chocolate", "rebeccapurple", "indianred"]
    for i in range(len(pixels)-1):
        plt.scatter(pixels[i+1][:, 0],pixels[i+1][:, 1], color = colors[i])
        
    plt.legend(legend)
    plt.title(tag_id)
    plt.show()
    
    
def create_data_dict(instances, unprocessed_map_data):
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
            sba_pixel_corners = compute_corner_pixels_no_tag_pose(idx,unprocessed_map_data).tolist()
        # prettified_corner_pixels = [sba_pixel_corners[:2,0],sba_pixel_corners[:2,1],sba_pixel_corners[:2,2],sba_pixel_corners[:2,3]]
        # results[idx]["corner_pixels"] = [pixel_pair.tolist() for pixel_pair in prettified_corner_pixels]
        results[idx]["corner_pixels"] = sba_pixel_corners
        results[idx]["tag_pose"] = (compute_tag_pose(idx,unprocessed_map_data).tolist())
        results[idx]["camera_pose"] = (compute_camera_pose(idx,unprocessed_map_data).tolist())
        
    return results

def overlay_tags(tags_to_overlay, unprocessed_map_data, tag_id):
    """
    Overlays the first observation of the tag on top of subsequent observations to see
    how much error has been accumulated.

    Args:
        tags_to_overlay (list)): a list of tags to overlay, where each "tag" is a dictionary of 3 things (corner pixels, camera pose and tag pose) with a key of the index.
        We only use the corner pixels if it's the first detection of a tag, because that's what is overlayed on everything
        unprocessed_map_data (_type_): _description_
    """
    
    all_pixels = []
    subsequent_detection_poses = []
    
    tag_idxs = list(tags_to_overlay.keys())
    first_detection = tags_to_overlay[tag_idxs[0]]
    first_detection_pose = np.array(tags_to_overlay[tag_idxs[0]]["tag_pose"])
    first_detection_pixels = np.array(tags_to_overlay[tag_idxs[0]]["corner_pixels"])
    all_pixels.append(first_detection_pixels)
    for idx in tag_idxs[1:]:
        subsequent_detection = tags_to_overlay[idx]
        init_observation_new_coord_frame = CAMERA_POSE_FLIPPER@np.linalg.inv(subsequent_detection["camera_pose"])@first_detection["camera_pose"]@CAMERA_POSE_FLIPPER@first_detection["tag_pose"]
        init_observation_new_coord_frame = init_observation_new_coord_frame
        subsequent_detection_poses.append(init_observation_new_coord_frame)
        all_pixels.append(compute_corner_pixels(tag_idxs[1], unprocessed_map_data, init_observation_new_coord_frame))
    
    
    # A FEW PRINT STATEMENTS FOR METRICS
    print("\n")
    print(f"TAG_ID: {tag_id}")
    print(f"NUMBER OF TIMES OBSERVED: {len(tag_idxs)} at {tag_idxs}")
    print("RMS ERROR:")
    for i, set_of_pixels in enumerate(all_pixels[1:]):
        print(f"error for tag {tag_idxs[i+1]} is {sba_error_metric(first_detection_pixels, set_of_pixels)[0]} pixels")
        print(f"that's roughly {sba_error_metric(first_detection_pose, subsequent_detection_poses[i])[0]} meters")
    
    visualizing_difference(all_pixels, tag_id, "LCD")
    

def loop_closure_drift(path):
    """
    Main pipeline of the loop closure workflow

    Args:
        path (_type_): _description_
    """

    with open(path) as data_file:
        unprocessed_map_data = json.load(data_file)
        
    seen_tags = []
    resultant_data = {}
    tags = unprocessed_map_data["tag_data"]
    for i in range(len(tags)):
        # print(seen_tags)
        if tags[i][0]["tag_id"] in seen_tags:
            matched_tags = ([idx for idx,tag_id in enumerate(seen_tags) if tag_id == tags[i][0]["tag_id"]])
            matched_tags.append(i)
            
            # Append the occurences to the tag in question.
            resultant_data[tags[i][0]["tag_id"]] = (create_data_dict(matched_tags,unprocessed_map_data))
            
        seen_tags.append(tags[i][0]["tag_id"])    
        # print(i)

    # print(seen_tags)  
    # print(format(resultant_data))
    
    with open("loop_closure_comparison.json","w") as write_file:
        json.dump(resultant_data, write_file, indent=4, sort_keys=True, )
    
    for key in resultant_data:
        overlay_tags(resultant_data[key], unprocessed_map_data, key)
    pass
    
    
if __name__ == "__main__":
             
    np.set_printoptions(suppress=True)
    throw_out_bad_tags("../error_analysis/datasets/floor_2_straight.json")
    
    loop_closure_drift("../error_analysis/datasets/floor_2_straight.json")
    
