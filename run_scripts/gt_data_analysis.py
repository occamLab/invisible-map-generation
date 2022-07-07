"""
Script to clean and manipulate the tag data that comes from RTABmap to see if it's functional
as ground truth data.

NOTE: there are a few terms used interchangeably in this code

gt = RTABmap tag locations --> whenever "gt" is used in this code, it's referencing the tag data we received
from RTABmap

im = invisible map tag locations --> whenever "im" is used in this code, it's reference the tag data we
received from invisible maps. Comes from the "processed_graph" data file.
"""

import json
import os
from turtle import rt
import numpy as np
import pyrr
import pdb
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

TEST = "mac_2_3"

with open("../rtabmap/gt_analysis_config.json") as jsonf:
    config = json.load(jsonf)[TEST]
    print(config)
    
VISUALIZE = True
RTABMAP_DATA_PATH = config["RTABMAP_DATA_PATH"]
OUTPUTTED_JSON_PATH = config["OUTPUTTED_JSON_PATH"]

# Only used for visualization of tag positions:
PROCESSED_GRAPH_DATA_PATH = config["PROCESSED_GRAPH_DATA_PATH"]


def check_data(new_data_entry):
    """
    The file containing all of the checks to ensure the processed data is useable

    Args:
        new_data_entry (list): A list of all necessary data (ID + Translation + POSE)

    Returns:
        Boolean: True if data is good, False if data is bad.
    """
    if len(new_data_entry) != 9:
        print(f"ERROR in {new_data_entry}")
        return False

    # odometry nodes are saved as markers as well, but with a negative ID.
    if float(new_data_entry[1]) < 0:
        print(f"Negative marker {new_data_entry[1]} detected and thrown")
        return False

    return True

def generate_correct_pose(pose):
    """
    The data we get from RTABmap makes every number a string and is in a right-handed coordinate system, as opposed to the
    left-handed coordinate system that IM asks for. This function fixes both of those problems by negating the X axis + turning every number into a float.

    NOTE that RTABmap pose data appears to give z as the gravity axis, but IM uses y as the gravity axis, hence why we computed an "alignment quaternion" to
    align the coordinate systems. 

    Args:
        pose (list): the 7 numbers composing the pose of an object [x,y,z,qx,qy,qz,qw]
    """

    # convert string to float
    pose = [float(number) for number in pose]
    
    
    # x, y, z, w, quaternion dictating initial orientation of tag
    rtabmap_quat = np.array([pose[3],pose[4],pose[5],pose[6]])
    
    # -90 degrees around y, -90 old x
    initial_alignment_quat = np.array([-0.5,0.5,0.5,0.5])
    
    # 180 degrees around z
    # matching_alignment_quat = np.array([0.5,0.5,0.5,0.5])
    matching_alignment_quat = np.array([0,0,1,0])
          
    final_quat = pyrr.quaternion.cross(pyrr.quaternion.cross(rtabmap_quat, matching_alignment_quat),initial_alignment_quat)
    # final_quat = pyrr.quaternion.cross(rtabmap_quat, initial_alignment_quat)
    # final_quat = [final_quat[1],final_quat[2],final_quat[3],final_quat[0]]
    
    # conver the pose using the alignment quaternion
    rtabmap_pose = np.array([pose[0],pose[1],pose[2]])
    final_pose = pyrr.quaternion.apply_to_vector(initial_alignment_quat, rtabmap_pose)
    
    print(final_pose)
    # create the new se3quat corresponding to the corrected pose
    for i in range(7):
        if i < 3:
            pose[i] = final_pose[i]
        else:
            pose[i] = final_quat[i-3]

    return pose

def process_IM_GT_data(file_path,tag_poses):
    """
    We want to test how the coordinates of our two data files line up. In order to do that, 
    we have to strip off the quaternions from the end of our ground truth data + pull that data from the
    invisible map json. 

    Args:
        file_path (string): the string that corresponds to the json of tag data from invisible maps
        tag_poses (list): the list of tag poses created from RTABmap's data

    Returns:
        2 numpy arrays. The first one is a list of (x,y,z) coordinates for the IM data, and the second one
        is a list of (x,y,z) coordinates for the ground truth data.
    """
    # Invisible Map Data
    IM_processed_poses = []
    with open(file_path,"r") as json_file:
        data = json.load(json_file)
        print(data.keys())
        for item in data["tag_vertices"]:
            IM_processed_poses.append([item["translation"]["x"],item["translation"]["y"],item["translation"]["z"]])
            
            
    # Ground Truth Data
    GT_processed_poses = []
    for pos in tag_poses:
        GT_processed_poses.append(pos["pose"][0:3])
        
    return np.array(IM_processed_poses),np.array(GT_processed_poses)

def plot_IM_GT_data(im,gt):
    """
    Visualize the positions of the IM tag data and the RTABmap tag data.

    Args:
        im (list): a list of tag poses from invisible map
        gt (list): a list of ground truth tag poses from rtabmap
    """
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.set_xlim(-10,10)
    ax.set_xlabel("x")
    ax.set_ylim(-10,10)
    ax.set_ylabel("y")
    ax.set_zlim(-10,10)
    ax.set_zlabel("z")
    ax.view_init(120,-90)
    
    # TODO: have to reorder points to add labelling
    
    # length_of_im_data = np.shape(im)
    # length_of_gt_data = np.shape(gt)
    
    # for i in range(length_of_im_data[0]):
    #     plt.plot(im[i,0],im[i,1],im[i,2], "o", c ="red")
    #     ax.text(im[i,0],im[i,1],im[i,2], i)
    # for i in range(length_of_gt_data[0]):
    #     plt.plot(gt[i,0],gt[i,1],gt[i,2], "o", c ="green")
    #     ax.text(gt[i,0],gt[i,1],gt[i,2], i)
    
    plt.plot(im[:,0],im[:,1],im[:,2], "o", c ="red")
    plt.plot(gt[:,0],gt[:,1],gt[:,2], "o", c ="green")
    
    ax.legend(["invisible map","ground truth"])
    plt.show()

def run(rtabmap_data_path,outputted_json_path, processed_graph_path, visualize):
    """
    Run the app's main pipeline

    Args:
        rtabmap_data_path (string): a string corresponding to the json file of tag data from rtabmap
        outputted_json_path (string): a string corresponding to the place where the gt_data json file should be stored
        processed_graph_path (string): a string corresponding to the place where we get our tag data from invisible maps
        visualize (boolean): choose whether or not to visualize the points.
    """
    # initialize empty list of tag poses
    tag_poses = []
    
    # If data comes in as a g2o file, change to .txt
    if rtabmap_data_path[-4:] == ".g2o":
        base = os.path.splitext(rtabmap_data_path)[0]
        os.rename(rtabmap_data_path, base + ".txt")
        rtabmap_data_path = rtabmap_data_path[:-4] + ".txt"

    # Handling data
    with open(rtabmap_data_path, "r") as f:
        for line in f.readlines():
            new_data_entry = line.split()

            # Right now we only care about handling tags (aka MARKERs)
            if new_data_entry[0] == "MARKER":

                # Only append if the data is good
                if check_data(new_data_entry):
                    pose = generate_correct_pose(new_data_entry[2:])
                    tag_poses.append(
                        {"tag_id": int(new_data_entry[1]), "pose": pose})
                else:
                    print(f"This data is incorrectly formatted: {new_data_entry}")

            else:
                print(f"I don't have any way to handle {new_data_entry}")
                
    # dump it to a json file
    with open(outputted_json_path, "w") as f:
        data = {
            "poses":
                tag_poses
        }

        json.dump(data, f, indent=2)
        
    # Visualize anchor positions 
    if visualize and processed_graph_path != "":
        im,gt = process_IM_GT_data(processed_graph_path,tag_poses)
        plot_IM_GT_data(im,gt)

if __name__ == "__main__":
    run(RTABMAP_DATA_PATH,OUTPUTTED_JSON_PATH,PROCESSED_GRAPH_DATA_PATH,VISUALIZE)