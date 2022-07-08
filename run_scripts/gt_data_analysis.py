"""
Script to clean and manipulate the tag data that comes from RTABmap to see if it's functional
as ground truth data.

NOTE: there are a few terms used interchangeably in this code

gt = RTABmap tag locations --> whenever "gt" is used in this code, it's referencing the tag data we received
from RTABmap

im = invisible map tag locations --> whenever "im" is used in this code, it's reference the tag data we
received from invisible maps. Comes from the "processed_graph" data file.
"""

import re
import argparse
import json
import os
import numpy as np
import pyrr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R


def make_parser():
    """
    Creates a ArgumentParser object.
    """
    p = argparse.ArgumentParser(description="Process RTABmap ground truth data")

    p.add_argument(
        "-n", help = "name of the test you'd like to run in the configuration file")
    
    p.add_argument(
        "-v", help = "triggers the visualizer", action = "store_true")
    
    p.add_argument(
        "-o", help = "triggers the visualization of orientation", action = "store_true"
    )
    
    return p

def check_quaternion_data(new_data_entry):
    """
    The file containing all of the checks to ensure the processed quaternion data is useable

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
    The data we get from RTABmap is in a right-handed coordinate system, as opposed to the
    left-handed coordinate system that IM asks for. This function fixes both of those problems by negating the X axis + turning every number into a float.

    NOTE that RTABmap pose data appears to give z as the gravity axis, but IM uses y as the gravity axis, hence why we computed an "alignment quaternion" to
    align the coordinate systems. 

    Args:
        pose (list): the 7 numbers composing the pose of an object [x,y,z,qx,qy,qz,qw]
    """
    
    # x, y, z, w, quaternion dictating initial orientation of tag
    rtabmap_quat = np.array([pose[3],pose[4],pose[5],pose[6]])
    
    # Matching the April Tag's coordinate frame with RTABmap
    initial_alignment_quat = np.array([-0.5,0.5,0.5,0.5])
    
    # Quaternion to match RTABmap's coordinate frame with IM
    matching_alignment_quat = np.array([-0.5,0.5,0.5,0.5])
    final_quat = pyrr.quaternion.cross(pyrr.quaternion.cross(rtabmap_quat, matching_alignment_quat),initial_alignment_quat)
    
    # conver the pose using the alignment quaternion
    rtabmap_pose = np.array([pose[0],pose[1],pose[2]])
    final_pose = pyrr.quaternion.apply_to_vector(initial_alignment_quat, rtabmap_pose)
    
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
    IM_processed_quats = []
    with open(file_path,"r") as json_file:
        data = json.load(json_file)
        for item in data["tag_vertices"]:
            IM_processed_poses.append([item["translation"]["x"],item["translation"]["y"],item["translation"]["z"]])
            if VISUALIZE == 2:
                IM_processed_quats.append([item["rotation"]["x"],item["rotation"]["y"],item["rotation"]["z"],item["rotation"]["w"]])           
            
    # Ground Truth Data
    GT_processed_poses = []
    GT_processed_quats = []
    for pos in tag_poses:
        GT_processed_poses.append(pos["pose"][0:3])
        GT_processed_quats.append(pos["pose"][3:])
        
        
    return np.array(IM_processed_poses),np.array(GT_processed_poses),np.array(IM_processed_quats), np.array(GT_processed_quats)

def plot_IM_GT_data(im_pos,gt_pos,im_quat,gt_quat):
    """
    Visualize the positions of the IM tag data and the RTABmap tag data.

    Args:
        im (list): a list of tag poses from invisible map
        gt (list): a list of ground truth tag poses from rtabmap
    """
    fig = plt.figure()
    ax = plt.axes(projection = "3d")
    ax.set_xlim(-1,10)
    ax.set_xlabel("x")
    ax.set_ylim(-1,10)
    ax.set_ylabel("y")
    ax.set_zlim(-1,10)
    ax.set_zlabel("z")
    ax.view_init(120,-90)
    
    # TODO: have to reorder points to add labelling
    
    # length_of_im_data = np.shape(im)
    # length_of_gt_data = np.shape(gt)
    
    # for i in range(length_of_im_data[0]):
    #     plt.plot(im[i,0],im[i,1],im[i,2], "o", c ="red")
    #     ax.text(im[i,0],im[i,1],im[i,2], i)
    # for i in range(length_of_gt_data[0])s:
    #     plt.plot(gt[i,0],gt[i,1],gt[i,2], "o", c ="green")
    #     ax.text(gt[i,0],gt[i,1],gt[i,2], i)
    
    plt.plot(im_pos[:,0],im_pos[:,1],im_pos[:,2], "-o", c ="black")
    plt.plot(gt_pos[:,0],gt_pos[:,1],gt_pos[:,2], "-o", c ="orange")
    
    
    # If there were orientations computed
    if len(im_quat > 0) and len(gt_quat>0):
        # show orientation
        for i in range(len(im_quat)):
            im_coordinate_frame = R.from_quat(im_quat[i]).as_matrix()
            ax.quiver(im_pos[i,0],im_pos[i,1],im_pos[i,2],im_coordinate_frame[0][0],im_coordinate_frame[1][0],im_coordinate_frame[2][0], color = "r")
            ax.quiver(im_pos[i,0],im_pos[i,1],im_pos[i,2],im_coordinate_frame[0][1],im_coordinate_frame[1][1],im_coordinate_frame[2][1], color = "g")
            ax.quiver(im_pos[i,0],im_pos[i,1],im_pos[i,2],im_coordinate_frame[0][2],im_coordinate_frame[1][2],im_coordinate_frame[2][2], color = "b")
        for i in range(len(gt_quat)):
            gt_coordinate_frame = R.from_quat(gt_quat[i]).as_matrix()
            ax.quiver(gt_pos[i,0],gt_pos[i,1],gt_pos[i,2],gt_coordinate_frame[0][0],gt_coordinate_frame[1][0],gt_coordinate_frame[2][0], color = "r")
            ax.quiver(gt_pos[i,0],gt_pos[i,1],gt_pos[i,2],gt_coordinate_frame[0][1],gt_coordinate_frame[1][1],gt_coordinate_frame[2][1], color = "g")
            ax.quiver(gt_pos[i,0],gt_pos[i,1],gt_pos[i,2],gt_coordinate_frame[0][2],gt_coordinate_frame[1][2],gt_coordinate_frame[2][2], color = "b")
   
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
            
            new_data_entry = [entry for entry in re.split("=|,|\n|\|| |xyz|rpy| ", line) if entry != '']
            # print(new_data_entry)

            # XYZ and quaternion data, detecable because the first element of the list is "MARKER"
            if new_data_entry[0] == "MARKER":
                
                # Only append if the data is good
                if check_quaternion_data(new_data_entry):
                    
                    pose = generate_correct_pose([float(number) for number in new_data_entry[2:]])
                    
                    tag_poses.append(
                        {"tag_id": int(new_data_entry[1]), "pose": pose})
                else:
                    print(f"This data is incorrectly formatted: {new_data_entry}")
            
            # XYZ and RPY data, detecable because the markers are negative + no text
            elif new_data_entry[0][0] == "-":
                # print("translation and roll pitch yaw")
                new_data_entry = [float(entry) for entry in new_data_entry]
                xyz = new_data_entry[1:4]
                rpy = new_data_entry[4:]
                quat = list(pyrr.quaternion.create_from_eulers(rpy))
                pose = generate_correct_pose(xyz+quat)
                tag_poses.append({"tag_id": -int(new_data_entry[0]), "pose": pose})
                
            else:
                print(f"I don't have any way to handle {new_data_entry}")
                
    # dump it to a json file
    with open(outputted_json_path, "w") as f:
        tag_poses = sorted(tag_poses, key=lambda x: x['tag_id'])
        data = {
            "poses":
                tag_poses
        }
        json.dump(data, f, indent=2)
        
    # Visualize anchor positions 
    if visualize > 0 and processed_graph_path != "":
        im_pos,gt_pos,im_quat,gt_quat = process_IM_GT_data(processed_graph_path,tag_poses)
        plot_IM_GT_data(im_pos,gt_pos,im_quat,gt_quat)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Take in the name of the test for configuration purposes
    NAME_OF_TEST = args.n

    with open("../rtabmap/gt_analysis_config.json") as config_file:
        config = json.load(config_file)[NAME_OF_TEST]
        print(f"your configuration: {config}")
    
    # By default, visualization is 0. The addition of tags will change the value of visualize.
    VISUALIZE = 0
    if args.v:
        # Visualize
        VISUALIZE = 1
        if args.o:
            # Visualize orientations too
            VISUALIZE = 2
        
    RTABMAP_DATA_PATH = config["RTABMAP_DATA_PATH"]
    OUTPUTTED_JSON_PATH = config["OUTPUTTED_JSON_PATH"]
    PROCESSED_GRAPH_DATA_PATH = config["PROCESSED_GRAPH_DATA_PATH"]

    run(RTABMAP_DATA_PATH,OUTPUTTED_JSON_PATH,PROCESSED_GRAPH_DATA_PATH,VISUALIZE)