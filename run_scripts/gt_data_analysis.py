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
import pdb
import numpy as np
import pyrr
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def make_parser():
    """
    Creates a ArgumentParser object.
    """
    p = argparse.ArgumentParser(
        description="Process RTABmap ground truth data")

    p.add_argument(
        "-n", help="name of the test you'd like to run in the configuration file")

    p.add_argument(
        "-v", help="triggers the visualizer", action="store_true")

    return p


def interpret_data_type(new_data_entry):
    """
    The file interpretting the data file that comes in and tells the computer to either use roll, pitch, yaw or a quaternion on the next step.

    Although it seems like what this function does is redundant, it's important that any data we get through this pipeline is exactly
    what we think it's going to be. Otherwise we might be debugging for another 5 days only to find out that there was an error in our data that
    caused all of our problems o-o

    Args:
        new_data_entry (list): A line of data

    Returns:
        String: either "rpy" or "quat". If it's "rpy", use roll, pitch, yaw workflow. If "quat", use quaternion
    """

    # markers with negative IDs are thrown
    if float(new_data_entry[1]) < 0:
        print(f"Negative marker {new_data_entry[1]} detected and thrown")
        return False

    # se3quats need 9 things to work with our code
    if new_data_entry[0] == "quat" and len(new_data_entry) == 9:
        return "quat"

    # xyzrpys need 8 things to work with our code
    if new_data_entry[0] == "rpy" and len(new_data_entry) == 8:
        return "rpy"

    return False


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
    rtabmap_quat = np.array([pose[3], pose[4], pose[5], pose[6]])

    # Matching the April Tag's coordinate frame with RTABmap
    initial_alignment_quat = np.array([-0.5, 0.5, 0.5, 0.5])

    # Quaternion to match RTABmap's coordinate frame with IM
    matching_alignment_quat = np.array([-0.5, 0.5, 0.5, 0.5])
    final_quat = pyrr.quaternion.cross(pyrr.quaternion.cross(
        rtabmap_quat, matching_alignment_quat), initial_alignment_quat)
    rtabmap_quat = np.array([pose[3],pose[4],pose[5],pose[6]])
    
    # -90 degrees around y, -90 old x
    initial_alignment_quat = np.array([-0.5,0.5,0.5,0.5])
    
    # 180 degrees around z
    # matching_alignment_quat = np.array([0.5,0.5,0.5,0.5])
    matching_alignment_quat = np.array([0,0,1,0])
          
    final_quat = pyrr.quaternion.cross(pyrr.quaternion.cross(rtabmap_quat, matching_alignment_quat),initial_alignment_quat)
    # final_quat = pyrr.quaternion.cross(rtabmap_quat, initial_alignment_quat)
    # final_quat = [final_quat[1],final_quat[2],final_quat[3],final_quat[0]]
    
    pdb.set_trace()
    # conver the pose using the alignment quaternion
    rtabmap_pose = np.array([pose[0], pose[1], pose[2]])
    final_pose = pyrr.quaternion.apply_to_vector(
        initial_alignment_quat, rtabmap_pose)

    # create the new se3quat corresponding to the corrected pose
    for i in range(7):
        if i < 3:
            pose[i] = final_pose[i]
        else:
            pose[i] = final_quat[i-3]

    return pose


def process_IM_GT_data(file_path, tag_poses):
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
    with open(file_path, "r") as json_file:
        data = json.load(json_file)
        for item in data["tag_vertices"]:
            IM_processed_poses.append(
                [item["translation"]["x"], item["translation"]["y"], item["translation"]["z"]])
            IM_processed_quats.append(
                [item["rotation"]["x"], item["rotation"]["y"], item["rotation"]["z"], item["rotation"]["w"]])

    # Ground Truth Data
    GT_processed_poses = []
    GT_processed_quats = []
    for pos in tag_poses:
        GT_processed_poses.append(pos["pose"][0:3])
        GT_processed_quats.append(pos["pose"][3:])

    return np.array(IM_processed_poses), np.array(GT_processed_poses), np.array(IM_processed_quats), np.array(GT_processed_quats)

def plot_IM_GT_data(im_pos, gt_pos, im_quat, gt_quat):
    """
    Visualize the positions of the IM tag data and the RTABmap tag data.

    Args:
        im (list): a list of tag poses from invisible map
        gt (list): a list of ground truth tag poses from rtabmap
    """
    fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.set_xlim(auto=True)
    ax.set_xlabel("x")
    ax.set_ylim(-1, 5)
    ax.set_ylabel("y")
    ax.set_zlim(auto=True)
    ax.set_zlabel("z")
    ax.view_init(120, -90)

    plt.plot(im_pos[:, 0], im_pos[:, 1], im_pos[:, 2], "-o", c="black")
    plt.plot(gt_pos[:, 0], gt_pos[:, 1], gt_pos[:, 2], "-o", c="orange")

    # If there were orientations computed
    if len(im_quat > 0) and len(gt_quat > 0):
        # show orientation
        for i in range(len(im_quat)):
            im_coordinate_frame = R.from_quat(im_quat[i]).as_matrix()
            ax.quiver(im_pos[i, 0], im_pos[i, 1], im_pos[i, 2], im_coordinate_frame[0]
                      [0], im_coordinate_frame[1][0], im_coordinate_frame[2][0], color="r")
            ax.quiver(im_pos[i, 0], im_pos[i, 1], im_pos[i, 2], im_coordinate_frame[0]
                      [1], im_coordinate_frame[1][1], im_coordinate_frame[2][1], color="g")
            ax.quiver(im_pos[i, 0], im_pos[i, 1], im_pos[i, 2], im_coordinate_frame[0]
                      [2], im_coordinate_frame[1][2], im_coordinate_frame[2][2], color="b")
        for i in range(len(gt_quat)):
            gt_coordinate_frame = R.from_quat(gt_quat[i]).as_matrix()
            ax.quiver(gt_pos[i, 0], gt_pos[i, 1], gt_pos[i, 2], gt_coordinate_frame[0]
                      [0], gt_coordinate_frame[1][0], gt_coordinate_frame[2][0], color="r")
            ax.quiver(gt_pos[i, 0], gt_pos[i, 1], gt_pos[i, 2], gt_coordinate_frame[0]
                      [1], gt_coordinate_frame[1][1], gt_coordinate_frame[2][1], color="g")
            ax.quiver(gt_pos[i, 0], gt_pos[i, 1], gt_pos[i, 2], gt_coordinate_frame[0]
                      [2], gt_coordinate_frame[1][2], gt_coordinate_frame[2][2], color="b")

    ax.legend(["invisible map", "ground truth"])
    plt.show()


def run(rtabmap_data_path, outputted_json_path, processed_graph_path, visualize):
    """
    Run the script's main pipeline

    Args:
        rtabmap_data_path (string): a string corresponding to the json file of tag data from rtabmap
        outputted_json_path (string): a string corresponding to the place where the gt_data json file should be stored
        processed_graph_path (string): a string corresponding to the place where we get our tag data from invisible maps
        visualize (boolean): choose whether or not to visualize the points / orientations.
    """
    # initialize empty list of tag poses
    tag_poses = []

    with open(rtabmap_data_path, "r") as f:
        for line in f.readlines():
            new_data_entry = line.split()
            print(new_data_entry)

            # XYZ and quaternion data
            if interpret_data_type(new_data_entry) == "quat":

                pose = generate_correct_pose(
                    [float(number) for number in new_data_entry[2:]])

                tag_poses.append(
                    {"tag_id": int(new_data_entry[1]), "pose": pose})

            # XYZ and RPY data
            elif interpret_data_type(new_data_entry) == "rpy":

                # Convert RPY to a quat to use generate_correct_pose
                new_data_entry = [float(number)
                                  for number in new_data_entry[1:]]
                xyz = new_data_entry[1:4]
                rpy = new_data_entry[4:]
                quat = list(pyrr.quaternion.create_from_eulers(rpy))

                pose = generate_correct_pose(xyz+quat)
                tag_poses.append(
                    {"tag_id": int(new_data_entry[0]), "pose": pose})

            else:
                print(f"I don't have any way to handle {new_data_entry}")

    # dump it to a json file
    with open(outputted_json_path, "w") as f:

        # Need to sort tag poses so algorithm aligns tags correctly.
        tag_poses = sorted(tag_poses, key=lambda x: x['tag_id'])
        data = {
            "poses":
                tag_poses
        }
        json.dump(data, f, indent=2)

    # Visualize anchor positions
    if visualize and processed_graph_path != "":
        im_pos, gt_pos, im_quat, gt_quat = process_IM_GT_data(
            processed_graph_path, tag_poses)
        plot_IM_GT_data(im_pos, gt_pos, im_quat, gt_quat)


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    # Take in the name of the test for configuration purposes
    NAME_OF_TEST = args.n.lower()

    with open("../rtabmap/gt_analysis_config.json") as config_file:
        config = json.load(config_file)[NAME_OF_TEST]
        print(f"your configuration: {config}")

    # By default, visualization is False. The addition of visualization tag will make it True
    VISUALIZE = False
    if args.v:
        # Visualize
        VISUALIZE = True

    RTABMAP_DATA_PATH = config["RTABMAP_DATA_PATH"]
    OUTPUTTED_JSON_PATH = config["OUTPUTTED_JSON_PATH"]
    PROCESSED_GRAPH_DATA_PATH = config["PROCESSED_GRAPH_DATA_PATH"]

    run(RTABMAP_DATA_PATH, OUTPUTTED_JSON_PATH,
        PROCESSED_GRAPH_DATA_PATH, VISUALIZE)
