"""
Analyzing the error distribution of oblique tag observations v.s. straight-on tag observations.
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import benchmarking_utils as B
import map_processing.throw_out_bad_tags as tag_filter


def make_parser():
    """
    Creates a ArgumentParser object for CLI.
    """
    p = argparse.ArgumentParser(
        description="Visualize and analyze error from oblique/straight tag observations")

    p.add_argument(
        "-n", help="name of the test you'd like to run in the configuration file")
    
    p.add_argument(
        "-v", help="visualize data", action = "store_true")

    p.add_argument(
        "-i", help="give additional information (mean/std/etc.)", action="store_true")

    return p

def error_calculation (mat1, mat2):
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
    net_rpy = R.from_matrix(net_rotation).as_euler("xyz", degrees = True)
    return net_rpy
      
def run(oblq_path,strt_path,visualize,info,test_name):
    """
    Run error analysis

    Args:
        oblq_path (str): path to oblique data
        strt_path (str): path to straight data
        visualize (bool): visualize the data or not
        info (bool): print information about data or not
        test_name (str): name of test (for titling/saving purposes)
    """
    with open(oblq_path,"r") as datafile:
        o_rolls = []
        o_pitches = []
        o_yaws = []
        
        data_dict = json.load(datafile)
        for tag in data_dict["tag_data"]:
            none_pose = tag[0]["orig_pose"]
            LIDAR_pose = tag[0]["tag_pose"]
            rpy = error_calculation(none_pose,LIDAR_pose)
            
            o_rolls.append(rpy[0])
            o_pitches.append(rpy[1])
            o_yaws.append(rpy[2])

    with open(strt_path,"r") as datafile:
        s_rolls = []
        s_pitches = []
        s_yaws = []

        data_dict = json.load(datafile)
        for tag in data_dict["tag_data"]:
            none_pose = tag[0]["orig_pose"]
            LIDAR_pose = tag[0]["tag_pose"]
            rpy = error_calculation(none_pose,LIDAR_pose)
            
            s_rolls.append(rpy[0])
            s_pitches.append(rpy[1])
            s_yaws.append(rpy[2])
            
    if info:
        B.data_information((o_rolls,o_pitches,o_yaws,s_rolls,s_pitches,s_yaws))
    if visualize:
        B.visualize_data_spread((o_rolls,o_pitches,o_yaws,s_rolls,s_pitches,s_yaws),test_name)

if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    NAME_OF_TEST = args.n.lower()

    with open("benchmarking_config.json") as config_file:
        config = json.load(config_file)["OB_VS_STR"][NAME_OF_TEST]
        print(f"your configuration: {config}")
        
    OBLQ_PATH = config["OBLIQUE_DATA_PATH"]
    STRT_PATH = config["STRAIGHT_ON_DATA_PATH"]
    
    VISUALIZE = args.v
    GIVE_INFO = args.i
    run(OBLQ_PATH,STRT_PATH,VISUALIZE,GIVE_INFO,NAME_OF_TEST)