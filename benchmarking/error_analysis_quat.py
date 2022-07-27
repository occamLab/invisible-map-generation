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
import map_processing.benchmarking_utils as B
import map_processing.sba_evaluator as sba

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
        o_qx = []
        o_qy = []
        o_qz = []
        o_x = []
        o_y = []
        o_z = []
        
        data_dict = json.load(datafile)
        for tag in data_dict["tag_data"]:
            none_pose = tag[0]["orig_pose"]
            LIDAR_pose = tag[0]["tag_pose"]
            rpy, o_xyz= B.rotational_difference_calculation(none_pose,LIDAR_pose)
            
            o_qx.append(rpy[0])
            o_qy.append(rpy[1])
            o_qz.append(rpy[2])
            o_x.append(o_xyz[0])
            o_y.append(o_xyz[1])
            o_z.append(o_xyz[2])

    with open(strt_path,"r") as datafile:
        s_qx = []
        s_qy = []
        s_qz = []
        s_x = []
        s_y = []
        s_z = []

        data_dict = json.load(datafile)
        for tag in data_dict["tag_data"]:
            none_pose = tag[0]["orig_pose"]
            LIDAR_pose = tag[0]["tag_pose"]
            rpy, s_xyz = B.rotational_difference_calculation(none_pose,LIDAR_pose)
            
            s_qx.append(rpy[0])
            s_qy.append(rpy[1])
            s_qz.append(rpy[2])
            s_x.append(s_xyz[0])
            s_y.append(s_xyz[1])
            s_z.append(s_xyz[2])
        
            
    if info:
        B.data_information((o_qx,o_qy,o_qz,s_qx,s_qy,s_qz))
    if visualize:
        B.visualize_data_spread((o_qx,o_qy,o_qz,s_qx,s_qy,s_qz), (o_x,o_y,o_z,s_x,s_y,s_z), test_name)
    
    variances = B.compute_variances([o_qx,o_qy,o_qz,o_x,o_y,o_z])
    
    print(f"q_x variance: {variances[0]}")   
    print(f"q_y variance: {variances[1]}")   
    print(f"q_z variance: {variances[2]}")   
    print(f"x variance: {variances[3]}")   
    print(f"y variance: {variances[4]}")   
    print(f"z variance: {variances[5]}")   


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    NAME_OF_TEST = args.n.lower()

    with open("../benchmarking/benchmarking_config.json") as config_file:
        config = json.load(config_file)["OB_VS_STR"][NAME_OF_TEST]
        print(f"your configuration: {config}")
        
    OBLQ_PATH = config["OBLIQUE_DATA_PATH"]
    STRT_PATH = config["STRAIGHT_ON_DATA_PATH"]
    sba.throw_out_bad_tags(OBLQ_PATH, visualize = False, show_coords = False)

    VISUALIZE = args.v
    GIVE_INFO = args.i
    
    run(OBLQ_PATH,STRT_PATH,VISUALIZE,GIVE_INFO,NAME_OF_TEST)