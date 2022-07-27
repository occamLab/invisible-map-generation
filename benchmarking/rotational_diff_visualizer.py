"""
Analyzing the error distribution of oblique tag observations v.s. straight-on tag observations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse


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
    
def error_information (rpys):
    """
    Give basic information about both datasets -- median, mean, range, std, etc.

    Args:
        rpys(list): a list of lists containing rolls, pitches, and yaws data. 
    """
    names = ["oblique-rolls","oblique-pitches","oblique-yaws",
             "straight-rolls","straight-pitches","straight-yaws"]
    
    for i, dataset in enumerate(rpys):
        # print(dataset)
        print(names[i])
        print(f"Median: {np.median(dataset)}")
        print(f"Mean: {np.mean(dataset)}")
        print(f"Range: {max(dataset)-min(dataset)}")
        print(f"Standard Deviation: {np.std(dataset)}")
        print("\n")
    
def error_visualization(rpys,test_name):
    """
    Visualize both datasets

    Args:
        o_rolls (list): list of rolls from oblique observations
        o_pitches (list): list of pitches from oblique observations
        o_yaws (list): list of yaws from oblique observations
        s_rolls (list): list of rolls from straight observations
        s_pitches (list): list of pitches from straight observations
        s_yaws (list): list of yaws from straight observations
        test_name (str): name of the test (for titling/saving purposes)

    Returns:
        True: when complete!
    """
    o_rolls,o_pitches,o_yaws,s_rolls,s_pitches,s_yaws = rpys
    plt.figure(figsize= (15,10))
    plt.suptitle(test_name, fontsize = 20)
    
    plt.subplot(2,3,1, title = "oblique-rolls")
    plt.hist(o_rolls, range = (-10,10), bins = 25, color = "red")

    plt.subplot(2,3,2, title = "oblique-pitches" )
    plt.hist(o_pitches, range = (-10,10), bins = 25, color = "green")
    
    plt.subplot(2,3,3, title = "oblique-yaws")
    plt.hist(o_yaws, range = (-10,10), bins = 25, color = "blue")
    
    plt.subplot(2,3,4, title = "straight-rolls")
    plt.hist(s_rolls, range = (-10,10), bins = 25, color = "red")

    plt.subplot(2,3,5, title = "straight-pitches")
    plt.hist(s_pitches, range = (-10,10), bins = 25, color = "green")

    plt.subplot(2,3,6, title = "straight-yaws")
    plt.hist(s_yaws, range = (-10,10), bins = 25, color = "blue")
    plt.savefig(f"imgs/{test_name}.png")
    plt.show()
    
    return True
    
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
        error_information((o_rolls,o_pitches,o_yaws,s_rolls,s_pitches,s_yaws))
    if visualize:
        error_visualization((o_rolls,o_pitches,o_yaws,s_rolls,s_pitches,s_yaws),test_name)

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