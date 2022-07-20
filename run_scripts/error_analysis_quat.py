"""
Analyzing the error distribution of oblique tag observations v.s. straight-on tag observations.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as R
import argparse
import sba_evaluator_replace_pixels as sba

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
    # print(net_rotation)
    net_quat = R.from_matrix(net_rotation).as_quat()
    
    
    for value in net_quat:
        value = value/net_quat[3]
        

    # data collected with/without LIDAR
    t_x = without_LIDAR_array[0,3] - LIDAR_array[0,3]
    t_y = without_LIDAR_array[1,3] - LIDAR_array[1,3]
    t_z = without_LIDAR_array[2,3] - LIDAR_array[2,3]
    # print(net_translation)
    net_translation = (t_x, t_y, t_z)
    
    return net_quat[0:3], net_translation
    
def error_information (qxyzs):
    """
    Give basic information about both datasets -- median, mean, range, std, etc.

    Args:
        qxyzs(list): a list of lists containing rolls, pitches, and yaws data. 
    """
    names = ["oblique-rolls","oblique-pitches","oblique-yaws",
             "straight-rolls","straight-pitches","straight-yaws"]
    
    for i, dataset in enumerate(qxyzs):
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

def error_visualization(qxyzs, xyzs, test_name):
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


    # plt.savefig(f"../error_analysis/imgs/{test_name}.png")
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
            rpy, o_xyz= error_calculation(none_pose,LIDAR_pose)
            
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
            rpy, s_xyz = error_calculation(none_pose,LIDAR_pose)
            
            s_qx.append(rpy[0])
            s_qy.append(rpy[1])
            s_qz.append(rpy[2])
            s_x.append(s_xyz[0])
            s_y.append(s_xyz[1])
            s_z.append(s_xyz[2])
        
            
    if info:
        error_information((o_qx,o_qy,o_qz,s_qx,s_qy,s_qz))
    if visualize:
        error_visualization((o_qx,o_qy,o_qz,s_qx,s_qy,s_qz), (o_x,o_y,o_z,s_x,s_y,s_z), test_name)
    
    variances = compute_variances([o_qx,o_qy,o_qz,o_x,o_y,o_z])
    
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

    with open("../error_analysis/error_config.json") as config_file:
        config = json.load(config_file)[NAME_OF_TEST]
        print(f"your configuration: {config}")
        
    OBLQ_PATH = config["OBLIQUE_DATA_PATH"]
    STRT_PATH = config["STRAIGHT_ON_DATA_PATH"]
    sba.throw_out_bad_tags(OBLQ_PATH)

    
    VISUALIZE = args.v
    GIVE_INFO = args.i
    run(OBLQ_PATH,STRT_PATH,VISUALIZE,GIVE_INFO,NAME_OF_TEST)