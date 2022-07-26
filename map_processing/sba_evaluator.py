"""
Evaluating the effectiveness of the SBA (sparse bundle adjustment) algorithm by manually calculating it.
Takes in the pose of a tag, computes where the corner pixels SHOULD be on the
camera's screen, then compares it to where the pixels actually were on the screen.
Allows us to essentially remake the SBA algorithm.
"""

import matplotlib.pyplot as plt
import json
import numpy as np
import argparse
import map_processing.benchmarking_utils as B

def make_parser():
    """
    Creates a ArgumentParser object for CLI.
    """
    p = argparse.ArgumentParser(
        description="Visualize and analyze error from oblique/straight tag observations")

    p.add_argument(
        "-v", help="visualize data", action = "store_true")

    p.add_argument(
        "-c", help="show the individual coordinate values of each corner", action="store_true")

    p.add_argument(
        "-f", help="DON'T change the json file to reflect the thrown tags. Will automatically change the json unless this flag is specified.", action="store_false")

    return p
  
  
def sba_evaluate(tag_idx, unprocessed_map_data, visualize = False, show_coords = False):  
    """
    Run the SBA evaluator on a tag.
    
    Workflow is as follows:
    - Compute where the corner pixels "should" be given the tag's pose and camera intrinsics. (computed pixels)
    - Compare this to where invisible maps says the corner pixels are on the screen. (observed pixels)
    - If the error is too large, make "throw" true
    
    Args:
        tag_idx: tag index. We iterate through all of the tags, tag_idx is just a way to index into the dictionary.
        unprocessed_map_data: the dictionary pulled from the .json file containing all of the data regarding the map.
        
    Returns:
        RMS_error (float): a float expressing the amount of error found between the observed pixels and computed pixels
        throw (bool): true if the value should be thrown out. False otherwise.
        relevant tag : the tag corresponding to the current index.
        
    """  
    
    camera_intrinsics = B.compute_camera_intrinsics(tag_idx,unprocessed_map_data)
    observed_pixels = np.reshape(unprocessed_map_data["tag_data"][tag_idx][0]["tag_corners_pixel_coordinates"],[2,4],order = 'F')
    tag_pose = B.compute_tag_pose(tag_idx,unprocessed_map_data)
    corner_pixel_poses = B.set_corner_pixels_tag_frame()
    
    # This equation, whiteboarded out, to convert from the tag frame's corner pixels to the
    # corner pixels we see on the phone. 
    sba_pixel_corners = camera_intrinsics@B.MATRIX_SIZE_CONVERTER@tag_pose@corner_pixel_poses
 
    for i in range(4):
        for j in range(3):
            sba_pixel_corners[j, i] = sba_pixel_corners[j, i]/sba_pixel_corners[2,i]
            
    # pdb.set_trace()
    sba_pixel_corners = sba_pixel_corners[0:-1] 
    
    if show_coords:
        
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
    RMS_error, throw = B.compute_RMS_error(sba_pixel_corners, observed_pixels)
    # print(f"tag {relevant_tag} RMS error: {RMS_error}")
    
    if visualize and not throw: 
        B.visualizing_corner_pixel_differences([sba_pixel_corners, observed_pixels],relevant_tag , "CO")
    
    
    return RMS_error, throw, relevant_tag, sba_pixel_corners

def throw_out_bad_tags(data_path, visualize = False, show_coords= False, fix_it = True):
    """
    Helper function to throw out tags with too high of an error based upon the calculation done in
    sba_evaluate(). 

    Args:
        data_path (str): path to the file to be overwritten

    Returns:
        a bunch of print statements, but the helper function overwrites a file in its workflow.
        the print statement is just an indication of how much stuff was thrown out. 
    """
    
    with open(data_path) as data_file:
        unprocessed_map_data = json.load(data_file)
    
    throws = []
    throws_indeces = []
    errors = []
    
    for i in range(len(unprocessed_map_data["tag_data"])):
        sba_rms_error, throw,relevant_tag, corner_pixels = sba_evaluate(i, unprocessed_map_data, visualize, show_coords)

        unprocessed_map_data["tag_data"][i][0]["tag_corners_pixel_coordinates"] = np.ndarray.flatten(corner_pixels, order = "F").tolist()
        
        if throw: 
            throws.append(relevant_tag)
            throws_indeces.append(i)
            continue
        
        errors.append(sba_rms_error)
   

    unprocessed_map_data["tag_data"] = [i for j, i in enumerate(unprocessed_map_data["tag_data"]) if j not in throws_indeces]
    if fix_it:  
        with open(data_path,"w") as f:
            json.dump(unprocessed_map_data, f, indent = 2)
        
    percent_thrown = 100* (len(throws)/len(errors))
    
    return (f"average error: {np.mean(errors)}" + "\n" + 
            f"list of tags thrown: {throws}" + "\n" +
            f"percentage of tags thrown: {percent_thrown:.2f}%")
    
if __name__ == "__main__":
    np.set_printoptions(suppress=True)
    
    parser = make_parser()
    args = parser.parse_args()
    
    PATH = "../benchmarking/datasets/floor_2_obright_cleaned.json"
    
    FIX_IT = True
    if args.f:
        FIX_IT = False
    
    VISUALIZE = False
    if args.v:
        VISUALIZE = True
        
    SHOW_INDIVIDUAL_COORDS = False
    if args.c:
        SHOW_INDIVIDUAL_COORDS = True
    
    print(throw_out_bad_tags(PATH, VISUALIZE, SHOW_INDIVIDUAL_COORDS, FIX_IT))