"""
Convert a processed json into an unprocessed json
"""
import matplotlib.pyplot as plt
import json
import numpy as np
import sba_evaluator as sba

CAMERA_POSE_FLIPPER = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])


def create_unprocessed_json(unprocessed_path, processed_path):
    """

    Args:


    Returns:
    """
    # Load data
    with open(unprocessed_path) as data_file_unprocessed:
        unprocessed_data = json.load(data_file_unprocessed)
    with open(processed_path) as data_file_processed:
        processed_data = json.load(data_file_processed)

    for pose_idx in range(len(processed_data["odometry_vertices"])):
        # Rewrite pose data with processed pose data
        translation_pose = processed_data["odometry_vertices"][pose_idx]["translation"]
        rotation_pose = processed_data["odometry_vertices"][pose_idx]["rotation"]
        # pose_id = processed_data["odometry_["poseId"]


create_unprocessed_json("/home/rdave/invisible-map-generation/.cache/unprocessed_maps/rawMapData/zfd9Row5EpX9fhTgdYlBtTxliqR2/floor_2_obleft 83183896790958.json", "/home/rdave/invisible-map-generation/.cache/TestProcessed/rawMapData/zfd9Row5EpX9fhTgdYlBtTxliqR2/floor_2_obleft 83183896790958.json")

