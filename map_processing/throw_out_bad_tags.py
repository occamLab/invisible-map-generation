"""
Evaluating the effectiveness of the SBA (sparse bundle adjustment) algorithm by manually calculating it.
Takes in the pose of a tag, computes where the corner pixels SHOULD be on the
camera's screen, then compares it to where the pixels actually were on the screen.
Allows us to essentially remake the SBA algorithm.
"""
from collections import defaultdict
import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import json
import numpy as np
import argparse

import benchmarking.benchmarking_utils as B

# import map_processing
# from map_processing import benchmarking_utils as B


def make_parser():
    """
    Creates a ArgumentParser object for CLI.
    """
    p = argparse.ArgumentParser(
        description="Visualize and analyze error from oblique/straight tag observations"
    )

    p.add_argument("-v", help="visualize data", action="store_true")

    p.add_argument(
        "-i",
        help="show extra information, including coordinate locations/what disqualified certain tags, etc.",
        action="store_true",
    )

    p.add_argument(
        "-f",
        help="DON'T change the json file to reflect the thrown tags. Will automatically change the json unless this flag is specified.",
        action="store_true",
    )

    return p


def sba_evaluate(tag_idx, unprocessed_map_data, visualize=False, verbose=False):
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

    # We've discovered that the alpha metric works when observations are straight-on.
    THROW_OUT_OBLIQUE_TAGS = True
    PRINT_CORNER_PIXELS = False
    SHOW_RMS_ERROR = False

    camera_intrinsics = B.compute_camera_intrinsics(tag_idx, unprocessed_map_data)
    observed_pixels = np.reshape(
        unprocessed_map_data["tag_data"][tag_idx][0]["tag_corners_pixel_coordinates"],
        [2, 4],
        order="F",
    )
    # camera_pose = B.compute_camera_pose(tag_idx,unprocessed_map_data)
    tag_pose = B.compute_tag_pose(tag_idx, unprocessed_map_data)
    corner_pixel_poses = B.set_corner_pixels_tag_frame()

    relevant_tag = unprocessed_map_data["tag_data"][tag_idx][0]["tag_id"]
    # This equation, whiteboarded out, to convert from the tag frame's corner pixels to the
    # corner pixels we see on the phone.
    sba_pixel_corners = (
        camera_intrinsics @ B.MATRIX_SIZE_CONVERTER @ tag_pose @ corner_pixel_poses
    )

    for i in range(4):
        for j in range(3):
            sba_pixel_corners[j, i] = sba_pixel_corners[j, i] / sba_pixel_corners[2, i]

    # pdb.set_trace()
    sba_pixel_corners = sba_pixel_corners[0:-1]

    if verbose and PRINT_CORNER_PIXELS:
        print("bottom left:")
        print(sba_pixel_corners[:2, 0])

        print("bottom right:")
        print(sba_pixel_corners[:2, 1])
        np
        print("top right:")
        print(sba_pixel_corners[:2, 2])

        print("top left:")
        print(sba_pixel_corners[:2, 3])

        print("\n")

    RMS_error, throw = B.compute_RMS_error(sba_pixel_corners, observed_pixels)

    # Error between observed pixels in the camera + the calculated pixel positions based on camera and tag pose.
    if verbose and SHOW_RMS_ERROR:
        print(f"tag {relevant_tag} RMS error: {RMS_error} pixels")

    if visualize and not throw:
        B.visualizing_corner_pixel_differences(
            [sba_pixel_corners, observed_pixels], relevant_tag, "CO"
        )

    if throw is not True and THROW_OUT_OBLIQUE_TAGS is True:
        throw = B.check_straight_on_detection(relevant_tag, tag_pose, verbose)

    # We don't want to count tags that are observed from really far away.
    if throw is not True:
        throw = B.check_detection_distance(relevant_tag, tag_pose, verbose)

    return RMS_error, throw, relevant_tag, sba_pixel_corners


def throw_out_bad_tags(data_path, visualize=False, verbose=False, fix_it=True):
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

    throw_ids = []
    if (unprocessed_map_data["cloud_data"]):
        data = list(map(lambda x: x[0], unprocessed_map_data["cloud_data"]))
        by_id = defaultdict(list)

        for item in data:
            by_id[item['cloudIdentifier']].append(item)

        for anchor_id in by_id: 
            max_dis = 0
            for instance in by_id[anchor_id]:
                pose = list(filter(lambda x: x["id"] == instance["poseId"], unprocessed_map_data["pose_data"]))[0]["pose"]
                reshaped = np.reshape(pose, (4,4))
                translation1 = reshaped[3, :3]
                for i2 in by_id[anchor_id]:
                    p2 = list(filter(lambda x: x["id"] == i2["poseId"], unprocessed_map_data["pose_data"]))[0]["pose"]
                    r2 = np.reshape(p2, (4,4))
                    translation2 = r2[3, :3]
                    distance = np.linalg.norm(translation1 - translation2)
                    # print("pose distance: ", distance)
                    if distance > max_dis:
                        max_dis = distance
            if max_dis > 15:
                throw_ids.append(anchor_id)
        
        unprocessed_map_data["cloud_data"] = [
            i
            for j, i in enumerate(unprocessed_map_data["cloud_data"])
            if (i[0]["cloudIdentifier"] not in throw_ids)
        ]
    
    throws = []
    throws_indeces = []
    errors = []

    if (unprocessed_map_data["tag_data"]):
        for i in range(len(unprocessed_map_data["tag_data"])):
            sba_rms_error, throw, relevant_tag, corner_pixels = sba_evaluate(
                i, unprocessed_map_data, visualize, verbose
            )
            errors.append(sba_rms_error)
            unprocessed_map_data["tag_data"][i][0][
                "tag_corners_pixel_coordinates"
            ] = np.ndarray.flatten(corner_pixels, order="F").tolist()

            if throw:
                throws.append(relevant_tag)
                throws_indeces.append(i)
                continue

        unprocessed_map_data["tag_data"] = [
            i
            for j, i in enumerate(unprocessed_map_data["tag_data"])
            if j not in throws_indeces
        ]
    
    fix_it = True
    if fix_it:
        with open(data_path, "w") as f:
            json.dump(unprocessed_map_data, f, indent=2)

    # percent_thrown = 100 * (len(throws) / len(errors))

    return (
       f"Threw out {len(throws)} tags and {len(throw_ids)} anchors."
    )


if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    parser = make_parser()
    args = parser.parse_args()

    PATH = "../benchmarking/datasets/floor_2_straight_uncleaned.json"

    FIX_IT = True
    if args.f:
        print("not actually changing the file")
        FIX_IT = False

    VISUALIZE = False
    if args.v:
        VISUALIZE = True

    VERBOSE = False
    if args.i:
        VERBOSE = True

    print(throw_out_bad_tags(PATH, VISUALIZE, VERBOSE, FIX_IT))
