#!/usr/bin/env python3

import json

import convert_json
import numpy as np
import graph_utils
import matplotlib.pyplot as plt
import firebase_admin
from firebase_admin import db
from firebase_admin import credentials
from firebase_admin import storage
import os
from g2o import Quaternion

def axis_equal(ax):
    # Create cubic bounding box to simulate equal aspect ratio
    axis_range_from_limits = lambda limits: limits[1] - limits[0]
    max_range = np.array([axis_range_from_limits(ax.get_xlim()), axis_range_from_limits(ax.get_ylim()), axis_range_from_limits(ax.get_zlim())]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (ax.get_xlim()[1] + ax.get_xlim()[0])
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (ax.get_ylim()[1] + ax.get_ylim()[0])
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (ax.get_zlim()[1] + ax.get_zlim()[0])
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], 'w')

def optimize_map(x, tune_weights=False, visualize=False):
    test_graph = convert_json.as_graph(x)
    # higher means more noisy (note: the uncertainty estimates of translation seem to be pretty over optimistic, hence the large correction here)
    sensible_default_weights = np.array([
        -6.,  -6.,  -6.,  -6.,  -6.,  -6.,
        18,  18,  18,  18,  18,  18,
        0.,  0.,  0., -1,  -1,  1e2
    ])

    trust_odom = np.array([
        -3.,  -3.,  -3., -3.,  -3.,  -3.,
        10.6,  10.6, 10.6, 10.6,  10.6,  10.6,
        0.,  0.,  0., -1,  -1,  1e2
    ])

    trust_tags = np.array([
        10,  10,  10, 10,  10,  10,
        -10.6,  -10.6, -10.6, -10.6,  -10.6,  -10.6,
        0,  0,  0, -1e2,  3,  3
    ])
    test_graph.weights = sensible_default_weights

    # Load these weights into the graph
    test_graph.update_edges()
    test_graph.generate_unoptimized_graph()
    all_tags_original = test_graph.get_tags_all_position_estimate()
    starting_map = graph_utils.optimizer_to_map(
        test_graph.vertices, test_graph.unoptimized_graph)
    original_tag_verts = starting_map['tags']
    if tune_weights:
        test_graph.expectation_maximization_once()
        print("tuned weights", test_graph.weights)
    # Create the g2o object and optimize
    test_graph.generate_unoptimized_graph()
    test_graph.optimize_graph()

    # Change vertex estimates based off the optimized graph
    test_graph.update_vertices()

    resulting_map = graph_utils.optimizer_to_map(
        test_graph.vertices, test_graph.optimized_graph)
    locations = resulting_map['locations']
    tag_verts = resulting_map['tags']
    waypoint_verts = resulting_map['waypoints']
    if visualize:
        all_tags = test_graph.get_tags_all_position_estimate()

        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')
        plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], '.', c='b', label='Odom Vertices')
        plt.plot(original_tag_verts[:, 0], original_tag_verts[:, 1], original_tag_verts[:, 2], 'o', c='c', label='Tag Vertices Original')
        plt.plot(tag_verts[:, 0], tag_verts[:, 1], tag_verts[:, 2], 'o', c='r', label='Tag Vertices')
        for vert in tag_verts:
            ax.text(vert[0], vert[1], vert[2], str(int(vert[-1])), color='black')
        for tag_vert in tag_verts:
            R = Quaternion(tag_vert[3:-1]).rotation_matrix()
            axis_to_color = ['r','g','b']
            for axis_id in range(3):
                ax.quiver(tag_vert[0], tag_vert[1], tag_vert[2], R[0, axis_id], R[1, axis_id],
                          R[2, axis_id], length=1, color=axis_to_color[axis_id])
        plt.plot(waypoint_verts[1][:, 0], waypoint_verts[1][:, 1], waypoint_verts[1][:, 2], 'o', c='y', label='Waypoint Vertices')
        for vert_idx in range(len(waypoint_verts[0])):
            vert = waypoint_verts[1][vert_idx]
            waypoint_name = waypoint_verts[0][vert_idx]['name']
            ax.text(vert[0], vert[1], vert[2], waypoint_name, color='black')
        plt.plot(all_tags[:, 0], all_tags[:, 1], all_tags[:, 2], '.', c='g', label='All Tag Edges')
        plt.plot(all_tags_original[:, 0], all_tags_original[:, 1], all_tags_original[:, 2], '.', c='m', label='All Tag Edges Original')
        tag_edge_std_dev_before_and_after = compare_std_dev(all_tags, all_tags_original)
        tag_vertex_shift = original_tag_verts - tag_verts
        print("tag_vertex_shift", tag_vertex_shift)
        axis_equal(ax)
        plt.legend()
        plt.show()
    return tag_verts, waypoint_verts

def compare_std_dev(all_tags, all_tags_original):
    return {int(tag_id): (np.std(all_tags_original[all_tags_original[:, -1] == tag_id, :-1], axis=0),
            np.std(all_tags[all_tags[:, -1] == tag_id, :-1], axis=0)) for tag_id in np.unique(all_tags[:,-1])}


def make_processed_map_JSON(tag_locations, waypoint_locations):
    tag_vertex_map = map(lambda curr_tag: {'translation': {'x': curr_tag[0], 'y': curr_tag[1], 'z': curr_tag[2]},
                                           'rotation': {'x': curr_tag[3],
                                                        'y': curr_tag[4],
                                                        'z': curr_tag[5],
                                                        'w': curr_tag[6]},
                                           'id': int(curr_tag[7])}, tag_locations)
    waypoint_vertex_map = map(lambda idx: {'translation': {'x': waypoint_locations[1][idx][0], 'y': waypoint_locations[1][idx][1], 'z': waypoint_locations[1][idx][2]},
                                                     'rotation': {'x': waypoint_locations[1][idx][3],
                                                                  'y': waypoint_locations[1][idx][4],
                                                                  'z': waypoint_locations[1][idx][5],
                                                                  'w': waypoint_locations[1][idx][6]},
                                           'id': waypoint_locations[0][idx]['name']}, range(len(waypoint_locations[0])))
    return json.dumps({'tag_vertices': list(tag_vertex_map),
                       'odometry_vertices': [],
                       'waypoints_vertices': list(waypoint_vertex_map)})

def process_map(map_name, map_json, visualize=False):
    json_blob = bucket.get_blob(map_json)
    if json_blob is not None:
        json_data = json_blob.download_as_string()
        x = json.loads(json_data)
        tag_locations, waypoint_locations = optimize_map(x, False, visualize)
        processed_map = make_processed_map_JSON(tag_locations, waypoint_locations)
        processed_map_filename = os.path.basename(map_json)[:-5] + '_processed.json'
        processed_map_full_path = os.path.join('TestProcessed', processed_map_filename)
        processed_map_blob = bucket.blob(processed_map_full_path)
        processed_map_blob.upload_from_string(processed_map)
        db.reference('maps').child(map_name).child('map_file').set(processed_map_full_path)
        print("processed map", map_name)
    else:
        print("map file was missing")

def unprocessed_maps_callback(m):
    if type(m.data) == str:
        # a single new map just got added
        process_map(m.path.lstrip('/'), m.data)
    elif type(m.data) == dict:
        # this will be a dictionary of all the data that is there initially
        for map_name, map_json in m.data.items():
            process_map(map_name, map_json)

# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))

# Initialize the app with a service account, granting admin privileges
app = firebase_admin.initialize_app(cred, {
    'databaseURL': 'https://invisible-map-sandbox.firebaseio.com/',
    'storageBucket': 'invisible-map.appspot.com'
})

ref = db.reference('/unprocessed_maps')
to_process = ref.get()
bucket = storage.bucket(app=app)

ref.listen(unprocessed_maps_callback)
