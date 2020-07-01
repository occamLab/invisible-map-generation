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


def process_map(x):
    test_graph = convert_json.as_graph(x)

    test_graph.weights = np.array([
        0.,  0.,  0., 0.,  0.,  0.,
        0.,  0.,  0., 0.,  0.,  0.,
        0.,  0.,  0., -1e1,  1e2,  1e2
    ])

    # Load these weights into the graph
    test_graph.update_edges()

    # Create the g2o object and optimize
    test_graph.generate_unoptimized_graph()
    test_graph.optimize_graph()

    # Change vertex estimates based off the optimized graph
    test_graph.update_vertices()

    resulting_map = graph_utils.optimizer_to_map(
        test_graph.vertices, test_graph.optimized_graph)
    locations = resulting_map['locations']
    tag_verts = resulting_map['tags']

    return tag_verts

def make_processed_map_JSON(tag_locations):
    tag_vertex_map = map(lambda curr_tag: {'translation': {'x': curr_tag[0], 'y': curr_tag[1], 'z': curr_tag[2]},
                                           'rotation': {'x': curr_tag[3],
                                                        'y': curr_tag[4],
                                                        'z': curr_tag[5],
                                                        'w': curr_tag[6]},
                                           'id': int(curr_tag[7])}, tag_locations)
    return json.dumps({'tag_vertices': list(tag_vertex_map),
                       'odometry_vertices': [],
                       'waypoints_vertices': []})

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
# TODO: for some reason many of the maps have multiple copies of the same tag in the same frame
for map_name, map_json in to_process.items():
    print(map_name)
    json_blob = bucket.get_blob(map_json)
    if json_blob is not None:
        json_data = json_blob.download_as_string()
        x = json.loads(json_data)
        tag_locations = process_map(x)
        processed_map = make_processed_map_JSON(tag_locations)
        processed_map_filename = os.path.basename(map_json)[:-5] + '_processed.json'
        processed_map_full_path = os.path.join('TestProcessed', processed_map_filename)
        processed_map_blob = bucket.blob(processed_map_full_path)
        processed_map_blob.upload_from_string(processed_map)
        db.reference('maps').child(map_name).child('map_file').set(processed_map_full_path)
    else:
        print("map file was missing")
