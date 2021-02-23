#!/usr/bin/env python3

import argparse
import glob
import json
import os

import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from g2o import SE3Quat, Quaternion
from varname import nameof

import convert_json_sba
import graph_utils


class GraphHandler:
    class OptimizationError(Exception):
        """Exception to raise when there is an error during graph optimization
        """

        def __init__(self, message=None):
            self.message = message

        def __str__(self):
            if self.message:
                return self.__class__.__name__ + ", " + self.message
            else:
                return self.__class__.__name__ + " was raised"

    # higher means more noisy (note: the uncertainty estimates of translation seem to be pretty over optimistic,
    # hence the large correction here) trying to lock orientation
    weights_dict = {
        "sensible_default_weights": np.array([
            -6., -6., -6., -6., -6., -6.,
            18, 18, 0, 0, 0, 0,
            0., 0., 0., -1, 1e2, -1
        ]),
        "trust_odom": np.array([
            -3., -3., -3., -3., -3., -3.,
            10.6, 10.6, 10.6, 10.6, 10.6, 10.6,
            0., 0., 0., -1, -1, 1e2
        ]),
        "trust_tags": np.array([
            10, 10, 10, 10, 10, 10,
            -10.6, -10.6, -10.6, -10.6, -10.6, -10.6,
            0, 0, 0, -1e2, 3, 3
        ]),
    }

    _app_initialize_dict = {
        'databaseURL': 'https://invisible-map-sandbox.firebaseio.com/',
        'storageBucket': 'invisible-map.appspot.com'
    }

    # Relevant bucket references
    _unprocessed_maps_bucket_ref = "unprocessed_maps"
    _processed_maps_bucket_ref = "TestProcessed"

    def __init__(self, selected_weights, cred):
        # Initialize the app with a service account, granting admin privileges
        self.app = firebase_admin.initialize_app(cred, GraphHandler._app_initialize_dict)
        self.bucket = storage.bucket(app=self.app)

        self.unprocessed_map_ref = db.reference("/" + GraphHandler._unprocessed_maps_bucket_ref)
        self.unprocessed_map_to_process = self.unprocessed_map_ref.get()

        self.selected_weights = str(selected_weights)
        self._cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache")

    def firebase_listen_unprocessed_maps(self):
        self.unprocessed_map_ref.listen(self._unprocessed_maps_callback)

    def process_maps(self, pattern, upload=False):
        self._resolve_cache_folder()
        matching_maps = glob.glob(os.path.join(self._cache_path, "**/" + pattern), recursive=True)

        if len(matching_maps) == 0:
            print("No maps matching pattern {} in recursive search of {}".format(pattern, self._cache_path))

        for map_json_abs_path in matching_maps:
            print("Attempting to process map {}".format(map_json_abs_path))
            try:
                with open(os.path.join(self._cache_path, map_json_abs_path), "r") as json_string_file:
                    json_string = json_string_file.read()
                    json_string_file.close()
                map_json = os.path.sep.join(map_json_abs_path.split(os.path.sep)[len(self._cache_path.split(
                    os.path.sep)) + 1:])
                map_name = self._read_cache_directory(os.path.basename(map_json))
                self._process_map(map_name, map_json, json_string, True, upload)
            except Exception as ex:
                print("Could not process cached map at {} due to error: {}".format(map_json_abs_path, ex))

    # -- Private Methods --

    def _firebase_get_unprocessed_map(self, map_name, map_json):
        json_blob = self.bucket.get_blob(map_json)
        if json_blob is not None:
            json_data = json_blob.download_as_bytes()
            json_string = json.loads(json_data)
            self._cache_map(GraphHandler._unprocessed_maps_bucket_ref, map_name, map_json, json.dumps(json_string))
            return True
        else:
            print("Map '{}' was missing".format(map_name))
            return False

    def _process_map(self, map_name, map_json, json_string, visualize=False, upload=False):
        tag_locations, odom_locations, waypoint_locations = self._optimize_map(json_string, False, visualize)
        processed_map_json = GraphHandler.make_processed_map_JSON(tag_locations, odom_locations, waypoint_locations)
        self._cache_map(GraphHandler._processed_maps_bucket_ref, map_name, map_json, processed_map_json)

        if upload:
            processed_map_filename = os.path.basename(map_json)[:-5] + '_processed.json'
            processed_map_full_path = os.path.join(GraphHandler._processed_maps_bucket_ref, processed_map_filename)
            processed_map_blob = self.bucket.blob(processed_map_full_path)
            processed_map_blob.upload_from_string(processed_map_json)
            db.reference('maps').child(map_name).child('map_file').set(processed_map_full_path)
        print("processed map", map_name)

    def _append_to_cache_directory(self, key, value):
        assert(isinstance(key, str))
        assert(isinstance(value, str))
        directory_json_path = os.path.join(self._cache_path, "directory.json")
        with open(directory_json_path, "r") as directory_file_read:
            directory_json = json.loads(directory_file_read.read())
            directory_file_read.close()
        directory_json[key] = value
        new_directory_json = json.dumps(directory_json)
        with open(directory_json_path, "w") as directory_file_write:
            directory_file_write.write(new_directory_json)
            directory_file_write.close()

    def _read_cache_directory(self, key):
        assert(isinstance(key, str))
        with open(os.path.join(self._cache_path, "directory.json"), "r") as directory_file:
            directory_json = json.loads(directory_file.read())
            directory_file.close()
            return directory_json[key]

    def _cache_map(self, bucket_ref, map_name, map_json, json_string):
        """Saves a map to a json file in the self._cache_path directory

        Arguments:
            map_name: Name of the map (cached file is stored under the file name map_name.json)
            json_string: Json string to write to file

        Returns: True if map was successfully cached, and false otherwise
        """
        for arg in [bucket_ref, map_json, json_string]:
            if not isinstance(arg, str):
                print("Cannot cache map because '{}' argument is not a string".format(nameof(arg)))
                return False
        if not self._resolve_cache_folder():
            print("Cannot cache map because cache folder cannot be created at {}".format(self._cache_path))
            return False

        cached_file_path = os.path.join(self._cache_path, bucket_ref, map_json)
        try:
            map_json_split = map_json.split("/")
            map_json_split_idx = 0
            while map_json_split_idx < len(map_json_split) - 1:
                dir_to_check = os.path.join(self._cache_path, bucket_ref, os.path.sep.join(map_json_split[
                                                                                    :map_json_split_idx + 1]))
                if not os.path.exists(dir_to_check):
                    os.mkdir(dir_to_check)
                map_json_split_idx += 1

            with open(cached_file_path, "w") as map_json_file:
                map_json_file.write(json_string)
                map_json_file.close()

            self._append_to_cache_directory(os.path.basename(map_json), map_name)
            print("Successfully cached {}".format(cached_file_path))
            return True
        except Exception as ex:
            print("Could not cache map {} due to error: {}".format(map_json, ex))
            return False

    def _resolve_cache_folder(self):
        """Returns true if the cache folder exists, and attempts to create a new one if there is none.

        Two subdirectories named after the relevant bucket parents are also created.

        Returns:
            True if the cache folder at self._cache_path exists after this function returns; False returned otherwise
        """
        processed_path = os.path.join(self._cache_path, GraphHandler._processed_maps_bucket_ref)
        unprocessed_path = os.path.join(self._cache_path, GraphHandler._unprocessed_maps_bucket_ref)
        for path in [self._cache_path, processed_path, unprocessed_path]:
            if os.path.exists(path):
                continue
            try:
                os.mkdir(path)
            except Exception as ex:
                print("Could not create a cache directory at {} due to error: {}".format(path, ex))
                return False

        directory_path = os.path.join(self._cache_path, "directory.json")
        if not os.path.exists(directory_path):
            try:
                with open(os.path.join(self._cache_path, "directory.json"), "w") as directory_file:
                    directory_file.write(json.dumps({}))
                    directory_file.close()
                return True
            except Exception as ex:
                print("Could not create {} file due to error: {}".format(directory_path, ex))
        else:
            return True

    def _unprocessed_maps_callback(self, m):
        if type(m.data) == str:
            # A single new map just got added
            self._firebase_get_unprocessed_map(m.path.lstrip('/'), m.data)
        elif type(m.data) == dict:
            # This will be a dictionary of all the data that is there initially
            for map_name, map_json in m.data.items():
                self._firebase_get_unprocessed_map(map_name, map_json)

    def _optimize_map(self, x, tune_weights=False, visualize=False):
        test_graph = convert_json_sba.as_graph(x)
        test_graph.weights = GraphHandler.weights_dict[self.selected_weights]

        # Load these weights into the graph
        test_graph.update_edges()
        test_graph.generate_unoptimized_graph()

        # Commented out: unused
        # all_tags_original = graph_utils.get_tags_all_position_estimate(test_graph)

        starting_map = graph_utils.optimizer_to_map(
            test_graph.vertices, test_graph.unoptimized_graph, is_sparse_bundle_adjustment=True)
        original_tag_verts = GraphHandler.locations_from_transforms(starting_map['tags'])
        if tune_weights:
            test_graph.expetation_maximization_once()
            print("tuned weights", test_graph.weights)

        # Create the g2o object and optimize
        test_graph.generate_unoptimized_graph()
        test_graph.optimize_graph()

        # Change vertex estimates based off the optimized graph
        test_graph.update_vertices()

        prior_map = graph_utils.optimizer_to_map(
            test_graph.vertices, test_graph.unoptimized_graph)
        resulting_map = graph_utils.optimizer_to_map(
            test_graph.vertices,
            test_graph.optimized_graph,
            is_sparse_bundle_adjustment=True)
        prior_locations = GraphHandler.locations_from_transforms(prior_map['locations'])
        locations = GraphHandler.locations_from_transforms(resulting_map['locations'])

        tag_verts = GraphHandler.locations_from_transforms(resulting_map['tags'])
        tagpoint_positions = resulting_map['tagpoints']
        waypoint_verts = resulting_map['waypoints']
        if visualize:
            f = plt.figure()
            ax = f.add_subplot(111, projection='3d')
            plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], '.', c='b', label='Odom Vertices')
            plt.plot(prior_locations[:, 0], prior_locations[:, 1], prior_locations[:, 2], '.', c='g',
                     label='Prior Odom Vertices')
            plt.plot(original_tag_verts[:, 0], original_tag_verts[:, 1], original_tag_verts[:, 2], 'o', c='c',
                     label='Tag Vertices Original')
            plt.plot(tag_verts[:, 0], tag_verts[:, 1], tag_verts[:, 2], 'o', c='r', label='Tag Vertices')
            for tag_vert in tag_verts:
                R = Quaternion(tag_vert[3:-1]).rotation_matrix()
                axis_to_color = ['r', 'g', 'b']
                for axis_id in range(3):
                    ax.quiver(tag_vert[0], tag_vert[1], tag_vert[2], R[0, axis_id], R[1, axis_id],
                              R[2, axis_id], length=1, color=axis_to_color[axis_id])
            plt.plot(tagpoint_positions[:, 0], tagpoint_positions[:, 1], tagpoint_positions[:, 2], '.', c='m',
                     label='Tag Corners')
            for vert in tag_verts:
                ax.text(vert[0], vert[1], vert[2], str(int(vert[-1])), color='black')
            plt.plot(waypoint_verts[1][:, 0], waypoint_verts[1][:, 1], waypoint_verts[1][:, 2], 'o', c='y',
                     label='Waypoint Vertices')
            for vert_idx in range(len(waypoint_verts[0])):
                vert = waypoint_verts[1][vert_idx]
                waypoint_name = waypoint_verts[0][vert_idx]['name']
                ax.text(vert[0], vert[1], vert[2], waypoint_name, color='black')

            # plt.plot(all_tags[:, 0], all_tags[:, 1], all_tags[:, 2], '.', c='g', label='All Tag Edges')
            # plt.plot(all_tags_original[:, 0], all_tags_original[:, 1], all_tags_original[:, 2], '.', c='m',
            #          label='All Tag Edges Original')

            # Commented-out: unused
            # all_tags = graph_utils.get_tags_all_position_estimate(test_graph)
            # tag_edge_std_dev_before_and_after = compare_std_dev(all_tags, all_tags_original)

            tag_vertex_shift = original_tag_verts - tag_verts
            print("tag_vertex_shift", tag_vertex_shift)
            plt.legend()
            GraphHandler.axis_equal(ax)
            plt.show()
        return tag_verts, locations, waypoint_verts

    # -- Static Methods --

    @staticmethod
    def locations_from_transforms(locations):
        for i in range(locations.shape[0]):
            locations[i, :7] = SE3Quat(locations[i, :7]).inverse().to_vector()
        return locations

    @staticmethod
    def axis_equal(ax):
        """
        Create cubic bounding box to simulate equal aspect ratio
        """
        axis_range_from_limits = lambda limits: limits[1] - limits[0]
        max_range = np.array([axis_range_from_limits(ax.get_xlim()), axis_range_from_limits(ax.get_ylim()),
                              axis_range_from_limits(ax.get_zlim())]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
                ax.get_xlim()[1] + ax.get_xlim()[0])
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
                ax.get_ylim()[1] + ax.get_ylim()[0])
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
                ax.get_zlim()[1] + ax.get_zlim()[0])

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    @staticmethod
    def compare_std_dev(all_tags, all_tags_original):
        return {int(tag_id): (np.std(all_tags_original[all_tags_original[:, -1] == tag_id, :-1], axis=0),
                              np.std(all_tags[all_tags[:, -1] == tag_id, :-1], axis=0)) for tag_id in
                np.unique(all_tags[:, -1])}

    @staticmethod
    def make_processed_map_JSON(tag_locations, odom_locations, waypoint_locations):
        tag_vertex_map = map(lambda curr_tag: {
            'translation': {'x': curr_tag[0], 'y': curr_tag[1], 'z': curr_tag[2]},
            'rotation': {'x': curr_tag[3],
                         'y': curr_tag[4],
                         'z': curr_tag[5],
                         'w': curr_tag[6]},
            'id': int(curr_tag[7])}, tag_locations)
        odom_vertex_map = map(lambda curr_odom: {
            'translation': {'x': curr_odom[0], 'y': curr_odom[1],
                            'z': curr_odom[2]},
            'rotation': {'x': curr_odom[3],
                         'y': curr_odom[4],
                         'z': curr_odom[5],
                         'w': curr_odom[6]},
            'poseId': int(curr_odom[8])}, odom_locations)
        waypoint_vertex_map = map(lambda idx: {
            'translation': {'x': waypoint_locations[1][idx][0],
                            'y': waypoint_locations[1][idx][1],
                            'z': waypoint_locations[1][idx][2]},
            'rotation': {'x': waypoint_locations[1][idx][3],
                         'y': waypoint_locations[1][idx][4],
                         'z': waypoint_locations[1][idx][5],
                         'w': waypoint_locations[1][idx][6]},
            'id': waypoint_locations[0][idx]['name']},
                                  range(len(waypoint_locations[0])))
        return json.dumps({'tag_vertices': list(tag_vertex_map),
                           'odometry_vertices': list(odom_vertex_map),
                           'waypoints_vertices': list(waypoint_vertex_map)})


def make_parser():
    """Makes an argument parser object for this program

    Returns:
        Argument parser
    """
    parser = argparse.ArgumentParser(description="Acquire (from cache or Firebase) graphs, run optimization, and plot")
    parser.add_argument(
        "-p",
        type=str,
        help="Pattern to match to graph names; matching graph names in cache are optimized and plotted (e.g., "
             "'-g *Living_Room*' will plot any cached map with 'Living_Room' in its name); if no pattern is specified, "
             "then all cached maps are plotted and optimized (default pattern is '*'). The cache directory is searched "
             "recursively, and '**/' is automatically prepended to the pattern"
    )
    parser.add_argument(
        "-f",
        action="store_true",
        help="Acquire maps from firebase and overwrite existing cache. Mutually exclusive with the rest of the options."
    )
    parser.add_argument(
        "-F",
        action="store_true",
        help="Upload optimized graphs to firebase"
    )
    return parser


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.f and (args.p or args.F):
        print("Option in addition to -f specified, but -f optoin is mutually exclusive with other options due to the "
              "asynchronous nature of Firebase updating.")
        exit()

    # Fetch the service account key JSON file contents
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    graph_handler = GraphHandler("sensible_default_weights", cred)

    if args.f:
        graph_handler.firebase_listen_unprocessed_maps()
    else:
        if args.p:
            map_pattern = args.p
        else:
            map_pattern = "*"

        graph_handler.process_maps(map_pattern, args.F)
