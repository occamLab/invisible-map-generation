#!/usr/bin/env python3
"""
Contains the GraphManager class and a main routine that makes use of it.

Print the usage instructions:
>> python3 GraphManager.py -h

Example usage that listens to the unprocessed maps database reference:
>> python3 GraphManager.py -f

Example usage that optimizes and plots all graphs matching the pattern specified by the -p flag:
>> python3 GraphManager.py -p "unprocessed_maps/**/*Living Room*"

Notes:
- This script was adapted from the script test_firebase_sba as of commit 74891577511869f7cd3c4743c1e69fb5145f81e0
- The maps that are *processed* and cached are of a different format than the unprocessed graphs and cannot be-loaded
  for further processing.

Author: Duncan Mazza
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import *

import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
from g2o import Quaternion
from varname import nameof

import convert_json_sba
import graph_utils
from graph import Graph
from json import tool


class GraphManager:
    """Class that manages graphs by interfacing with firebase, keeping a cache of data downloaded from firebase, and
    providing methods wrapping graph optimization and plotting capabilities.

    Class Attributes:
        _weights_dict (Dict[str, np.ndarray]): Maps descriptive names of weight vectors to the corresponding weight
         vector, Higher values in the vector indicate greater noise (note: the uncertainty estimates of translation 
         seem to be pretty over optimistic, hence the large correction here) for the orientation
        _app_initialize_dict (Dict[str, str]): Used for initializing the `app` attribute
        _unprocessed_listen_to (str): Simultaneously specifies database reference to listen to in the `firebase_listen`
         method and the cache location of any maps associated with that database reference.
        _processed_upload_to (str): Simultaneously specifies Firebase bucket path to upload processed graphs to and the
         cache location of processed graphs.
        _comparison_graph1_subgraph_weights (List[str]): A list that contains a subset of the keys in
        `_weights_dict`; the keys identify the different weights vectors applied to the first subgraph when the
        `_compare_weights` method is invoked.

    Attributes:
        _app (firebase_admin.App): App initialized with a service account, granting admin privileges
        _bucket: Handle to the Google Cloud Storage bucket
        _db_ref: Database reference representing the node as specified by the `GraphManager._unprocessed_listen_to`
         class attribute _selected_weights (np.ndarray): Vector selected from the `GraphManager._weights_dict`
        _cache_path (str): String representing the absolute path to the cache folder. The cache path is evaluated to
         always be located at `<path to this file>.cache/`
    """
    _weights_dict = {
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

    _comparison_graph1_subgraph_weights = ["sensible_default_weights", "trust_odom", "trust_tags"]

    _app_initialize_dict = {
        'databaseURL': 'https://invisible-map-sandbox.firebaseio.com/',
        'storageBucket': 'invisible-map.appspot.com'
    }

    _unprocessed_listen_to = "unprocessed_maps"
    _processed_upload_to = "TestProcessed"

    class MapInfo:
        """Container for identifying information for a graph (useful for caching process)

        Attributes:
            map_name (str): Specifies the child of the 'maps' database reference to upload the optimized
             graph to; also passed as the `map_name` argument to the `_cache_map` method
            map_json (str): String corresponding to both the bucket blob name of the map and the path to cache the
             map relative to `parent_folder`
            map_dct (dict): String of json containing graph
        """

        def __init__(self, map_name: str, map_json: str, map_dct: Dict = None):
            self.map_name: str = str(map_name)
            self.map_dct: Union[dict, str] = dict(map_dct)
            self.map_json: str = str(map_json)

    def __init__(self, weights_specifier: str, firebase_creds: firebase_admin.credentials.Certificate):
        """Initializes GraphManager instance (only populates instance attributes)

        Args:
             weights_specifier (str): Used as the key to access the corresponding value in `GraphManager._weights_dict`
             firebase_creds (firebase_admin.credentials.Certificate): Firebase credentials to pass as the first
             argument to `firebase_admin.initialize_app(cred, ...)`
        """
        self._app = firebase_admin.initialize_app(firebase_creds, GraphManager._app_initialize_dict)
        self._bucket = storage.bucket(app=self._app)
        self._db_ref = db.reference("/" + GraphManager._unprocessed_listen_to)
        self._selected_weights = str(weights_specifier)
        self._cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache")

    def firebase_listen(self) -> None:
        """Invokes the `listen` method of the `_db_ref` attribute and provides the `_df_listen_callback` method as the
        callback function argument.
        """
        self._db_ref.listen(self._df_listen_callback)

    def process_maps(self, pattern: str, visualize: bool = True, upload: bool = False, compare: bool = False) -> None:
        """Invokes optimization and plotting routines for any cached graphs matching the specified pattern.

        The `_resolve_cache_dir` method is first called, then the `glob` package is used to find matching files.
        Matching maps' json strings are loaded, parsed, and provided to the `_process_map` method. If an exception is
        raised in the process of loading a map or processing it, it is caught and its details are printed to the
        command line.

        Additionally, save the optimized json in `<cache directory>/GraphManager._processed_upload_to`.

        Args:
            pattern (str): Pattern to find matching cached graphs (which are stored as `.json` files. The cache
             directory (specified by the `_cache_path` attribute) is searched recursively
            visualize (bool): Value passed as the visualize argument to the invocation of the `_process_map` method.
            upload (bool): Value passed as the upload argument to the invocation of the `_process_map` method.
            compare (bool): If true, run the routine for comparing graph optimization (invokes the `_compare_weights`
             method)
        """
        self._resolve_cache_dir()
        matching_maps = glob.glob(os.path.join(self._cache_path, pattern), recursive=True)

        if len(matching_maps) == 0:
            print("No maps matching pattern {} in recursive search of {}".format(pattern, self._cache_path))
            return

        for map_json_abs_path in matching_maps:
            if os.path.isdir(map_json_abs_path):
                continue  # Ignore directories

            print("\n---- Attempting to process map {} ----".format(map_json_abs_path))
            with open(os.path.join(self._cache_path, map_json_abs_path), "r") as json_string_file:
                json_string = json_string_file.read()
                json_string_file.close()
            map_json = os.path.sep.join(map_json_abs_path.split(os.path.sep)[len(self._cache_path.split(
                os.path.sep)) + 1:])
            map_dct = json.loads(json_string)
            map_name = self._read_cache_directory(os.path.basename(map_json))

            map_info = GraphManager.MapInfo(map_name, map_name, map_dct)

            if compare:
                if upload:
                    print("Warning: Ignoring True upload argument because comparing graphs")
                self._compare_weights(map_info, visualize)
            else:
                graph = convert_json_sba.as_graph(map_info.map_dct)

                graph_plot_title = None
                chi2_plot_title = None
                if visualize:
                    graph_plot_title = "Optimization results for map: {}".format(map_info.map_name)
                    chi2_plot_title = "Odom. node incident edges chi2 values for map: {}".format(map_info.map_name)

                tag_locations, odom_locations, waypoint_locations, opt_chi2, _ = \
                    self._optimize_graph(graph, False, visualize, weights_key=None, graph_plot_title=graph_plot_title,
                                         chi2_plot_title=chi2_plot_title)
                processed_map_json = GraphManager.make_processed_map_JSON(tag_locations, odom_locations,
                                                                          waypoint_locations)
                print("Processed map: {}".format(map_info.map_name))
                if upload:
                    self._upload(map_info, processed_map_json)
                    print("Uploaded processed map: {}".format(map_info.map_name))

                self._cache_map(GraphManager._processed_upload_to, map_info, processed_map_json)

    # -- Private Methods --

    def _compare_weights(self, map_info: GraphManager.MapInfo, visualize=True) -> None:
        """Invocation results in the weights comparison routine.

        Iterate through the different weight vectors (using the iter_weights variable) and, for each, do the
        following:
        1. Acquire two sub-graphs: one from the first half of the ordered odometry nodes (called g1sg) and one from the
           other half (called g2sg); note that g2sg is created from the convert_json_sba.as_graph method with the
           fix_tag_vertices as True, whereas g1sg is created with fix_tag_vertices as False.
        2. Optimize the g1sg with the iter_weights, then transfer the estimated locations of its tag vertices to the
           g2sg. The assumption is that a majority - if not all - tag vertices are present in both sub-graphs; the
           number of instances where this is not true is tallied, and warning messages are printed accordingly.
        3. g2sg is then optimized with the self.selected_weights attribute selecting its weights (as opposed to
           g1sg which is optimized using the weights selected by iter_weights)
        The results of the comparison are then printed.

        Args:
            map_info (GraphManager.MapInfo): Map to use for weights comparison
            visualize (bool): Used as the visualize argument for the `_process_map` method invocation.
        """
        results = "\n### Results  ###\n"

        # After iterating through the different weights, the results of the comparison are printed.
        for iter_weights in GraphManager._comparison_graph1_subgraph_weights:
            graph1 = convert_json_sba.as_graph(map_info.map_dct, fix_tag_vertices=False)
            ordered_odom_edges = graph1.get_ordered_odometry_edges()[0]
            start_uid = graph1.edges[ordered_odom_edges[0]].startuid
            end_uid = graph1.edges[ordered_odom_edges[-1]].enduid
            floored_middle = (start_uid + end_uid) // 2
            g1sg = graph1.get_subgraph(start_vertex_uid=start_uid, end_vertex_uid=floored_middle)

            print("\n-- Processing sub-graph without tags fixed, using weights set: {} --".format(iter_weights))

            g1sg_plot_title = None
            g1sg_chi2_plot_title = None
            if visualize:
                g1sg_plot_title = "Optimization results for 1st sub-graph\n from map: {}".format(map_info.map_name)
                g1sg_chi2_plot_title = "Odom. node incident edges chi2 values for\n 1st sub-graph from  map: {}".format(
                    map_info.map_name)

            tag_locations, odom_locations, waypoint_locations, pre_fixed_chi_sqr, g1sg_odom_adj_chi2 = \
                self._optimize_graph(g1sg, False, visualize, iter_weights, g1sg_plot_title, g1sg_chi2_plot_title)
            processed_map_json_1 = GraphManager.make_processed_map_JSON(tag_locations, odom_locations,
                                                                        waypoint_locations, g1sg_odom_adj_chi2)

            self._cache_map(GraphManager._processed_upload_to, map_info, processed_map_json_1,
                            "-comparison-subgraph-1-with_weights-set{}".format(iter_weights))

            print("\n-- Processing sub-graph with tags fixed using weights set: {} --".format(self._selected_weights))

            # Get optimized tag vertices from g1sg and transfer their estimated positions to g2sg;
            # check whether there are any vertices present in g1sg that are not present in g2sg
            # (print warning of how many vertices fit this criteria if nonzero)
            graph2 = convert_json_sba.as_graph(map_info.map_dct, fix_tag_vertices=True)
            g2sg = graph2.get_subgraph(start_vertex_uid=floored_middle + 1, end_vertex_uid=end_uid)
            missing_vertex_count = 0
            for graph1_sg_vert in g1sg.get_tag_verts():
                if not g2sg.vertices.__contains__(graph1_sg_vert):
                    missing_vertex_count += 1
                else:
                    g2sg.vertices[graph1_sg_vert].estimate = \
                        g1sg.vertices[graph1_sg_vert].estimate

            if missing_vertex_count > 0:
                print("Warning: {} {} present in first subgraph that are not present in the second subgraph ("
                      "{} ignored)".format(missing_vertex_count, "vertices" if missing_vertex_count > 1 else
                                           "vertex", "these were" if missing_vertex_count > 1 else "this was"))

            # Check whether there are any tag vertices present in g2sg that are not present in the
            # g1sg; for each occurrence of this, delete the vertex from g2sg. After check is
            # complete, print a warning of how many vertices were deleted if nonzero.
            deleted_vertex_count = 0
            for graph2_sg_vert in g2sg.get_tag_verts():
                if not g1sg.vertices.__contains__(graph2_sg_vert):
                    g2sg.delete_tag_vertex(graph2_sg_vert)
                    deleted_vertex_count += 1

            if deleted_vertex_count > 0:
                print("Warning: {} {} present in second subgraph that are not present in the first subgraph ("
                      "{} deleted from the second subgraph)"
                      .format(deleted_vertex_count, "vertices" if deleted_vertex_count > 1 else "vertex",
                              "these were" if deleted_vertex_count > 1 else "this was"))

            g2sg_plot_title = None
            g2sg_chi2_plot_title = None
            if visualize:
                g2sg_plot_title = "Optimization results for 2nd sub-graph\n from map: {}".format(map_info.map_name)
                g2sg_chi2_plot_title = "Odom. node incident edges chi2 values for\n 2nd sub-graph from  map: {}".format(
                    map_info.map_name)

            tag_locations, odom_locations, waypoint_locations, fixed_tag_chi_sqr, g2sg_odom_adj_chi2 = \
                self._optimize_graph(g2sg, False, visualize, weights_key=None, graph_plot_title=g2sg_plot_title,
                                     chi2_plot_title=g2sg_chi2_plot_title)
            processed_map_json_2 = GraphManager.make_processed_map_JSON(tag_locations, odom_locations,
                                                                        waypoint_locations, g2sg_odom_adj_chi2)
            self._cache_map(GraphManager._processed_upload_to, map_info, processed_map_json_2,
                            "-comparison-subgraph-2-with_weights-set{}".format(self._selected_weights))

            results += "Pre-fixed-tags with weights set {}: chi-sqr = {}\n" \
                       "Subsequent optimization, fixed-tags with weights set {}: chi-sqr = {}\n" \
                       "Abs(delta chi-sqr): {}\n\n" \
                .format(iter_weights, pre_fixed_chi_sqr, self._selected_weights, fixed_tag_chi_sqr,
                        abs(pre_fixed_chi_sqr - fixed_tag_chi_sqr))

            # TODO: Sanity check with extreme weights

            # TODO: Visualize the chi2 in the graph plot (e.g., color-code nodes based on chi2 of edges);
            #  maybe also sync up with Jacquie re: her plotting efforts? Add another field to json of the
            #  map
        print(results)

    def _firebase_get_unprocessed_map(self, map_name: str, map_json: str) -> bool:
        """Acquires a map from the specified blob and caches it.

        A diagnostic message is printed if the `map_json` blob name was not found by Firebase.

        Args:
            map_name (str): Value passed as the `map_name` argument to the `_cache_map` method; the value of map_name is
             ultimately used for uploading a map to firebase by specifying the child of the 'maps' database reference.
            map_json (str): Value passed as the `blob_name` argument to the `get_blob` method of the `_bucket`
             attribute.

        Returns:
            True if the map was successfully acquired and cached, and false if the map was not found by Firebase
        """
        map_info = GraphManager.MapInfo(map_name, map_json, None)

        json_blob = self._bucket.get_blob(map_info.map_json)
        if json_blob is not None:
            json_data = json_blob.download_as_bytes()
            json_string = json.loads(json_data)
            self._cache_map(GraphManager._unprocessed_listen_to, map_info, json.dumps(json_string, indent=2))
            return True
        else:
            print("Map '{}' was missing".format(map_info.map_name))
            return False

    def _upload(self, map_info: GraphManager.MapInfo, json_string: str) -> None:
        """Uploads the map json string into the Firebase bucket under the path
        `<GraphManager._processed_upload_to>/<processed_map_filename>` and updates the appropriate database reference.

        Note that no exception catching is implemented.

        Args:
            map_info (GraphManager.MapInfo): Contains the map name and map json path
            json_string (str): Json string of the map to upload
        """
        processed_map_filename = os.path.basename(map_info.map_json)[:-5] + '_processed.json'
        processed_map_full_path = GraphManager._processed_upload_to + "/" + processed_map_filename
        print("Attempting to upload {} to the bucket blob {}".format(map_info.map_name, processed_map_full_path))
        processed_map_blob = self._bucket.blob(processed_map_full_path)
        processed_map_blob.upload_from_string(json_string)
        print("Successfully uploaded map data for {}".format(map_info.map_name))
        db.reference('maps').child(map_info.map_name).child('map_file').set(processed_map_full_path)
        print("Successfully uploaded database reference maps/{}/map_file to contain the blob path".format(
            map_info.map_name))

    def _append_to_cache_directory(self, key: str, value: str) -> None:
        """Appends the specified key-value pair to the dictionary stored as a json file in
        `<cache folder>/directory.json`.

        If the key already exists in the dictionary, its value is overwritten. Note that no error handling is
        implemented.

        Args:
            key (str): Key to store `value` in
            value (str): Value to store under `key`
        """
        directory_json_path = os.path.join(self._cache_path, "directory.json")
        with open(directory_json_path, "r") as directory_file_read:
            directory_json = json.loads(directory_file_read.read())
            directory_file_read.close()
        directory_json[key] = value
        new_directory_json = json.dumps(directory_json, indent=2)
        with open(directory_json_path, "w") as directory_file_write:
            directory_file_write.write(new_directory_json)
            directory_file_write.close()

    def _read_cache_directory(self, key: str) -> str:
        """Reads the dictionary stored as a json file in `<cache folder>/directory.json` and returns the value
        associated with the specified key.

        Note that no error handling is implemented.

        Args:
            key (str): Key to query the dictionary

        Returns:
            Value associated with the key
        """
        with open(os.path.join(self._cache_path, "directory.json"), "r") as directory_file:
            directory_json = json.loads(directory_file.read())
            directory_file.close()
            return directory_json[key]

    def _cache_map(self, parent_folder: str, map_info: GraphManager.MapInfo, json_string: str, file_suffix: Union[
                   str, None] = None) -> bool:
        """Saves a map to a json file in cache directory.

        Catches any exceptions raised when saving the file (exceptions are raised for invalid arguments) and displays an
        appropriate diagnostic message if one is caught. All of the arguments are checked to ensure that they are, in
        fact strings; if any are not, then a diagnostic message is printed and False is returned.

        Arguments:
            parent_folder (str): Specifies the sub-directory of the cache directory that the map is cached in
            map_info (GraphManager.MapInfo): Contains the map name and map json path in the `map_name` and `map_json`
             fields respectively. If the last 5 characters of this string do not form the substring ".json",
             then ".json" will be appended automatically.
            json_string (str): The json string that defines the map (this is what is written as the contents of the
             cached map file).
            file_suffix (str): String to append to the file name given by `map_info.map_json`.

        Returns:
            True if map was successfully cached, and False otherwise

        Raises:
            ValueError: Raised if there is any argument (except `file_suffix`) that is of an incorrect type
            NotADirectoryError: Raised if `_resolve_cache_dir` method returns false.
        """
        if not isinstance(map_info, GraphManager.MapInfo):
            raise ValueError("Cannot cache map because '{}' argument is not a {} instance"
                             .format(nameof(map_info), nameof(GraphManager.MapInfo)))
        for arg in [parent_folder, map_info.map_name, map_info.map_json, json_string]:
            if not isinstance(arg, str):
                raise ValueError("Cannot cache map because '{}' argument is not a string".format(nameof(arg)))

        if not self._resolve_cache_dir():
            raise NotADirectoryError("Cannot cache map because cache folder existence could not be resolved at path {}"
                                     .format(self._cache_path))

        file_suffix_str = (file_suffix if isinstance(file_suffix, str) else "")
        map_json_to_use = str(map_info.map_json)
        if len(map_json_to_use) < 6:
            map_json_to_use += file_suffix_str + ".json"
        else:
            if map_json_to_use[-5:] != ".json":
                map_json_to_use += file_suffix_str + ".json"
            else:
                map_json_to_use = map_json_to_use[:-5] + file_suffix_str + ".json"

        cached_file_path = os.path.join(self._cache_path, parent_folder, map_json_to_use)
        try:
            cache_to = os.path.join(parent_folder, map_json_to_use)
            cache_to_split = cache_to.split(os.path.sep)
            cache_to_split_idx = 0
            while cache_to_split_idx < len(cache_to_split) - 1:
                dir_to_check = os.path.join(self._cache_path, os.path.sep.join(cache_to_split[:cache_to_split_idx + 1]))
                if not os.path.exists(dir_to_check):
                    os.mkdir(dir_to_check)
                cache_to_split_idx += 1

            with open(cached_file_path, "w") as map_json_file:
                map_json_file.write(json_string)
                map_json_file.close()

            self._append_to_cache_directory(os.path.basename(map_json_to_use), map_info.map_name)
            print("Successfully cached {}".format(cached_file_path))
            return True
        except Exception as ex:
            print("Could not cache map {} due to error: {}".format(map_json_to_use, ex))
            return False

    def _resolve_cache_dir(self) -> bool:
        """Returns true if the cache folder exists, and attempts to create a new one if there is none.

        A file named `directory.json` is also created in the cache folder.

        This method catches all exceptions associated with creating new directories/files and displays a corresponding
        diagnostic message.

        Returns:
            True if no exceptions were caught and False otherwise
        """
        if not os.path.exists(self._cache_path):
            try:
                os.mkdir(self._cache_path)
            except Exception as ex:
                print("Could not create a cache directory at {} due to error: {}".format(self._cache_path, ex))
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

    def _df_listen_callback(self, m) -> None:
        """Callback function used in the `firebase_listen` method.
        """
        if type(m.data) == str:
            # A single new map just got added
            self._firebase_get_unprocessed_map(m.path.lstrip('/'), m.data)
        elif type(m.data) == dict:
            # This will be a dictionary of all the data that is there initially
            for map_name, map_json in m.data.items():
                self._firebase_get_unprocessed_map(map_name, map_json)

    def _optimize_graph(self, graph: Graph, tune_weights: bool = False, visualize: bool = False, weights_key: \
                        Union[None, str] = None, graph_plot_title: Union[str, None] = None, chi2_plot_title: \
                        Union[str, None] = None) -> \
            Tuple[np.ndarray, np.ndarray, Tuple[List[Dict], np.ndarray], float, np.ndarray]:
        """Optimizes the input graph

        Arguments:
            graph (Graph): A graph instance to optimize. The following methods of this instance are (possibly) called:
             update_edges, generate_unoptimized_graph, get_tags_all_position_estimate, expectation_maximization_once,
             generate_unoptimized_graph, and optimize_graph. Additionally, the `weights` attribute of this instance is
             set.
            tune_weights (bool): A boolean for whether `expectation_maximization_once` is called on the graph instance.
            visualize (bool): A boolean for whether the `visualize` static method of this class is called.
            weights_key (str or None): Specifies the weight vector to set the `weights` attribute of the graph
             instance to from one of the weight vectors in `GraphManager._weights_dict`. If weights_key is None,
             then the weight vector corresponding to `self._selected_weights` is selected; otherwise, the weights_key
             selects the weight vector from the dictionary.
            graph_plot_title (str or None): Plot title argument to pass to the visualization routine.

        Returns:
            A tuple containing:
            - The numpy array of tag vertices from the optimized graph
            - The numpy array of odometry vertices from the optimized graph
            - The numpy array of waypoint vertices from the optimized graph
            - The total chi2 value of the optimized graph as returned by the `optimize_graph` method of the `graph`
              instance.
            - A vector numpy array where each element corresponds to the chi2 value for each odometry node; each chi2
              value is calculated as the sum of chi2 values of the (up to) two incident edges to the odometry node
              that connects it to (up to) two other odometry nodes.
        """
        graph.weights = GraphManager._weights_dict[weights_key if isinstance(weights_key, str) else
                                                   self._selected_weights]

        # Load these weights into the graph
        graph.update_edges()
        graph.generate_unoptimized_graph()

        # Commented out: unused
        # all_tags_original = graph.get_tags_all_position_estimate()

        starting_map = graph_utils.optimizer_to_map(
            graph.vertices, graph.unoptimized_graph,
            is_sparse_bundle_adjustment=True
        )
        original_tag_verts = graph_utils.locations_from_transforms(starting_map['tags'])

        if tune_weights:
            graph.expectation_maximization_once()
            print("tuned weights", graph.weights)
        # Create the g2o object and optimize
        graph.generate_unoptimized_graph()
        opt_chi2 = graph.optimize_graph()

        # Change vertex estimates based off the optimized graph
        graph.update_vertices()

        prior_map = graph_utils.optimizer_to_map_chi2(graph, graph.unoptimized_graph)
        resulting_map = graph_utils.optimizer_to_map_chi2(graph, graph.optimized_graph,
                                                          is_sparse_bundle_adjustment=True)
        odom_chi2_adj_vec: np.ndarray = resulting_map['locationsAdjChi2']

        prior_locations = graph_utils.locations_from_transforms(prior_map['locations'])
        locations = graph_utils.locations_from_transforms(resulting_map['locations'])
        tag_verts = graph_utils.locations_from_transforms(resulting_map['tags'])
        tagpoint_positions = resulting_map['tagpoints']
        waypoint_verts = resulting_map['waypoints']

        if visualize:
            self.visualize(locations, prior_locations, tag_verts, tagpoint_positions, waypoint_verts,
                           original_tag_verts, graph_plot_title)
            GraphManager.plot_adj_chi2(resulting_map, chi2_plot_title)

        return tag_verts, locations, waypoint_verts, opt_chi2, odom_chi2_adj_vec

    # -- Static Methods --

    @staticmethod
    def plot_adj_chi2(map_from_opt: Dict, plot_title: Union[str, None] = None):
        uids_chi2_comb = []
        for idx, uid in enumerate(map_from_opt['uids']):
            uids_chi2_comb.append((uid, map_from_opt['locationsAdjChi2'][idx]))

        uids_chi2_comb.sort(key=lambda x: x[0])
        y_axis = np.zeros(np.shape(map_from_opt['locationsAdjChi2']))
        for idx in range(len(uids_chi2_comb)):
            y_axis[idx] = uids_chi2_comb[idx][1]

        plt.plot(sorted(map_from_opt['uids']), y_axis + 1)
        plt.xlabel("Odometry vertex UID")
        if plot_title is not None:
            plt.title(plot_title)
        plt.yscale("log")
        plt.ylabel("lg(1 + chi2)")

    @staticmethod
    def visualize(locations: np.ndarray, prior_locations: np.ndarray, tag_verts: np.ndarray, tagpoint_positions: \
                  np.ndarray, waypoint_verts: Tuple[List, np.ndarray], original_tag_verts: Union[None, np.ndarray] \
                  = None, plot_title: Union[str, None] = None) -> None:
        """Visualization used during the optimization routine.
        """
        f = plt.figure()
        ax = f.add_subplot(111, projection='3d')

        plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], '.', c='b', label='Odom Vertices')
        plt.plot(prior_locations[:, 0], prior_locations[:, 1], prior_locations[:, 2], '.', c='g',
                 label='Prior Odom Vertices')

        if original_tag_verts is not None:
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
        # all_tags = graph_utils.get_tags_all_position_estimate(graph)
        # tag_edge_std_dev_before_and_after = compare_std_dev(all_tags, all_tags_original)

        tag_vertex_shift = original_tag_verts - tag_verts
        print("tag_vertex_shift", tag_vertex_shift)
        plt.legend()
        GraphManager.axis_equal(ax)

        if isinstance(plot_title, str):
            plt.title(plot_title)

        plt.show()

    @staticmethod
    def axis_equal(ax):
        """Create cubic bounding box to simulate equal aspect ratio
        """
        axis_range_from_limits = lambda limits: limits[1] - limits[0]
        max_range = np.array([axis_range_from_limits(ax.get_xlim()), axis_range_from_limits(ax.get_ylim()),
                              axis_range_from_limits(ax.get_zlim())]).max()
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * \
            (ax.get_xlim()[1] + ax.get_xlim()[0])
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * \
            (ax.get_ylim()[1] + ax.get_ylim()[0])
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * \
            (ax.get_zlim()[1] + ax.get_zlim()[0])

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], 'w')

    @staticmethod
    def compare_std_dev(all_tags, all_tags_original):
        return {int(tag_id): (np.std(all_tags_original[all_tags_original[:, -1] == tag_id, :-1], axis=0),
                              np.std(all_tags[all_tags[:, -1] == tag_id, :-1], axis=0)) for tag_id in
                np.unique(all_tags[:, -1])}

    @staticmethod
    def make_processed_map_JSON(tag_locations: np.ndarray, odom_locations: np.ndarray, waypoint_locations: Tuple[
                                List[Dict], np.ndarray], adj_chi2_arr: Union[None, np.ndarray] = None) -> str:
        tag_vertex_map = map(
            lambda curr_tag: {
                'translation': {'x': curr_tag[0], 'y': curr_tag[1], 'z': curr_tag[2]},
                'rotation': {'x': curr_tag[3],
                             'y': curr_tag[4],
                             'z': curr_tag[5],
                             'w': curr_tag[6]},
                'id': int(curr_tag[7])
            }, tag_locations
        )

        if adj_chi2_arr is None:
            odom_vertex_map = map(
                lambda curr_odom: {
                    'translation': {'x': curr_odom[0], 'y': curr_odom[1],
                                    'z': curr_odom[2]},
                    'rotation': {'x': curr_odom[3],
                                 'y': curr_odom[4],
                                 'z': curr_odom[5],
                                 'w': curr_odom[6]},
                    'poseId': int(curr_odom[8]),
                }, odom_locations
            )
        else:
            odom_locations_with_chi2 = np.concatenate([odom_locations, adj_chi2_arr], axis=1)
            odom_vertex_map = map(
                lambda curr_odom: {
                    'translation': {'x': curr_odom[0], 'y': curr_odom[1],
                                    'z': curr_odom[2]},
                    'rotation': {'x': curr_odom[3],
                                 'y': curr_odom[4],
                                 'z': curr_odom[5],
                                 'w': curr_odom[6]},
                    'poseId': int(curr_odom[8]),
                    'adjChi2': curr_odom[9]
                }, odom_locations_with_chi2
            )
        waypoint_vertex_map = map(
            lambda idx: {
                'translation': {'x': waypoint_locations[1][idx][0],
                                'y': waypoint_locations[1][idx][1],
                                'z': waypoint_locations[1][idx][2]},
                'rotation': {'x': waypoint_locations[1][idx][3],
                             'y': waypoint_locations[1][idx][4],
                             'z': waypoint_locations[1][idx][5],
                             'w': waypoint_locations[1][idx][6]},
                'id': waypoint_locations[0][idx]['name']
            }, range(len(waypoint_locations[0]))
        )
        return json.dumps({'tag_vertices': list(tag_vertex_map),
                           'odometry_vertices': list(odom_vertex_map),
                           'waypoints_vertices': list(waypoint_vertex_map)}, indent=2)


def make_parser():
    """Makes an argument p object for this program

    Returns:
        Argument p
    """
    p = argparse.ArgumentParser(description="Acquire (from cache or Firebase) graphs, run optimization, and plot")
    p.add_argument(
        "-p",
        type=str,
        help="Pattern to match to graph names; matching graph names in cache are optimized and plotted (e.g., "
             "'-g *Living_Room*' will plot any cached map with 'Living_Room' in its name); if no pattern is specified, "
             "then all cached maps are plotted and optimized (default pattern is '*'). The cache directory is searched "
             "recursively, and '**/' is automatically prepended to the pattern"
    )
    p.add_argument(
        "-f",
        action="store_true",
        help="Acquire maps from Firebase and overwrite existing cache. Mutually exclusive with the rest of the options."
    )
    p.add_argument(
        "-F",
        action="store_true",
        help="Upload any graphs to Firebase that are optimized while this script is running. This option is mutually "
             "exclusive with the -c option."
    )
    p.add_argument(
        "-c",
        action="store_true",
        help="Compare graph optimizations by computing two different optimizations for two sub-graphs of the "
             "specified graph: one where the tag vertices are not fixed, and one where they are. This option is "
             "mutually exclusive with the -F option."
    )
    p.add_argument(
        "-v",
        action="store_true",
        help="Visualize plots"
    )
    return p


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.f and (args.p or args.F or args.c):
        print("Option in addition to -f specified, but -f option is mutually exclusive with other options due to the "
              "asynchronous nature of Firebase updating.")
        exit()

    if args.c and args.F:
        print("Options -c and -F are mutually exclusive; uploading is disabled for graph comparison")
        exit()

    # Fetch the service account key JSON file contents
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    graph_handler = GraphManager("sensible_default_weights", cred)

    if args.f:
        graph_handler.firebase_listen()
        exit()

    map_pattern = args.p if args.p else "*"
    graph_handler.process_maps(map_pattern, visualize=args.v, upload=args.F, compare=args.c)
