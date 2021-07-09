#!/usr/bin/env python3
"""
Contains the GraphManager class. For the command line utility that makes use of it, see graph_manager_user.py. The
graph_optimization_analysis.ipynb notebook also makes use of this class.

Author: Duncan Mazza
"""

from __future__ import annotations

import copy
import glob
import json
import os
from typing import *
from threading import Semaphore, Thread
from threading import Timer

import firebase_admin
import matplotlib.pyplot as plt
import numpy as np
from firebase_admin import db
from firebase_admin import storage
from g2o import Quaternion, SE3Quat, SparseOptimizer
from geneticalgorithm import geneticalgorithm as ga
from varname import nameof

import as_graph
import graph_utils
from graph_utils import occam_room_tags
from graph import Graph
from graph_vertex_edge_classes import Vertex, VertexType


class GraphManager:
    """Class that manages graphs by interfacing with firebase, keeping a cache of data downloaded from firebase, and
    providing methods wrapping graph optimization and plotting capabilities.

    Class Attributes:
        _weights_dict (Dict[str, np.ndarray]): Maps descriptive names of weight vectors to the corresponding weight
         vector, Higher values in the vector indicate greater noise (note: the uncertainty estimates of translation 
         seem to be pretty over optimistic, hence the large correction here) for the orientation
        _app_initialize_dict (Dict[str, str]): Used for initializing the app attribute
        _unprocessed_listen_to (str): Simultaneously specifies database reference to listen to in the firebase_listen
         method and the cache location of any maps associated with that database reference.
        _processed_upload_to (str): Simultaneously specifies Firebase bucket path to upload processed graphs to and the
         cache location of processed graphs.
        _comparison_graph1_subgraph_weights (List[str]): A list that contains a subset of the keys in
         _weights_dict; the keys identify the different weights vectors applied to the first subgraph when the
         compare_weights method is invoked.
        _initialized_app: Set to true when the Firebase app is initialized and is used to only let the app be
         initialized on the first invocation of the GraphManager constructor.
        _app (firebase_admin.App): App initialized with a service account, granting admin privileges

    Attributes:
        _bucket: Handle to the Google Cloud Storage bucket
        _db_ref: Database reference representing the node as specified by the GraphManager._unprocessed_listen_to
         class attribute _selected_weights (np.ndarray): Vector selected from the GraphManager._weights_dict
        _cache_path: String representing the absolute path to the cache folder. The cache path is evaluated to
         always be located at <path to this file>.cache/
    """

    # Importance is set to e^{-weight}
    ordered_weights_dict_keys: List[str] = [
        "sensible_default_weights",
        "trust_odom",
        "trust_tags",
        "genetic_results",
        "best_sweep",
        "comparison_baseline"
    ]
    _weights_dict: Dict[str, Dict[str, np.ndarray]] = {
        "sensible_default_weights": graph_utils.normalize_weights({
            'odometry': np.array([-6., -6., -6., -6., -6., -6.]),
            'tag_sba': np.array([18, 18]),
            'tag': np.array([18, 18, 0, 0, 0, 0]),
            'dummy': np.array([-1, 1e2, -1]),
        }),
        "trust_odom": graph_utils.normalize_weights({
            'odometry': np.array([-3., -3., -3., -3., -3., -3.]),
            'tag_sba': np.array([10.6, 10.6]),
            'tag': np.array([10.6, 10.6, 10.6, 10.6, 10.6, 10.6]),
            'dummy': graph_utils.default_dummy_weights,
        }),
        "trust_tags": graph_utils.normalize_weights({
            'odometry': np.array([10, 10, 10, 10, 10, 10]),
            'tag_sba': np.array([-10.6, -10.6]),
            'tag': np.array([-10.6, -10.6, -10.6, -10.6, -10.6, -10.6]),
            'dummy': graph_utils.default_dummy_weights,
        }),
        "genetic_results": graph_utils.normalize_weights({  # only used for SBA - no non-SBA tag weights
            'odometry': np.array([9.25, -7.96, -1.27, 7.71, -1.7, -0.08]),
            'tag_sba': np.array([9.91, 8.88]),
            'dummy': graph_utils.default_dummy_weights,
        }),
        "best_sweep": graph_utils.weights_from_ratio(np.exp(9.9)),
        "comparison_baseline": graph_utils.normalize_weights({
            'odometry': np.ones(6),
            'tag_sba': np.ones(2),
            'tag': np.ones(6),
            'dummy': graph_utils.default_dummy_weights,
        })
    }
    _comparison_graph1_subgraph_weights: List[str] = ["sensible_default_weights", "trust_odom", "trust_tags",
                                                      "genetic_results", "best_sweep"]

    _app_initialize_dict: Dict[str, str] = {
        "databaseURL": "https://invisible-map-sandbox.firebaseio.com/",
        "storageBucket": "invisible-map.appspot.com"
    }
    _initialized_app: bool = False

    _unprocessed_listen_to: str = "unprocessed_maps"
    _processed_upload_to: str = "TestProcessed"

    class MapInfo:
        """Container for identifying information for a graph (useful for caching process)

        Attributes:
            map_name (str): Specifies the child of the "maps" database reference to upload the optimized
             graph to; also passed as the map_name argument to the _cache_map method
            map_json (str): String corresponding to both the bucket blob name of the map and the path to cache the
             map relative to parent_folder
            map_dct (dict): String of json containing graph
        """
        def __init__(self, map_name: str, map_json: str, map_dct: Dict = None):
            self.map_name: str = str(map_name)
            self.map_dct: Union[dict, str] = dict(map_dct) if map_dct is not None else {}
            self.map_json: str = str(map_json)

    def __init__(self, weights_specifier: int, firebase_creds: firebase_admin.credentials.Certificate,
                 pso: int = 0):
        """Initializes GraphManager instance (only populates instance attributes)

        Args:
             weights_specifier (int): Used as the key to access the corresponding value in
             GraphManager._weights_dict (integer is mapped to the key with the GraphManager.ordered_weights_dict_keys
              list).
             firebase_creds (firebase_admin.credentials.Certificate): Firebase credentials to pass as the first
              argument to firebase_admin.initialize_app(cred, ...)
             pso (int): Integer corresponding to the enum value in as_graph.PrescalingOptEnum which selects the
              type of prescaling weights used in non-SBA optimizations
        """
        if not GraphManager._initialized_app:
            GraphManager._app = firebase_admin.initialize_app(firebase_creds, GraphManager._app_initialize_dict)
            GraphManager._initialized_app = True

        self._bucket = storage.bucket(app=GraphManager._app)
        self._db_ref = db.reference("/" + GraphManager._unprocessed_listen_to)

        self._pso = as_graph.PrescalingOptEnum.get_by_value(pso)
        self._selected_weights: str = GraphManager.ordered_weights_dict_keys[weights_specifier]
        self._cache_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".cache")

        # Thread-related attributes for firebase_listen invocation (instantiation here is arbitrary)
        self._listen_kill_timer: Timer = Timer(0, lambda x: x)
        self._firebase_listen_sem: Semaphore = Semaphore()
        self._timer_mutex: Semaphore = Semaphore()
        self._firebase_listen_max_wait: int = 0

    def firebase_listen_in_thread(self):
        self._db_ref.listen(self._df_listen_callback)

    def firebase_listen(self, max_wait: int = 3) -> None:
        """Invokes the listen method of the _db_ref attribute and provides the _df_listen_callback method as the
        callback function argument.

        This function is multi-threaded: the database listening happens in a new thread, and the parent thread blocks on
        its child"s completion.

        Args:
            max_wait: The maximum amount of time in seconds to wait after receiving a response before terminating the
                      database listening and un-blocking the parent thread.
        """
        self._firebase_listen_max_wait = max_wait
        self._firebase_listen_sem = Semaphore(0)
        self._timer_mutex = Semaphore(1)
        self._listen_kill_timer = Timer(self._firebase_listen_max_wait, self._firebase_listen_sem.release)
        self._listen_kill_timer.start()
        thread_obj = Thread(target=self.firebase_listen_in_thread)
        thread_obj.start()
        self._firebase_listen_sem.acquire()
        thread_obj.join()
        print("Finished listening to Firebase")

    def process_maps(self, pattern: str, visualize: bool = True, upload: bool = False, compare: bool = False,
                     new_pso: Union[None, int] = None, new_weights_specifier: Union[None, int] = None,
                     fixed_vertices: Union[VertexType, Tuple[VertexType]] = ()) -> None:
        """Invokes optimization and plotting routines for any cached graphs matching the specified pattern.

        The _resolve_cache_dir method is first called, then the glob package is used to find matching files.
        Matching maps' json strings are loaded, parsed, and provided to the _process_map method. If an exception is
        raised in the process of loading a map or processing it, it is caught and its details are printed to the
        command line.

        Additionally, save the optimized json in <cache directory>/GraphManager._processed_upload_to.

        Args:
            pattern: Pattern to find matching cached graphs (which are stored as .json files. The cache
            pattern: Pattern to find matching cached graphs (which are stored as .json files. The cache
             directory (specified by the _cache_path attribute) is searched recursively
            visualize: Value passed as the visualize argument to the invocation of the _process_map method.
            upload: Value passed as the upload argument to the invocation of the _process_map method.
            compare: If true, run the routine for comparing graph optimization (invokes the compare_weights
             method).
            new_pso: If not None, then it overrides what was specified by the constructor's pso argument (and changes
             the corresponding _pso instance attribute).
            new_weights_specifier: If not none, then it overrides what was specified by the constructor's
             weights_specifier argument (and changes the corresponding _selected_weights instance attribute).
            fixed_vertices: Parameter to pass to as_graph
        """
        if new_pso is not None:
            self._pso = as_graph.PrescalingOptEnum.get_by_value(new_pso)
        if new_weights_specifier is not None:
            self._selected_weights: str = GraphManager.ordered_weights_dict_keys[new_weights_specifier]

        if len(pattern) == 0:
            print("Empty pattern provided; no maps will be processed")
            return

        self._resolve_cache_dir()
        matching_maps = glob.glob(os.path.join(self._cache_path, pattern), recursive=True)

        if len(matching_maps) == 0:
            print("No maps matching pattern {} in recursive search of {}".format(pattern, self._cache_path))
            return

        for map_json_abs_path in matching_maps:
            if os.path.isdir(map_json_abs_path):
                continue  # Ignore directories

            print("\n---- Attempting to process map {} ----".format(map_json_abs_path))
            map_info = self._map_info_from_path(map_json_abs_path)

            if compare:
                if upload:
                    print("Warning: Ignoring True upload argument because comparing graphs")
                self.compare_weights(map_info, visualize)
            else:
                graph = as_graph.as_graph(map_info.map_dct, fixed_vertices=fixed_vertices, prescaling_opt=self._pso)

                graph_plot_title = None
                chi2_plot_title = None
                if visualize:
                    graph_plot_title = "Optimization results for map: {}".format(map_info.map_name)
                    chi2_plot_title = "Odom. node incident edges chi2 values for map: {}".format(map_info.map_name)

                tag_locations, odom_locations, waypoint_locations, opt_chi2, adj_chi2, visible_tags_count = \
                    self._optimize_graph(graph,
                                         tune_weights=False,
                                         visualize=visualize,
                                         weights_key=None,
                                         graph_plot_title=graph_plot_title,
                                         chi2_plot_title=chi2_plot_title)
                processed_map_json = GraphManager.make_processed_map_JSON(tag_locations, odom_locations,
                                                                          waypoint_locations, adj_chi2,
                                                                          visible_tags_count)
                print("Processed map: {}".format(map_info.map_name))
                if upload:
                    self._upload(map_info, processed_map_json)
                    print("Uploaded processed map: {}".format(map_info.map_name))

                self._cache_map(GraphManager._processed_upload_to, map_info, processed_map_json)

    def compare_weights(self, map_info: GraphManager.MapInfo, visualize=True) -> None:
        """Invocation results in the weights comparison routine.

        Iterate through the different weight vectors (using the iter_weights variable) and, for each, do the
        following:
        1. Acquire two sub-graphs: one from the first half of the ordered odometry nodes (called g1sg) and one from the
           other half (called g2sg); note that g2sg is created from the as_graph.as_graph function with the
           fix_tag_vertices as True, whereas g1sg is created with fix_tag_vertices as False.
        2. Optimize the g1sg with the iter_weights, then transfer the estimated locations of its tag vertices to the
           g2sg. The assumption is that a majority - if not all - tag vertices are present in both sub-graphs; the
           number of instances where this is not true is tallied, and warning messages are printed accordingly.
        3. g2sg is then optimized with the self.selected_weights attribute selecting its weights (as opposed to
           g1sg which is optimized using the weights selected by iter_weights)
        The results of the comparison are then printed.

        Args:
            map_info (GraphManager.MapInfo): Map to use for weights comparison
            visualize (bool): Used as the visualize argument for the _process_map method invocation.
        """
        results = "\n### Results ###\n\n"
        g1sg, g2sg = self.create_graphs_for_chi2_comparison(map_info.map_dct)

        missing_vertex_count = 0
        for graph1_sg_vert in g1sg.get_tag_verts():
            if not g2sg.vertices.__contains__(graph1_sg_vert):
                missing_vertex_count += 1
        if missing_vertex_count > 0:
            print("Warning: {} {} present in first subgraph that are not present in the second subgraph ("
                  "{} ignored)".format(missing_vertex_count, "vertices" if missing_vertex_count > 1 else
                                       "vertex", "these were" if missing_vertex_count > 1 else "this was"))

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

        # After iterating through the different weights, the results of the comparison are printed.
        for iter_weights in GraphManager._comparison_graph1_subgraph_weights:
            print("\n-- Processing sub-graph without tags fixed, using weights set: {} --".format(iter_weights))
            if visualize:
                g1sg_plot_title = "Optimization results for 1st sub-graph from map: {} (weights = {})".format(
                    map_info.map_name, iter_weights)
                g1sg_chi2_plot_title = "Odom. node incident edges' chi2 values for 1st sub-graph from map: {} (" \
                                       "weights = {})".format(map_info.map_name, iter_weights)
            else:
                g1sg_plot_title = None
                g1sg_chi2_plot_title = None

            g1sg_tag_locs, g1sg_odom_locs, g1sg_waypoint_locs, g1sg_chi_sqr, g1sg_odom_adj_chi2, \
                g1sg_visible_tags_count = self._optimize_graph(g1sg, tune_weights=False, visualize=visualize,
                                                               weights_key=iter_weights,
                                                               graph_plot_title=g1sg_plot_title,
                                                               chi2_plot_title=g1sg_chi2_plot_title)
            processed_map_json_1 = GraphManager.make_processed_map_JSON(g1sg_tag_locs, g1sg_odom_locs,
                                                                        g1sg_waypoint_locs, g1sg_odom_adj_chi2,
                                                                        g1sg_visible_tags_count)
            del g1sg_tag_locs, g1sg_odom_locs, g1sg_waypoint_locs  # No longer needed

            self._cache_map(GraphManager._processed_upload_to, map_info, processed_map_json_1,
                            "-comparison-subgraph-1-with_weights-set{}".format(iter_weights))
            del processed_map_json_1  # No longer needed

            print("\n-- Processing sub-graph with tags fixed using weights set: {} --".format(self._selected_weights))

            # Get optimized tag vertices from g1sg and transfer their estimated positions to g2sg
            for graph1_sg_vert in g1sg.get_tag_verts():
                if g2sg.vertices.__contains__(graph1_sg_vert):
                    g2sg.vertices[graph1_sg_vert].estimate = g1sg.vertices[graph1_sg_vert].estimate

            if visualize:
                g2sg_plot_title = "Optimization results for 2nd sub-graph from map: {} (weights = {})".format(
                    map_info.map_name, self._selected_weights)
                g2sg_chi2_plot_title = "Odom. node incident edges chi2 values for 2nd sub-graph from  map: {} (" \
                                       "weights = {}))".format(map_info.map_name, self._selected_weights)
            else:
                g2sg_plot_title = None
                g2sg_chi2_plot_title = None

            g2sg_tag_locs, g2sg_odom_locs, g2sg_waypoint_locs, g2sg_chi_sqr, g2sg_odom_adj_chi2, \
                g2sg_visible_tags_count = self._optimize_graph(g2sg, tune_weights=False, visualize=visualize,
                                                               weights_key=None, graph_plot_title=g2sg_plot_title,
                                                               chi2_plot_title=g2sg_chi2_plot_title)
            processed_map_json_2 = GraphManager.make_processed_map_JSON(g2sg_tag_locs, g2sg_odom_locs,
                                                                        g2sg_waypoint_locs, g2sg_odom_adj_chi2,
                                                                        g2sg_visible_tags_count)
            del g2sg_tag_locs, g2sg_odom_locs, g2sg_waypoint_locs  # No longer needed

            self._cache_map(GraphManager._processed_upload_to, map_info, processed_map_json_2,
                            "-comparison-subgraph-2-with_weights-set{}".format(self._selected_weights))
            del processed_map_json_2  # No longer needed

            results += "No fixed tags with weights set {}: chi2 = {}\n" \
                       "Subsequent optimization, fixed tags with weights set {}: chi2 = {}\n" \
                       "Abs(delta chi2): {}\n\n".format(iter_weights, g1sg_chi_sqr, self._selected_weights,
                                                        g2sg_chi_sqr, abs(g1sg_chi_sqr - g2sg_chi_sqr))
        print(results)

    def optimize_weights(self, map_json_path: str, verbose: bool = True) -> np.ndarray:
        """
        Determines the best weights to optimize a graph with

        Args:
            map_json_path: the path to the json containing the unprocessed map information
            verbose (bool): whether to provide output for the chi2 calculation

        Returns:
            A list of the best weights
        """
        map_dct = self._map_info_from_path(map_json_path).map_dct
        graph = as_graph.as_graph(map_dct)
        # sg1, sg2 = self.create_graphs_for_chi2_comparison(map_dct)
        model = ga(function=lambda X: self.get_ground_truth_from_graph(X, graph, occam_room_tags, verbose),
                   dimension=8, variable_type='real', variable_boundaries=np.array([[-10, 10]] * 8),
                   algorithm_parameters={'max_num_iteration': 2000,
                                         'population_size': 50,
                                         'mutation_probability': 0.1,
                                         'elit_ratio': 0.01,
                                         'crossover_probability': 0.5,
                                         'parents_portion': 0.3,
                                         'crossover_type': 'uniform',
                                         'max_iteration_without_improv': None})
        model.run()
        return model.report

    def sweep_weights(self, map_json_path: str, two_d: bool = True, bounds: Tuple[float, float] = (-10., 10.),
                      step: float = 0.5, verbose: bool = False, visualize: bool = True) -> np.ndarray:
        """
        Sweeps a set of weights, returning the resulting chi2 values from each

        Args:
            map_json_path (str): the path to the json containing the map data to optimize on
            two_d (bool): whether to assume all weights for a certain edge are the same
            bounds (tuple): the lower and upper limits of which to sweep, inclusive
            step (float): the step size to use between sweeps
            verbose (bool): whether to print out the chi2 values
            visualize (bool): whether to display the visualization plot. If not two_d, this will be ignored

        Returns:
            An ndarray, where each axis is a weight and each value is the resulting chi2. Note that the indexes will
                start at 0 with a step size of 1 regardless of actual bounds and step size
        """
        map_dct = self._map_info_from_path(map_json_path).map_dct
        #sg1, sg2 = self.create_graphs_for_chi2_comparison(map_dct)
        graph = as_graph.as_graph(map_dct)
        sweep = np.arange(bounds[0], bounds[1] + step, step)
        dimensions = 2 if two_d else 8
        metrics = self._sweep_weights(graph, occam_room_tags, sweep, dimensions, verbose=verbose)

        if two_d and visualize:
            graph_utils.plot_metrics(sweep, metrics)
        if verbose:
            best_metric = metrics.min()
            best_weights = [sweep[i[0]] for i in np.where(metrics == best_metric)]
            print(f'\nBEST METRIC: {best_weights}: {best_metric}')
        return metrics

    def get_optimized_graph_info(self, graph: Graph, weights: Union[int, float, str, Dict[str, np.ndarray], np.ndarray,
                                                                    None] = None, verbose: bool = False,
                                 vertex_types: List[VertexType] = [VertexType.TAG]) -> Tuple[float, Dict[int, Vertex]]:
        """
        Finds the chi2 and vertex locations of the optimized graph without changing the graph itself
        """

        # Load in new weights and update graph
        optimizer = self.get_optimizer(graph, weights)

        # Find info
        chi2 = Graph.check_optimized_edges(optimizer, verbose=verbose)
        vertices = {uid: Vertex(graph.vertices[uid].mode, optimizer.vertex(uid).estimate().vector(),
                                graph.vertices[uid].fixed, graph.vertices[uid].meta_data)
                    for uid in optimizer.vertices() if vertex_types is None or graph.vertices[uid].mode in vertex_types}
        return chi2, vertices

    def get_chi2_from_subgraphs(self, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray]],
                                subgraphs: Union[Tuple[Graph, Graph], Graph],
                                comparison_weights: Union[int, str, Dict[str, np.ndarray]] = 'comparison_baseline',
                                verbose: bool = False) -> float:
        if isinstance(subgraphs, Graph):
            subgraphs = self.create_graphs_for_chi2_comparison(subgraphs)
        else:
            self._weights_dict['variable'] = self._weights_to_dict(weights)
        _, vertices = self.get_optimized_graph_info(subgraphs[0], weights='variable', verbose=verbose)
        for uid, vertex in vertices.items():
            if subgraphs[1].vertices.__contains__(uid):
                subgraphs[1].vertices[uid].estimate = vertex.estimate
        return self.get_optimized_graph_info(subgraphs[1], weights=self.ordered_weights_dict_keys[comparison_weights] if
                                             isinstance(comparison_weights, int) else comparison_weights,
                                             verbose=verbose)[0]

    def get_chi2_by_edge_from_subgraphs(self, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray]],
                                        subgraphs: Union[Tuple[Graph, Graph], Dict],
                                        comparison_weights: Union[int, str, Dict[str, np.ndarray]]
                                        = 'comparison_baseline', verbose: bool = False) -> Dict[str, Dict[str, float]]:
        if isinstance(subgraphs, Dict):
            subgraphs = self.create_graphs_for_chi2_comparison(subgraphs)
        self._weights_dict['variable'] = self._weights_to_dict(weights)
        _, vertices = self.get_optimized_graph_info(subgraphs[0], weights='variable', verbose=verbose)
        for uid, vertex in vertices.items():
            if subgraphs[1].vertices.__contains__(uid):
                subgraphs[1].vertices[uid].estimate = vertex.estimate

        return subgraphs[1].get_chi2_by_edge_type(self.get_optimizer(subgraphs[1], comparison_weights), verbose=verbose)

    def get_ground_truth_from_graph(self, weights: Union[str, Dict[str, np.ndarray], np.ndarray], graph: Graph,
                                    ground_truth_tags: np.ndarray, verbose: bool = False) -> float:
        if isinstance(weights, str):
            weight_name = weights
        else:
            weight_name = 'variable'
            self._weights_dict[weight_name] = weights if isinstance(weights, dict) else\
                graph_utils.weight_dict_from_array(weights)
        _, vertices = self.get_optimized_graph_info(graph, weights=weight_name)
        optimized_tag_verts = np.zeros((len(vertices), 7))
        for vertex in vertices.values():
            estimate = vertex.estimate
            optimized_tag_verts[vertex.meta_data['tag_id']] = \
                (SE3Quat([0, 0, -1, 0, 0, 0, 1]) * SE3Quat(estimate)).inverse().to_vector()
        metric = GraphManager.ground_truth_metric(optimized_tag_verts, ground_truth_tags,
                                                  verbose=verbose)
        if verbose:
            print(metric)
        return metric

    def create_graphs_for_chi2_comparison(self, graph: Dict) -> Tuple[Graph, Graph]:
        """
        Creates then splits a graph in half, as required for weight comparison

        Specifically, this will create the graph based off the information in dct with the given prescaling option. It will
        then exactly halve this graph's vertices into two graphs. The first will allows the tag vertices to vary, while the
        second does not.

        Args:
            graph (Dict): A dictionary containing the unprocessed data to create the graph

        Returns:
            A tuple of 2 graphs, an even split of graph, as described above
        """
        graph1 = as_graph.as_graph(graph, prescaling_opt=self._pso)
        graph2 = as_graph.as_graph(graph, fixed_vertices=VertexType.TAG, prescaling_opt=self._pso)
        dummy_nodes = [0, 0]
        for vertex in graph1.vertices.values():
            if vertex.mode == VertexType.DUMMY:
                dummy_nodes[0] += 1
        for vertex in graph2.vertices.values():
            if vertex.mode == VertexType.DUMMY:
                dummy_nodes[1] += 1
        print(f'Dummy nodes: {dummy_nodes}')
        ordered_odom_edges = graph1.get_ordered_odometry_edges()[0]
        start_uid = graph1.edges[ordered_odom_edges[0]].startuid
        middle_uid_lower = graph1.edges[ordered_odom_edges[len(ordered_odom_edges) // 2]].startuid
        middle_uid_upper = graph1.edges[ordered_odom_edges[len(ordered_odom_edges) // 2]].enduid
        end_uid = graph1.edges[ordered_odom_edges[-1]].enduid

        print(f'start: {start_uid} mid_lower: {middle_uid_lower} mid_upper: {middle_uid_upper} end: {end_uid} total: {len(graph1.vertices)}')

        g1sg = graph1.get_subgraph(start_vertex_uid=start_uid, end_vertex_uid=middle_uid_lower)
        g2sg = graph2.get_subgraph(start_vertex_uid=middle_uid_upper, end_vertex_uid=end_uid)

        return g1sg, g2sg

    def get_optimizer(self, graph: Graph, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray], None])\
            -> SparseOptimizer:
        """
        Returns the optimized g20.SparseOptimizer for the given graph with the given weights, or the graph's default
        weights if no weights are given.
        """
        if weights is not None:
            graph.weights = self._weights_to_dict(weights)
            graph.update_edges()

        optimizer = graph.graph_to_optimizer()
        optimizer.initialize_optimization()
        optimizer.optimize(1024)
        return optimizer

    # -- Private Methods --
    def _weights_to_dict(self, weights: Union[int, float, str, np.ndarray, Dict[str, np.ndarray], None]):
        """
        Converts each representation of weights to a weight dictionary
        """
        if isinstance(weights, int):
            return self._weights_dict[self.ordered_weights_dict_keys[weights]]
        elif isinstance(weights, float):
            return graph_utils.weights_from_ratio(weights)
        elif isinstance(weights, str):
            return self._weights_dict[weights]
        elif isinstance(weights, np.ndarray):
            return graph_utils.weight_dict_from_array(weights)
        elif isinstance(weights, dict):
            return weights
        else:
            return self._selected_weights

    def _map_info_from_path(self, map_json_path: str) -> MapInfo:
        map_json_abs_path = os.path.join(self._cache_path, map_json_path)
        with open(map_json_abs_path, "r") as json_string_file:
            json_string = json_string_file.read()
            json_string_file.close()

        map_json = os.path.sep.join(map_json_abs_path.split(os.path.sep)[len(self._cache_path.split(
            os.path.sep)) + 1:])
        map_dct = json.loads(json_string)
        map_name = self._read_cache_directory(os.path.basename(map_json))
        return GraphManager.MapInfo(map_name, map_name, map_dct)

    def _sweep_weights(self, graph: Graph, ground_truth_tags: np.ndarray, sweep: np.ndarray, dimensions: int,
                       verbose: bool = False, _cur_weights: np.ndarray = np.asarray([])) -> np.ndarray:
        """
        Sweeps the weights with the current chi2 algorithm evaluated on the given map

        Args:
            graph (Graph): the graph for weight comparison
            ground_truth_tags (ndarray): An array of SE3Quats, the poses of the ground truth
            sweep (ndarray): a 1D array containing the values to sweep over
            dimensions (int): the number of dimensions to sweep over (2 or 12)
            verbose (bool): whether to print the chi2 values
            _cur_weights (ndarray): the weights that are already set (do not set manually!)
        """
        if dimensions == 1:
            metrics = np.asarray([])
            for weight in sweep:
                full_weights = np.append(_cur_weights, weight)
                if verbose:
                    print(f'{full_weights.tolist()}: ', end='')
                metric = self.get_ground_truth_from_graph(full_weights, graph, ground_truth_tags)
                if verbose:
                    print(metric)
                metrics = np.append(metrics, metric)
            return metrics
        else:
            metrics = np.asarray([])
            first_run = True
            for weight in sweep:
                if first_run:
                    metrics = self._sweep_weights(graph, ground_truth_tags, sweep, dimensions - 1, verbose,
                                                np.append(_cur_weights, weight)).reshape(1, -1)
                    first_run = False
                else:
                    metrics = np.concatenate((metrics, self._sweep_weights(graph, ground_truth_tags, sweep, dimensions-1,
                                                                           verbose, np.append(_cur_weights, weight))
                                              .reshape(1, -1)))
            return metrics

    def _firebase_get_unprocessed_map(self, map_name: str, map_json: str) -> bool:
        """Acquires a map from the specified blob and caches it.

        A diagnostic message is printed if the map_json blob name was not found by Firebase.

        Args:
            map_name (str): Value passed as the map_name argument to the _cache_map method; the value of map_name is
             ultimately used for uploading a map to firebase by specifying the child of the 'maps' database reference.
            map_json (str): Value passed as the blob_name argument to the get_blob method of the _bucket
             attribute.

        Returns:
            True if the map was successfully acquired and cached, and false if the map was not found by Firebase
        """
        # Reset the timer
        self._timer_mutex.acquire()
        self._listen_kill_timer.cancel()
        self._listen_kill_timer = Timer(self._firebase_listen_max_wait, self._firebase_listen_sem.release)
        self._listen_kill_timer.start()
        self._timer_mutex.release()

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
        <GraphManager._processed_upload_to>/<processed_map_filename> and updates the appropriate database reference.

        Note that no exception catching is implemented.

        Args:
            map_info (GraphManager.MapInfo): Contains the map name and map json path
            json_string (str): Json string of the map to upload
        """
        processed_map_filename = os.path.basename(map_info.map_json)[:-5] + "_processed.json"
        processed_map_full_path = GraphManager._processed_upload_to + "/" + processed_map_filename
        print("Attempting to upload {} to the bucket blob {}".format(map_info.map_name, processed_map_full_path))
        processed_map_blob = self._bucket.blob(processed_map_full_path)
        processed_map_blob.upload_from_string(json_string)
        print("Successfully uploaded map data for {}".format(map_info.map_name))
        db.reference("maps").child(map_info.map_name).child("map_file").set(processed_map_full_path)
        print("Successfully uploaded database reference maps/{}/map_file to contain the blob path".format(
            map_info.map_name))

    def _append_to_cache_directory(self, key: str, value: str) -> None:
        """Appends the specified key-value pair to the dictionary stored as a json file in
        <cache folder>/directory.json.

        If the key already exists in the dictionary, its value is overwritten. Note that no error handling is
        implemented.

        Args:
            key (str): Key to store value in
            value (str): Value to store under key
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
        """Reads the dictionary stored as a json file in <cache folder>/directory.json and returns the value
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
            map_info (GraphManager.MapInfo): Contains the map name and map json path in the map_name and map_json
             fields respectively. If the last 5 characters of this string do not form the substring ".json",
             then ".json" will be appended automatically.
            json_string (str): The json string that defines the map (this is what is written as the contents of the
             cached map file).
            file_suffix (str): String to append to the file name given by map_info.map_json.

        Returns:
            True if map was successfully cached, and False otherwise

        Raises:
            ValueError: Raised if there is any argument (except file_suffix) that is of an incorrect type
            NotADirectoryError: Raised if _resolve_cache_dir method returns false.
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

        A file named directory.json is also created in the cache folder.

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
        """Callback function used in the firebase_listen method.
        """
        if type(m.data) == str:
            # A single new map just got added
            self._firebase_get_unprocessed_map(m.path.lstrip("/"), m.data)
        elif type(m.data) == dict:
            # This will be a dictionary of all the data that is there initially
            for map_name, map_json in m.data.items():
                self._firebase_get_unprocessed_map(map_name, map_json)

    def _optimize_graph(self, graph: Graph, tune_weights: bool = False, visualize: bool = False, weights_key: \
                        Union[None, str] = None, graph_plot_title: Union[str, None] = None, chi2_plot_title: \
                        Union[str, None] = None) -> Tuple[np.ndarray, np.ndarray, Tuple[List[Dict], np.ndarray],
                                                          float, np.ndarray, np.ndarray]:
        """Optimizes the input graph

        Arguments:
            graph (Graph): A graph instance to optimize. The following methods of this instance are (possibly) called:
             update_edges, generate_unoptimized_graph, get_tags_all_position_estimate, expectation_maximization_once,
             generate_unoptimized_graph, and optimize_graph. Additionally, the weights attribute of this instance is
             set.
            tune_weights (bool): A boolean for whether expectation_maximization_once is called on the graph instance.
            visualize (bool): A boolean for whether the visualize static method of this class is called.
            weights_key (str or None): Specifies the weight vector to set the weights attribute of the graph
             instance to from one of the weight vectors in GraphManager._weights_dict. If weights_key is None,
             then the weight vector corresponding to self._selected_weights is selected; otherwise, the weights_key
             selects the weight vector from the dictionary.
            graph_plot_title (str or None): Plot title argument to pass to the visualization routine.

        Returns:
            A tuple containing:
            - The numpy array of tag vertices from the optimized graph
            - The numpy array of odometry vertices from the optimized graph
            - The numpy array of waypoint vertices from the optimized graph
            - The total chi2 value of the optimized graph as returned by the optimize_graph method of the graph
              instance.
            - A numpy array where each element corresponds to the chi2 value for each odometry node; each chi2
              value is calculated as the sum of chi2 values of the (up to) two incident edges to the odometry node
              that connects it to (up to) two other odometry nodes.
            - A numpy array where each element corresponds to the number of visible tag vertices from the corresponding
              odometry vertices.
        """
        graph.weights = GraphManager._weights_dict[weights_key if isinstance(weights_key, str) else
                                                   self._selected_weights]

        # Load these weights into the graph
        graph.update_edges()
        graph.generate_unoptimized_graph()

        # Commented out: unused
        # all_tags_original = graph.get_tags_all_position_estimate()

        starting_map = graph_utils.optimizer_to_map(graph.vertices, graph.unoptimized_graph,
                                                    is_sparse_bundle_adjustment=self._pso == 0)
        original_tag_verts = graph_utils.locations_from_transforms(starting_map["tags"]) \
            if self._pso == as_graph.PrescalingOptEnum.USE_SBA else starting_map["tags"]

        if tune_weights:
            graph.expectation_maximization_once()
            print("tuned weights", graph.weights)

        opt_chi2 = graph.optimize_graph()

        # Change vertex estimates based off the optimized graph
        graph.update_vertices()

        prior_map = graph_utils.optimizer_to_map_chi2(graph, graph.unoptimized_graph)
        resulting_map = graph_utils.optimizer_to_map_chi2(graph, graph.optimized_graph,
                                                          is_sparse_bundle_adjustment=self._pso == 0)

        odom_chi2_adj_vec: np.ndarray = resulting_map["locationsAdjChi2"]
        visible_tags_count_vec: np.ndarray = resulting_map["visibleTagsCount"]

        prior_locations = graph_utils.locations_from_transforms(prior_map["locations"]) \
            if self._pso == as_graph.PrescalingOptEnum.USE_SBA else prior_map["locations"]
        locations = graph_utils.locations_from_transforms(resulting_map["locations"]) \
            if self._pso == as_graph.PrescalingOptEnum.USE_SBA else resulting_map["locations"]
        tag_verts = graph_utils.locations_from_transforms(resulting_map["tags"]) \
            if self._pso == as_graph.PrescalingOptEnum.USE_SBA else resulting_map["tags"]
        tagpoint_positions = resulting_map["tagpoints"]
        waypoint_verts = tuple(resulting_map["waypoints"])

        if visualize:
            self.plot_optimization_result(locations, prior_locations, tag_verts, tagpoint_positions, waypoint_verts,
<<<<<<< HEAD
                                          original_tag_verts, None, graph_plot_title)
=======
                                          original_tag_verts, None, graph_plot_title, is_sba=self._pso == 0)
>>>>>>> 3dba2b792fcb3b3a483d4a0f583bce3710e6feec
            GraphManager.plot_adj_chi2(resulting_map, chi2_plot_title)

        return tag_verts, locations, tuple(waypoint_verts), opt_chi2, odom_chi2_adj_vec, visible_tags_count_vec

    # -- Static Methods --
    @staticmethod
    def plot_adj_chi2(map_from_opt: Dict, plot_title: Union[str, None] = None):
        locations_chi2_viz_tags = []
        locations_shape = np.shape(map_from_opt["locations"])
        for i in range(locations_shape[0]):
            locations_chi2_viz_tags.append((map_from_opt["locations"][i], map_from_opt["locationsAdjChi2"][i],
                                            map_from_opt["visibleTagsCount"][i]))
        locations_chi2_viz_tags.sort(key=lambda x: x[0][7])  # Sorts by UID, which is at the 7th index

        chi2_values = np.zeros([locations_shape[0], 1])  # Contains adjacent chi2 values
        viz_tags = np.zeros([locations_shape[0], 3])
        odom_uids = np.zeros([locations_shape[0], 1])  # Contains UIDs
        for idx in range(locations_shape[0]):
            chi2_values[idx] = locations_chi2_viz_tags[idx][1]
            odom_uids[idx] = int(locations_chi2_viz_tags[idx][0][7])

            # Each row: UID, chi2_value, and num. viz. tags (only if != 0). As of now, the number of visible tags is
            # ignored when plotting (plot only shows boolean value: whether at least 1 tag vertex is visible)
            num_tag_verts = locations_chi2_viz_tags[idx][2]
            if num_tag_verts != 0:
                viz_tags[idx, :] = np.array([odom_uids[idx], chi2_values[idx], num_tag_verts]).flatten()

        odom_uids.flatten()
        chi2_values.flatten()

        f = plt.figure()
        ax: plt.Axes = f.add_axes([0.1, 0.1, 0.8, 0.7])
        ax.plot(odom_uids, chi2_values)
        ax.scatter(viz_tags[:, 0], viz_tags[:, 1], marker="o", color="red")
        ax.set_xlim(min(odom_uids), max(odom_uids))
        ax.legend(["chi2 value", ">=1 tag vertex visible"])
        ax.set_xlabel("Odometry vertex UID")

        plt.xlabel("Odometry vertex UID")
        if plot_title is not None:
            plt.title(plot_title, wrap=True)
        plt.show()

    @staticmethod
    def ground_truth_metric(optimized_tag_verts: np.ndarray, ground_truth_tags: np.ndarray, verbose: bool = False)\
            -> float:
        """
        Generates a metric to compare the accuracy of a map with the ground truth.

        Calculates the transforms from the anchor tag to each other tag for the optimized and the ground truth tags,
        then compares the transforms and finds the difference in the translation components.

        Args:
            optimized_tag_verts: A n-by-7 numpy.ndarray, where n is the number of tags in the map, and the 7 elements
                represent the translation (xyz) and rotation (quaternion) of the optimized tags.
            ground_truth_tags: A n-by-7 numpy.ndarray, where n is the number of tags in the map, and the 7 elements
                represent the translation (xyz) and rotation (quaternion) of the ground truth tags.
            anchor_tag: An int representing which tag to use as the comparison tag.
            verbose: A boolean representing whether to print the full comparisons for each tag.

        Returns:
            A float representing the average difference in tag positions (translation only) in meters.
        """
        num_tags = optimized_tag_verts.shape[0]
        sum_trans_diffs = np.zeros((num_tags,))
        for anchor_tag in range(num_tags):
            anchor_tag_se3quat = SE3Quat(optimized_tag_verts[anchor_tag])
            to_world = anchor_tag_se3quat * ground_truth_tags[anchor_tag].inverse()
            world_frame_ground_truth = np.asarray([(to_world * tag).to_vector() for tag in ground_truth_tags])[:, :3]
            sum_trans_diffs += np.linalg.norm(world_frame_ground_truth - optimized_tag_verts[:, :3], axis=1)
        avg_trans_diffs = sum_trans_diffs / num_tags
        avg = np.mean(avg_trans_diffs)
        if verbose:
            print(f'Ground truth metric is {avg}')
        # noinspection PyTypeChecker
        return avg

    @staticmethod
    def plot_optimization_result(locations: np.ndarray, prior_locations: np.ndarray, tag_verts: np.ndarray,
                                 tagpoint_positions: np.ndarray, waypoint_verts: Tuple[List, np.ndarray],
                                 original_tag_verts: Union[None, np.ndarray] = None,
                                 ground_truth_tags: Union[None, np.ndarray] = None,
                                 plot_title: Union[str, None] = None,
                                 is_sba: bool = False) -> None:
        """Visualization used during the optimization routine.
        """
        f = plt.figure()
        ax = f.add_axes([0.1, 0.1, 0.6, 0.75], projection="3d")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(120, -90)

        plt.plot(prior_locations[:, 0], prior_locations[:, 1], prior_locations[:, 2], "-", c="g",
                 label="Prior Odom Vertices")
        plt.plot(locations[:, 0], locations[:, 1], locations[:, 2], "-", c="b", label="Odom Vertices")

        if original_tag_verts is not None:
            if is_sba:
                for i, tag_vertex in enumerate(original_tag_verts):
                    original_tag_verts[i] = np.append((SE3Quat([0, 0, -1, 0, 0, 0, 1]) * SE3Quat(tag_vertex[:-1]).inverse())
                                                      .inverse().to_vector(), tag_vertex[-1])
            plt.plot(original_tag_verts[:, 0], original_tag_verts[:, 1], original_tag_verts[:, 2], "o", c="c",
                     label="Tag Vertices Original")

        # Fix the 1 meter offset on the tag anchors
        if is_sba:
            for i, tag_vertex in enumerate(tag_verts):
                tag_verts[i] = np.append((SE3Quat([0, 0, -1, 0, 0, 0, 1]) * SE3Quat(tag_vertex[:-1]).inverse()).inverse()
                                         .to_vector(), tag_vertex[-1])

        if ground_truth_tags is not None:
            tag_list = tag_verts.tolist()
            tag_list.sort(key=lambda x: x[-1])
            ordered_tags = np.asarray([tag[0:-1] for tag in tag_list])
            anchor_tag = 0
            anchor_tag_se3quat = SE3Quat(ordered_tags[anchor_tag])
            to_world = anchor_tag_se3quat * ground_truth_tags[anchor_tag].inverse()
            world_frame_ground_truth = np.asarray([(to_world * tag).to_vector() for tag in ground_truth_tags])

            print(f"\nAverage translation difference: {GraphManager.ground_truth_metric(ordered_tags, ground_truth_tags, True)}\n")

            plt.plot(world_frame_ground_truth[:, 0], world_frame_ground_truth[:, 1], world_frame_ground_truth[:, 2],
                     'o', c='k', label=f'Actual Tags')
            for i, tag in enumerate(world_frame_ground_truth):
                ax.text(tag[0], tag[1], tag[2], str(i), c='k')

        plt.plot(tag_verts[:, 0], tag_verts[:, 1], tag_verts[:, 2], "o", c="r", label="Tag Vertices")
        for tag_vert in tag_verts:
            R = Quaternion(tag_vert[3:-1]).rotation_matrix()
            axis_to_color = ["r", "g", "b"]
            for axis_id in range(3):
                ax.quiver(tag_vert[0], tag_vert[1], tag_vert[2], R[0, axis_id], R[1, axis_id],
                          R[2, axis_id], length=1, color=axis_to_color[axis_id])

        plt.plot(tagpoint_positions[:, 0], tagpoint_positions[:, 1], tagpoint_positions[:, 2], ".", c="m",
                 label="Tag Corners")

        for vert in tag_verts:
            ax.text(vert[0], vert[1], vert[2], str(int(vert[-1])), color="black")

        plt.plot(waypoint_verts[1][:, 0], waypoint_verts[1][:, 1], waypoint_verts[1][:, 2], "o", c="y",
                 label="Waypoint Vertices")

        for vert_idx in range(len(waypoint_verts[0])):
            vert = waypoint_verts[1][vert_idx]
            waypoint_name = waypoint_verts[0][vert_idx]["name"]
            ax.text(vert[0], vert[1], vert[2], waypoint_name, color="black")

        # plt.plot(all_tags[:, 0], all_tags[:, 1], all_tags[:, 2], '.', c='g', label='All Tag Edges')
        # plt.plot(all_tags_original[:, 0], all_tags_original[:, 1], all_tags_original[:, 2], '.', c='m',
        #          label='All Tag Edges Original')

        # Commented-out: unused
        # all_tags = graph_utils.get_tags_all_position_estimate(graph)
        # tag_edge_std_dev_before_and_after = compare_std_dev(all_tags, all_tags_original)

        tag_vertex_shift = original_tag_verts - tag_verts
        #print("tag_vertex_shift", tag_vertex_shift)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize="small")
        GraphManager.axis_equal(ax)
        plt.gcf().set_dpi(300)
        if isinstance(plot_title, str):
            plt.title(plot_title, wrap=True)
        plt.show()

    @staticmethod
    def axis_equal(ax):
        """Create cubic bounding box to simulate equal aspect ratio
        """
        axis_range_from_limits = lambda limits: limits[1] - limits[0]
        max_range = np.max(np.array([axis_range_from_limits(ax.get_xlim()), axis_range_from_limits(ax.get_ylim()),
                                    axis_range_from_limits(ax.get_zlim())]))
        Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * \
            (ax.get_xlim()[1] + ax.get_xlim()[0])
        Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * \
            (ax.get_ylim()[1] + ax.get_ylim()[0])
        Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * \
            (ax.get_zlim()[1] + ax.get_zlim()[0])

        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
            ax.plot([xb], [yb], [zb], "w")

    @staticmethod
    def compare_std_dev(all_tags, all_tags_original):
        return {int(tag_id): (np.std(all_tags_original[all_tags_original[:, -1] == tag_id, :-1], axis=0),
                              np.std(all_tags[all_tags[:, -1] == tag_id, :-1], axis=0)) for tag_id in
                np.unique(all_tags[:, -1])}

    @staticmethod
    def make_processed_map_JSON(tag_locations: np.ndarray, odom_locations: np.ndarray, waypoint_locations: Tuple[
        List[Dict], np.ndarray], adj_chi2_arr: Union[None, np.ndarray] = None,
                                visible_tags_count: Union[None, np.ndarray] = None) -> str:
        if (visible_tags_count is None) ^ (visible_tags_count is None):
            print("visible_tags_count and adj_chi2_arr arguments must both be None or non-None")

        tag_vertex_map = map(
            lambda curr_tag: {
                "translation": {"x": curr_tag[0], "y": curr_tag[1], "z": curr_tag[2]},
                "rotation": {"x": curr_tag[3],
                             "y": curr_tag[4],
                             "z": curr_tag[5],
                             "w": curr_tag[6]},
                "id": int(curr_tag[7])
            }, tag_locations
        )

        if adj_chi2_arr is None:
            odom_vertex_map = map(
                lambda curr_odom: {
                    "translation": {"x": curr_odom[0], "y": curr_odom[1],
                                    "z": curr_odom[2]},
                    "rotation": {"x": curr_odom[3],
                                 "y": curr_odom[4],
                                 "z": curr_odom[5],
                                 "w": curr_odom[6]},
                    "poseId": int(curr_odom[8]),
                }, odom_locations
            )
        else:
            odom_locations_with_chi2_and_viz_tags = np.concatenate([odom_locations, adj_chi2_arr, visible_tags_count],
                                                                   axis=1)
            odom_vertex_map = map(
                lambda curr_odom: {
                    "translation": {"x": curr_odom[0], "y": curr_odom[1],
                                    "z": curr_odom[2]},
                    "rotation": {"x": curr_odom[3],
                                 "y": curr_odom[4],
                                 "z": curr_odom[5],
                                 "w": curr_odom[6]},
                    "poseId": int(curr_odom[8]),
                    "adjChi2": curr_odom[9],
                    "vizTags": curr_odom[10]
                }, odom_locations_with_chi2_and_viz_tags
            )
        waypoint_vertex_map = map(
            lambda idx: {
                "translation": {"x": waypoint_locations[1][idx][0],
                                "y": waypoint_locations[1][idx][1],
                                "z": waypoint_locations[1][idx][2]},
                "rotation": {"x": waypoint_locations[1][idx][3],
                             "y": waypoint_locations[1][idx][4],
                             "z": waypoint_locations[1][idx][5],
                             "w": waypoint_locations[1][idx][6]},
                "id": waypoint_locations[0][idx]["name"]
            }, range(len(waypoint_locations[0]))
        )
        return json.dumps({"tag_vertices": list(tag_vertex_map),
                           "odometry_vertices": list(odom_vertex_map),
                           "waypoints_vertices": list(waypoint_vertex_map)}, indent=2)


if __name__ == "__main__":
    print("The command line functionality using the GraphManager class is in graph_manager_user.py. The "
          "graph_optimization_analysis.ipynb notebook also makes use of this class.")
