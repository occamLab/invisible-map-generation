"""
Script that makes use of the GraphManager class.

Print the usage instructions:
>> python3 optimize_graphs_and_manage_cache.py -h

Example usage that listens to the unprocessed maps' database reference:
>> python3 optimize_graphs_and_manage_cache.py -f

Example usage that optimizes and plots all graphs matching the pattern specified by the -p flag:
>> python3 optimize_graphs_and_manage_cache.py -p "unprocessed_maps/**/*Living Room*"

Notes:
- This script was adapted from the script test_firebase_sba as of commit 74891577511869f7cd3c4743c1e69fb5145f81e0
- The maps that are *processed* and cached are of a different format than the unprocessed graphs and cannot be-loaded
  for further processing.
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import numpy as np
from typing import Dict, Callable, Iterable, Any, Tuple, List

from map_processing import PrescalingOptEnum, VertexType
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import OComputeInfParams, GTDataSet, OConfig
from map_processing.graph_opt_hl_interface import holistic_optimize, WEIGHTS_DICT, WeightSpecifier
from map_processing.graph_opt_utils import rotation_metric
from map_processing.sweep import sweep_params

SWEEP_CONFIG: Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.OConfigEnum.ODOM_TAG_RATIO: (np.linspace, [1, 1, 1]),
    OConfig.OConfigEnum.LIN_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.ANG_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.GRAV_MAG: (np.linspace, [1, 1, 1]),
}

def find_optimal_map(cms: CacheManagerSingleton, to_fix: List[int], compute_inf_params: OComputeInfParams,
                     weights: int = 5, remove_bad_tag: bool = False, sweep: bool = False, sba: int = 0,
                     visualize: bool = False, map_pattern: str = "", sbea: bool = False, compare: bool = False,
                     upload: bool = False, num_processes: int = 1):
    """
    Based on parameters specified in the main script, the optimal map will be found and returned

    Args:
        cms: A CacheManagerSingleton that is used to find maps from the cache
        to_fix: A List of ints that represent the type of vertexes to fix
        compute_inf_params: A OComputeInfParams that includes the parameters
        weights: An int representing the index of the WEIGHTS_DICT to apply
        remove_bad_tag: A Boolean representing whether bad tags should be removed by running sba_evaluator or not
        sweep: A Boolean representing whether a parameter sweeps should be run or not
        sba: An Integer representing whether the map should be optimized using SBA or not (0 means yes, 1 means no)
        visualize: A Boolean representing whether the map should be visualized
        map_pattern: A String representing the pattern of the map to search for
        sbea: A Boolean representing whether sba should be applied or not
        compare: A Boolean representing whether compare should be applied or not
        upload: A Boolean representing whether the files are to be uploaded to FireBase or not
        num_processes: An int representing the number of processes to run the sweep with

    Returns:

    """
    matching_maps = cms.find_maps(map_pattern, search_only_unprocessed=True)
    if len(matching_maps) == 0:
        print(f"No matches for {map_pattern} in recursive search of {CacheManagerSingleton.CACHE_PATH}")
        return None

    # Remove tag observations that are bad
    if remove_bad_tag:
        this_path = cms.find_maps(map_pattern, search_only_unprocessed=True, paths=True)
        sba.throw_out_bad_tags(this_path[0])

    # Run optimizer
    for map_info in matching_maps:
        gt_data = cms.find_ground_truth_data_from_map_info(map_info)

        # Run sweep if specified
        if sweep:
            sweep_params(mi=map_info, ground_truth_data=gt_data,
                         base_oconfig=OConfig(is_sba=sba == PrescalingOptEnum.USE_SBA.value,
                                              compute_inf_params=compute_inf_params),
                         sweep_config=SWEEP_CONFIG, ordered_sweep_config_keys=[key for key in SWEEP_CONFIG.keys()],
                         verbose=True, generate_plot=True, show_plot=visualize, num_processes=num_processes,
                         no_sba_baseline=False)

        # If no sweep, then run basic optimization
        oconfig = OConfig(is_sba=sba == 0, weights=WEIGHTS_DICT[WeightSpecifier(weights)],
                          scale_by_edge_amount=sbea, compute_inf_params=compute_inf_params)
        fixed_vertices = set()
        for tag_type in to_fix:
            fixed_vertices.add(VertexType(tag_type))
        opt_result = holistic_optimize(
            map_info=map_info, pso=PrescalingOptEnum(sba), oconfig=oconfig,
            fixed_vertices=fixed_vertices, verbose=True, visualize=visualize, compare=compare, upload=upload,
            gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data) if gt_data is not None else None)

        # Get rotational metrics
        pre_optimized_tags = opt_result.map_pre.tags
        optimized_tags = opt_result.map_opt.tags
        rot_metric, max_rot_diff, max_rot_diff_idx = rotation_metric(pre_optimized_tags, optimized_tags)
        print(f"Rotation metric: {rot_metric}")
        print(f"Maximum rotation: {max_rot_diff} (tag id: {max_rot_diff_idx})")




