"""
This script exists to validate whether there exists a monotonic and positive relationship between metrics based on a
subset of analogous parameters between data set generation and optimization. Specifically, this script looks for a
monotonic and positive relationship between the ratio of linear to angular velocity variance for a generated data set
and the ratio of the optimal linear and angular velocity variance values for optimizing the data set.
"""

import os
import sys

import numpy as np
import datetime
from typing import List, Dict, Callable, Iterable, Any, Tuple

from map_processing import TIME_FORMAT
from map_processing.data_models import GenerateParams, UGDataSet, OConfig, OSweepResults, OMultiSweepResult
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.graph_generator import GraphGenerator
from map_processing.sweep import sweep_params
from g2o import SparseOptimizer

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

NUM_PROCESSES = 10
HOLD_RVERT_AT = 1e-5
RATIO_XZ_TO_Y_LIN_VEL_VAR = 10
NUM_REPEAT_GENERATE = 1
PATH_FROM = "duncan-occam-room-10-1-21-2-48 26773176629225.json"


ALT_OPT_CONFIG: Dict[OConfig.AltOConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    OConfig.AltOConfigEnum.LIN_TO_ANG_VEL_VAR: (np.linspace, [1e-3, 1e3, 10]),
    OConfig.AltOConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-6, 1e-3, 10])
}

ALT_GENERATE_CONFIG: Dict[GenerateParams.AltGenerateParamsEnum, Tuple[Callable, Iterable[Any]]] = {
    GenerateParams.AltGenerateParamsEnum.OBS_NOISE_VAR: (np.linspace, [0.1, 1, 10]),
    GenerateParams.AltGenerateParamsEnum.LIN_TO_ANG_VEL_VAR: (np.geomspace, [14.177, 14.177, 1])
}

BASE_GENERATE_PRAMS = GenerateParams(dataset_name=f"generated_from_{PATH_FROM.strip('.json')}")
BASE_OCONFIG = OConfig(is_sba=True)


# noinspection DuplicatedCode
def validate_analogous_params(
        data_set_generate_from: UGDataSet,
        alt_opt_config: Dict[OConfig.AltOConfigEnum, Tuple[Callable, Iterable[Any]]],
        alt_gen_config: Dict[GenerateParams.AltGenerateParamsEnum, Tuple[Callable, Iterable[Any]]],
        hold_rvert_at: float, ratio_xz_to_y_lin_vel_var: float, base_generate_params: GenerateParams,
        num_repeat_generate: int, verbose: bool = False, ntsba: bool = False,
        ograph: SparseOptimizer = None) -> OMultiSweepResult:
    # Evaluate the functions and their arguments stored as keys and values, respectively
    alt_param_multiplicands_for_generation: Dict[GenerateParams.AltGenerateParamsEnum, np.ndarray] = {}
    for generate_key, generate_value in alt_gen_config.items():
        if isinstance(generate_value, np.ndarray):
            alt_param_multiplicands_for_generation[generate_key] = generate_value
        else:
            alt_param_multiplicands_for_generation[generate_key] = generate_value[0](*generate_value[1])
    alt_param_multiplicands_for_opt: Dict[OConfig.AltOConfigEnum, np.ndarray] = {}
    for opt_key, opt_value in alt_opt_config.items():
        if isinstance(opt_value, np.ndarray):
            alt_param_multiplicands_for_opt[opt_key] = opt_value
        else:
            alt_param_multiplicands_for_opt[opt_key] = opt_value[0](*opt_value[1])

    # Get the data necessary to batch generate data sets and configure the parameter sweep over each of them
    _, generate_params_list = GenerateParams.alt_generate_params_generator(
        alt_param_multiplicands=alt_param_multiplicands_for_generation, base_generate_params=base_generate_params,
        hold_rvert_at=hold_rvert_at, ratio_xz_to_y_lin_vel_var=ratio_xz_to_y_lin_vel_var)
    if len(set(generate_params_list)) != len(generate_params_list):
        raise Exception(f"Non-unique set of {GenerateParams.__name__} objects created")
    param_multiplicands_for_opt = OConfig.alt_oconfig_generator_param_multiplicands(
        alt_param_multiplicands=alt_param_multiplicands_for_opt)

    # For each set of data generation parameters, generate NUM_REPEAT_GENERATE data sets and, for each of them, run an
    # optimization parameter sweep.
    num_to_generate = len(generate_params_list) * num_repeat_generate
    max_num_prefix_digits = int(np.log10(num_to_generate))
    now_str = datetime.datetime.now().strftime(TIME_FORMAT)
    sweep_results_list: List[List[OSweepResults]] = []
    gen_param_idx = -1
    for generate_param in generate_params_list:
        sweep_results_list_per_gen_params: List[OSweepResults] = []
        for _ in range(num_repeat_generate):
            gen_param_idx += 1
            num_zeros_to_prefix = max_num_prefix_digits - int(np.log10(gen_param_idx + 1))  # For printing messages
            generate_param.map_id = f"batch_generated_{now_str}_{'0' * num_zeros_to_prefix}{gen_param_idx + 1}"
            gg = GraphGenerator(path_from=data_set_generate_from, gen_params=generate_param)
            mi = gg.export_to_map_processing_cache()
            gt_data_set = CacheManagerSingleton.find_ground_truth_data_from_map_info(mi)

            sweep_results_list_per_gen_params.append(sweep_params(
                mi=mi,
                ground_truth_data=gt_data_set,
                base_oconfig=OConfig(is_sba=True),
                sweep_config=param_multiplicands_for_opt,
                ordered_sweep_config_keys=sorted(list(param_multiplicands_for_opt.keys())),
                num_processes=NUM_PROCESSES,
                verbose=True,
                cache_results=False,
                ntsba=ntsba,
                ograph=ograph))
            if verbose:
                print(f"Generated and swept optimizations for {'0' * num_zeros_to_prefix}{gen_param_idx + 1}/"
                      f"{num_to_generate}: {generate_param.dataset_name}")
        sweep_results_list.append(sweep_results_list_per_gen_params)

    return OMultiSweepResult(generate_params_list=generate_params_list, sweep_results_list=sweep_results_list)


# noinspection DuplicatedCode
if __name__ == "__main__":
    # Acquire the data set from which the generated maps are parsed
    matching_maps = CacheManagerSingleton.find_maps(PATH_FROM, search_only_unprocessed=True)
    if len(matching_maps) == 0:
        print(f"No matches for {PATH_FROM} in recursive search of {CacheManagerSingleton.CACHE_PATH}")
        exit(0)
    elif len(matching_maps) > 1:
        print(f"More than one match for {PATH_FROM} found in recursive search of {CacheManagerSingleton.CACHE_PATH}. "
              f"Will not batch-generate unless only one path is found.")
        exit(0)
    # Acquire the data set from which the generated data sets are derived from
    map_info = matching_maps.pop()
    data_set = UGDataSet(**map_info.map_dct)

    msr = validate_analogous_params(
        data_set_generate_from=data_set, alt_opt_config=ALT_OPT_CONFIG, alt_gen_config=ALT_GENERATE_CONFIG,
        hold_rvert_at=HOLD_RVERT_AT, ratio_xz_to_y_lin_vel_var=RATIO_XZ_TO_Y_LIN_VEL_VAR,
        base_generate_params=BASE_GENERATE_PRAMS, num_repeat_generate=NUM_REPEAT_GENERATE, verbose=True, ntsba=ntsba,
        ograph=ograph)
    results_file_name = f"analogous_parameter_sweep_{datetime.datetime.now().strftime(TIME_FORMAT)}"
    CacheManagerSingleton.cache_sweep_results(msr, results_file_name)

    # Un-comment the type of plot to be generated
    # fig = msr.plot_scatter_of_lin_to_ang_var_ratios()
    fig = msr.plot_scatter_of_obs_noise_params()

    fig.savefig(os.path.join(CacheManagerSingleton.SWEEP_RESULTS_PATH, results_file_name + ".png"), dpi=500)
