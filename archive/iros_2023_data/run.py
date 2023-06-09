import os
import sys
repository_root = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)
from map_processing.sweep import sweep_params
from map_processing.data_models import OComputeInfParams, OConfig
from map_processing.cache_manager import CacheManagerSingleton
from typing import Dict, Callable, Iterable, Any, Tuple
from firebase_admin import credentials
import numpy as np

SBA_SWEEP_CONFIG: Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.OConfigEnum.ODOM_TAG_RATIO: (np.linspace, [1, 1, 1]),
    OConfig.OConfigEnum.LIN_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.ANG_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.GRAV_MAG: (np.linspace, [1, 1, 1]),
}

NO_SBA_SWEEP_CONFIG: Dict[OConfig.OConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.OConfigEnum.ODOM_TAG_RATIO: (np.linspace, [1, 1, 1]),
    OConfig.OConfigEnum.LIN_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.ANG_VEL_VAR: (np.geomspace, [1e-10, 10, 10]),
    OConfig.OConfigEnum.TAG_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-10, 10, 10]),
    # OConfig.OConfigEnum.GRAV_MAG: (np.linspace, [1, 1, 1]),
}

def run_gt_correlation(small_map_name, big_map_name, pso):
    # Fetch the service account key JSON file contents
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_variable is None:
        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
    else:
        cms = CacheManagerSingleton(
            firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
        )
    small_map_info = cms.find_maps(small_map_name, search_restriction=0).pop()
    big_map_info = cms.find_maps(big_map_name, search_restriction=0).pop()

    compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3) * 1.0,
        tag_sba_var=1.0,
        ang_vel_var=1.0,
    )

    gt_dataset = cms.find_ground_truth_data_from_map_info(small_map_info)
    # big_gt_dataset = cms.find_ground_truth_data_from_map_info(big_map_info)
    # if small_gt_dataset != big_gt_dataset:
    #     raise ValueError("GT Data should be the same")
    # gt_dataset = small_gt_dataset
    sweep_config = NO_SBA_SWEEP_CONFIG if pso == 1 else SBA_SWEEP_CONFIG

    print("Running Small Map Sweep")
    small_sweep_result = sweep_params(
        mi=small_map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=12,
        cms=cms,
        simple_metrics=True
    )

    print("Running Big Map Sweep")
    big_sweep_result = sweep_params(
        mi=big_map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=12,
        cms=cms,
        simple_metrics=True
    )
    gt_list = []
    sweep_config = small_sweep_result.sweep_config
    for lin_var in sweep_config['lin_vel_var']:
        for ang_var in sweep_config['ang_vel_var']:
            for tag_var in sweep_config['tag_var']:
                small_gt = small_sweep_result.query_at({"lin_vel_var": lin_var, "ang_vel_var": ang_var, "tag_var": tag_var})
                big_gt = big_sweep_result.query_at({"lin_vel_var": lin_var, "ang_vel_var": ang_var, "tag_var": tag_var})
                gt_list.append([small_gt, big_gt])

    gt_list = np.array(gt_list)
    small_gt_results = gt_list[:,0]
    large_gt_results = gt_list[:,1]
    return small_gt_results, large_gt_results

def run_full_extrapolation(small_map_name, big_map_name, pso):
    # Fetch the service account key JSON file contents
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_variable is None:
        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
    else:
        cms = CacheManagerSingleton(
            firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
        )
    small_map_info = cms.find_maps(small_map_name, search_restriction=0).pop()
    big_map_info = cms.find_maps(big_map_name, search_restriction=0).pop()

    compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3) * 1.0,
        tag_sba_var=1.0,
        ang_vel_var=1.0,
    )

    gt_dataset = cms.find_ground_truth_data_from_map_info(small_map_info)
    # big_gt_dataset = cms.find_ground_truth_data_from_map_info(big_map_info)
    # if small_gt_dataset != big_gt_dataset:
    #     raise ValueError("GT Data should be the same")
    # gt_dataset = small_gt_dataset
    sweep_config = NO_SBA_SWEEP_CONFIG if pso == 1 else SBA_SWEEP_CONFIG

    print("Running Small Map Sweep")
    small_sweep_result = sweep_params(
        mi=small_map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=12,
        cms=cms,
        simple_metrics=True
    )
    small_sweep_gt = small_sweep_result.min_gt
    small_sweep_opt_params = small_sweep_result.args_producing_min

    print("Running Big Map Sweep")
    big_sweep_result = sweep_params(
        mi=big_map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=12,
        cms=cms,
        simple_metrics=True
    )
    big_sweep_gt = big_sweep_result.min_gt
    big_sweep_opt_params = big_sweep_result.args_producing_min

    print("Extapolating to small map")
    big_to_small_extrapolation_result = sweep_params(
        mi=small_map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=big_sweep_opt_params,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=8,
        cms=cms,
        simple_metrics=True
    )
    small_extrapolated_gt = big_to_small_extrapolation_result.min_gt

    print("Extrapolating to big map")
    small_to_big_extrapolation_result = sweep_params(
        mi=big_map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=small_sweep_opt_params,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=8,
        cms=cms,
        simple_metrics=True
    )
    big_extrapolated_gt = small_to_big_extrapolation_result.min_gt
    
    print(f"Small Map Min GT: {small_sweep_gt}")
    print(f"Small Map Extrapolated GT: {small_extrapolated_gt}")
    print(f"Big Map Min GT: {big_sweep_gt}")
    print(f"Big Map Extrapolated GT: {big_extrapolated_gt}")

def run_sweep(map_name, pso):
    # Fetch the service account key JSON file contents
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_variable is None:
        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
    else:
        cms = CacheManagerSingleton(
            firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
        )

    map_info = cms.find_maps(map_name, search_restriction=0).pop()

    compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3) * 1.0,
        tag_sba_var=1.0,
        ang_vel_var=1.0,
    )

    gt_dataset = cms.find_ground_truth_data_from_map_info(map_info)
    sweep_config = NO_SBA_SWEEP_CONFIG if pso == 1 else SBA_SWEEP_CONFIG
    sweep_result = sweep_params(
        mi=map_info,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba=pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=True,
        generate_plot=True,
        show_plot=True,
        num_processes=8,
        cms=cms,
        simple_metrics=False
    )

    return sweep_result

def extrapolate_parameters(map_name_one, map_name_two, pso):
    # Fetch the service account key JSON file contents
    env_variable = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if env_variable is None:
        cms = CacheManagerSingleton(firebase_creds=None, max_listen_wait=0)
    else:
        cms = CacheManagerSingleton(
            firebase_creds=credentials.Certificate(env_variable), max_listen_wait=0
        )

    map_info_one = cms.find_maps(map_name_one).pop()
    map_info_two = cms.find_maps(map_name_two).pop()

    compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3) * 1.0,
        tag_sba_var=1.0,
        ang_vel_var=1.0,
    )

    gt_dataset = cms.find_ground_truth_data_from_map_info(map_info_one)
    sweep_config = NO_SBA_SWEEP_CONFIG if pso == 1 else SBA_SWEEP_CONFIG

    sweep_one_result = sweep_params(
        mi=map_info_one,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=False,
        generate_plot=False,
        show_plot=False,
        num_processes=12,
        cms=cms,
        simple_metrics=True
    )

    sweep_two_result = sweep_params(
        mi=map_info_two,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=False,
        generate_plot=False,
        show_plot=False,
        num_processes=12,
        cms=cms,
        simple_metrics=True
    )
    sweep_two_gt = sweep_two_result.min_gt

    new_sweep_config = sweep_one_result.args_producing_min

    applied_params_result = sweep_params(
        mi=map_info_two,
        ground_truth_data=gt_dataset,
        base_oconfig=OConfig(
            is_sba= pso==0,
            compute_inf_params=compute_inf_params,
        ),
        sweep_config=new_sweep_config,
        ordered_sweep_config_keys=[key for key in sweep_config.keys()],
        verbose=False,
        generate_plot=False,
        show_plot=False,
        num_processes=8,
        cms=cms,
        simple_metrics=True
    )
    applied_gt = applied_params_result.min_gt

    print(f"Extrapolated gt: {applied_gt}")
    print(f"Full Sweep gt: {sweep_two_gt}")

if __name__ == "__main__":
    # extrapolate_parameters("floor_2_all_twice*", "floor_2_all_once*", 1)
    # run_sweep("p1_WH4_2.json", 1)
    run_gt_correlation("4_lou_to_rich_room.json", "p1_WH4_2.json", 1)