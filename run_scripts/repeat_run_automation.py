import os
import sys

repository_root = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import UGDataSet, GenerateParams
from map_processing import ASSUMED_TAG_SIZE, TIME_FORMAT
from map_processing.graph_generator import GraphGenerator
from run import run_sweep
import datetime
import json

def run():
    json_file_name = "repeat_alpha_test_results.json"
    base_map_name = "4_lou_to_rich_room*"
    num_repeats = 10

    # Odometry Noise:
    pos_noise = .1  # m
    orientation_noise = 0.001   # rad?
    odom_noise_tuple = (pos_noise, pos_noise, pos_noise, orientation_noise)
    odom_noise = {
        noise_param_enum: odom_noise_tuple[i]
        for i, noise_param_enum in enumerate(
            GenerateParams.OdomNoiseDims.ordering()
        )
    }

    # Tag Observation noise:
    tag_noise = 10  # Pixels

    matching_maps = CacheManagerSingleton.find_maps(
        base_map_name, search_restriction=0
    )

    if len(matching_maps) == 0:
                print(
                    f"No matches for {base_map_name} in recursive search of {CacheManagerSingleton.CACHE_PATH}"
                )
                exit(0)
    map_info = matching_maps.pop()
    data_set_parsed = UGDataSet(**map_info.map_dct)
    gen_params = GenerateParams(
        dataset_name=map_info.map_name,
        tag_size=ASSUMED_TAG_SIZE,
        odometry_noise_var=odom_noise,
        obs_noise_var=tag_noise
    )
    sweep_result_json = {
        base_map_name: {}
    }
    for i in range(num_repeats):
        gen_map_name = "generated_" + datetime.datetime.now().strftime(TIME_FORMAT)
        gen_params.map_id = gen_map_name
        gg = GraphGenerator(path_from=data_set_parsed,
            gen_params=gen_params)
        gg.export_to_map_processing_cache()
        sweep_result = run_sweep(gen_map_name + "*", 1)
        sweep_result_json[base_map_name][i] = {
            "dataset_name": gen_map_name,
            "tag_noise": tag_noise,
            "odom_noise": odom_noise_tuple,
            "pre_opt_gt": sweep_result.pre_opt_gt,
            "min_gt": sweep_result.min_gt_result,
            "alpha_min_gt": sweep_result.min_oresult_alpha.gt_metric_opt
        }

    output_file = open(json_file_name, 'w')
    json.dump(sweep_result_json, output_file, indent=4)
    output_file.close()

if __name__ == "__main__":
    run()