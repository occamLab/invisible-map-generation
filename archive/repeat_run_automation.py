import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from map_processing.cache_manager import CacheManagerSingleton
from map_processing.data_models import UGDataSet, GenerateParams
from map_processing import ASSUMED_TAG_SIZE, TIME_FORMAT
from map_processing.graph_generator import GraphGenerator
from run import run_sweep
import datetime
import json
import time


def run():
    json_file_name = "repeat_sweep_test_results_normalized.json"
    base_map_names = ["4_lou_to_rich_room*", "p1_WH4_2*", "p1_WH4*", "WH4*"]
    noise_configs = [
        {"obs": 0, "odom_tuple": (0.0, 0.0, 0.0, 0.0)},
        {"obs": 5, "odom_tuple": (0.0, 0.0, 0.0, 0.0)},
        {"obs": 10, "odom_tuple": (0.0, 0.0, 0.0, 0.0)},
        {"obs": 0, "odom_tuple": (0.01, 0.01, 0.01, 0.001)},
        {"obs": 0, "odom_tuple": (0.1, 0.1, 0.1, 0.025)},
        {"obs": 5, "odom_tuple": (0.01, 0.01, 0.01, 0.001)},
        {"obs": 10, "odom_tuple": (0.1, 0.1, 0.1, 0.0025)},
    ]

    sweep_result_json = {}
    counter = 0

    for base_map in base_map_names:
        matching_maps = CacheManagerSingleton.find_maps(base_map, search_restriction=0)
        map_info = matching_maps.pop()
        data_set_parsed = UGDataSet(**map_info.map_dct)

        sweep_result_json[base_map] = {}

        for i, noise_config in enumerate(noise_configs):
            counter += 1
            print(
                f"Parameter sweep {counter} started, {len(base_map_names)*len(noise_configs)-counter} remaining."
            )
            obs_noise_var = noise_config["obs"]
            odom_noise_tuple = noise_config["odom_tuple"]
            odom_noise = {
                noise_param_enum: odom_noise_tuple[i]
                for i, noise_param_enum in enumerate(
                    GenerateParams.OdomNoiseDims.ordering()
                )
            }

            gen_params = GenerateParams(
                dataset_name=map_info.map_name,
                tag_size=ASSUMED_TAG_SIZE,
                odometry_noise_var=odom_noise,
                obs_noise_var=obs_noise_var,
            )
            gen_map_name = "generated_" + datetime.datetime.now().strftime(TIME_FORMAT)
            gen_params.map_id = gen_map_name
            gg = GraphGenerator(path_from=data_set_parsed, gen_params=gen_params)
            gg.export_to_map_processing_cache()
            time.sleep(1)
            sweep_result = run_sweep(gen_map_name + "*", 1)
            sweep_result_json[base_map][i] = {
                "dataset_name": gen_map_name,
                "tag_noise": obs_noise_var,
                "odom_noise": odom_noise_tuple,
                "pre_opt_gt": sweep_result.pre_opt_gt,
                "min_gt": sweep_result.min_gt_result,
                "shift_min_gt": sweep_result.min_shift_gt,
            }
            print(f"Min GT: {sweep_result.min_gt_result}")
            print(f"Min Shift: {sweep_result.min_shift_gt}")

            output_file = open(json_file_name, "w")
            json.dump(sweep_result_json, output_file, indent=4)
            output_file.close()


if __name__ == "__main__":
    run()
