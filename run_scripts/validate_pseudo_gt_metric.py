"""
Script for generating the data to (in)validate the pseudo ground truth metric.
"""

import os
import sys
import argparse
import numpy as np
import datetime
from typing import List, Tuple, Callable, Iterable, Any, Dict
import matplotlib.pyplot as plt

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from map_processing import TIME_FORMAT, PrescalingOptEnum, GT_TAG_DATASETS
from map_processing.data_models import (
    GenerateParams,
    UGDataSet,
    OConfig,
    OResult,
    OSGPairResult,
    OResultPseudoGTMetricValidation,
)
from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.graph_generator import GraphGenerator
from map_processing.graph_opt_hl_interface import (
    optimize_graph,
    ground_truth_metric_with_tag_id_intersection,
    tag_pose_array_with_metadata_to_map,
    holistic_optimize,
)
from map_processing.graph import Graph

GENERATE_FROM_PATH = True
"""
If true, then a graph is generated from an existing data set. Otherwise, an "
"elliptical parameterized path is used.
"""
PATH_FROM = "duncan-occam-room-10-1-21-2-48 26773176629225.json"
NUM_REPEAT_GENERATE = 1

if GENERATE_FROM_PATH:
    GENERATE_FROM = GenerateParams(
        obs_noise_var=0.5,
        odometry_noise_var={
            GenerateParams.OdomNoiseDims.X: 1e-4,
            GenerateParams.OdomNoiseDims.Y: 1e-5,
            GenerateParams.OdomNoiseDims.Z: 1e-4,
            GenerateParams.OdomNoiseDims.RVERT: 1e-5,
        },
        dataset_name=f"generated_from_{PATH_FROM.strip('.json')}",
    )
else:
    GENERATE_FROM = GenerateParams(
        obs_noise_var=0.5,
        odometry_noise_var={
            GenerateParams.OdomNoiseDims.X: 1e-4,
            GenerateParams.OdomNoiseDims.Y: 1e-5,
            GenerateParams.OdomNoiseDims.Z: 1e-4,
            GenerateParams.OdomNoiseDims.RVERT: 1e-5,
        },
        dataset_name="3line",
        t_max=6 * np.pi,
        n_poses=300,
        parameterized_path_args={
            "e_cp": (0.0, 0.0),
            "e_xw": 8.0,
            "e_zw": 4.0,
            "xzp": 0.0,
        },
    )

ALT_OPT_CONFIG: Dict[OConfig.AltOConfigEnum, Tuple[Callable, Iterable[Any]]] = {
    # OConfig.AltOConfigEnum.LIN_TO_ANG_VEL_VAR: (np.geomspace, [10, 10, 1]),
    OConfig.AltOConfigEnum.TAG_SBA_VAR: (np.geomspace, [1e-6, 1e2, 30])
}

BASE_OCONFIG = OConfig(is_sba=True)


def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Script for generating the data to (in)validate the pseudo ground "
        "truth metric.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--lfc",
        action="store_true",
        help="Load from cache: If specified, then the cached data set specified by the"
        "hard-coded file path is loaded into memory and the plots are generated. "
        "Otherwise, a new data set is generated.",
        default=False,
    )
    return p


# noinspection DuplicatedCode
def validate_pseudo_gt_metric():
    # Acquire the data set from which the generated maps are parsed
    matching_maps = CacheManagerSingleton.find_maps(
        PATH_FROM, search_only_unprocessed=True
    )
    if len(matching_maps) == 0:
        print(
            f"No matches for {PATH_FROM} in recursive search of"
            f"{CacheManagerSingleton.CACHE_PATH}"
        )
        exit(0)
    elif len(matching_maps) > 1:
        print(
            f"More than one match for {PATH_FROM} found in recursive search of"
            f"{CacheManagerSingleton.CACHE_PATH}. Will not batch-generate unless"
            "only one path is found."
        )
        exit(0)

    # Acquire the data set from which the generated data sets are derived from
    map_info = matching_maps.pop()
    data_set_generate_from = UGDataSet(**map_info.map_dct)

    alt_param_multiplicands_for_opt: Dict[OConfig.AltOConfigEnum, np.ndarray] = {}
    for opt_key, opt_value in ALT_OPT_CONFIG.items():
        if isinstance(opt_value, np.ndarray):
            alt_param_multiplicands_for_opt[opt_key] = opt_value
        else:
            alt_param_multiplicands_for_opt[opt_key] = opt_value[0](*opt_value[1])
    param_multiplicands_for_opt = OConfig.alt_oconfig_generator_param_multiplicands(
        alt_param_multiplicands=alt_param_multiplicands_for_opt
    )
    _, oconfig_list = OConfig.oconfig_generator(
        param_multiplicands=param_multiplicands_for_opt,
        param_order=[key for key in param_multiplicands_for_opt.keys()],
        base_oconfig=BASE_OCONFIG,
    )

    results: List[Tuple[OConfig, OResult, OSGPairResult]] = []
    generate_idx = -1
    mi_and_gt_data_set_list: List[Tuple[MapInfo, Dict]] = []
    print(
        f"Generating {NUM_REPEAT_GENERATE} data set"
        f"{'s' if NUM_REPEAT_GENERATE > 1 else ''}..."
    )
    for i in range(NUM_REPEAT_GENERATE):
        if GENERATE_FROM_PATH:
            gg = GraphGenerator(
                path_from=data_set_generate_from, gen_params=GENERATE_FROM
            )
        else:
            gg = GraphGenerator(
                path_from=GraphGenerator.xz_path_ellipsis_four_by_two,
                gen_params=GENERATE_FROM,
                tag_poses_for_parameterized=GT_TAG_DATASETS["3line"],
            )
        mi = gg.export_to_map_processing_cache(file_name_suffix=f"_{i}")
        gt_data_set = CacheManagerSingleton.find_ground_truth_data_from_map_info(mi)
        mi_and_gt_data_set_list.append((mi, gt_data_set))

    total_num_optimizations = NUM_REPEAT_GENERATE * len(oconfig_list)
    for oconfig in oconfig_list:
        for mi, gt_data_set in mi_and_gt_data_set_list:
            generate_idx += 1
            oresult = optimize_graph(
                graph=Graph.as_graph(data_set=UGDataSet.parse_obj(mi.map_dct)),
                oconfig=oconfig,
            )
            oresult.gt_metric_opt = ground_truth_metric_with_tag_id_intersection(
                optimized_tags=tag_pose_array_with_metadata_to_map(
                    oresult.map_opt.tags
                ),
                ground_truth_tags=gt_data_set,
            )

            osg_pair_result: OSGPairResult = holistic_optimize(
                map_info=mi,
                pso=PrescalingOptEnum.USE_SBA,
                oconfig=oconfig,
                compare=True,
            )
            results.append((oconfig, oresult, osg_pair_result))
            print(
                f"Completed optimization {generate_idx + 1}/{total_num_optimizations}"
            )

    all_results_obj = OResultPseudoGTMetricValidation(
        results_list=results, generate_params=GENERATE_FROM
    )
    results_file_name = (
        f"pgt_validation_{datetime.datetime.now().strftime(TIME_FORMAT)}"
    )
    CacheManagerSingleton.cache_pgt_validation_results(
        results=all_results_obj, file_name=results_file_name
    )
    fig_1 = all_results_obj.plot_scatter(
        fitness_metric=OResultPseudoGTMetricValidation.ScatterYAxisOptions.CHI2,
        colorbar_variable=OResultPseudoGTMetricValidation.ScatterColorbarOptions.TAG_SBA_VAR,
    )
    fig_1.savefig(
        os.path.join(
            CacheManagerSingleton.PGT_VALIDATION_RESULTS_PATH,
            results_file_name + "_chi2.png",
        ),
        dpi=500,
    )
    fig_2 = all_results_obj.plot_scatter(
        fitness_metric=OResultPseudoGTMetricValidation.ScatterYAxisOptions.ALPHA,
        colorbar_variable=OResultPseudoGTMetricValidation.ScatterColorbarOptions.TAG_SBA_VAR,
    )
    fig_2.savefig(
        os.path.join(
            CacheManagerSingleton.PGT_VALIDATION_RESULTS_PATH,
            results_file_name + "_alpha.png",
        ),
        dpi=500,
    )


if __name__ == "__main__":
    parser = make_parser()
    args = parser.parse_args()

    if args.lfc:
        with open(
            "../.cache/pgt_validation_results/pgt_validation_22-06-27-19-07-50.json",
            "r",
        ) as f:
            s = f.read()
        pgt = OResultPseudoGTMetricValidation.parse_raw(s)
        pgt.plot_scatter(
            fitness_metric=OResultPseudoGTMetricValidation.ScatterYAxisOptions.CHI2,
            colorbar_variable=OResultPseudoGTMetricValidation.ScatterColorbarOptions.LIN_VEL_VAR,
        )
        plt.show()
        pgt.plot_scatter(
            fitness_metric=OResultPseudoGTMetricValidation.ScatterYAxisOptions.ALPHA,
            colorbar_variable=OResultPseudoGTMetricValidation.ScatterColorbarOptions.TAG_SBA_VAR,
        )
        plt.show()
    else:
        validate_pseudo_gt_metric()
