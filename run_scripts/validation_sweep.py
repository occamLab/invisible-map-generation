"""
This script exists to validate whether there exists a monotonic and positive relationship between metrics based on a
subset of analogous parameters between data set generation and optimization. Specifically, this script looks for a
monotonic and positive relationship between the ratio of linear to angular velocity variance for a generated data set
and the ratio of the optimal linear and angular velocity variance values for optimizing the data set.
"""

import os
import sys

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

import numpy as np
import datetime
from typing import List, Tuple, Callable, Iterable, Any, Dict
from enum import Enum

from map_processing import TIME_FORMAT
from map_processing.data_models import GenerateParams, UGDataSet, OConfig
from map_processing.cache_manager import CacheManagerSingleton
from map_processing.graph_generator import GraphGenerator

HOLD_RVERT_AT = 0.0001
RATIO_XZ_TO_Y_LIN_VEL_VAR = 10

PATH_FROM = "duncan-occam-room-10-1-21-2-38 267139330396791.json"
BASE_GENERATE_PRAMS = GenerateParams(dataset_name=f"generated_from_{PATH_FROM.strip('.json')}")


class AltGenerateParamsEnum(str, Enum):
    OBS_NOISE_VAR = "obs_noise_var"
    """
    Sets the observation noise variance of the generated data set
    """

    LIN_TO_ANG_VEL_VAR = "lin_to_ang_vel_var"
    """
    Defines the ratio between the magnitude of the linear velocity variance vector and the angular velocity variance.
    """


GENERATE_CONFIG: Dict[AltGenerateParamsEnum, Tuple[Callable, Iterable[Any]]] = {
    AltGenerateParamsEnum.OBS_NOISE_VAR: (np.linspace, [2, 2, 1]),
    AltGenerateParamsEnum.LIN_TO_ANG_VEL_VAR: (np.linspace, [0.1, 10, 10])
}


def alt_generate_params_generator(alt_param_multiplicands: Dict[AltGenerateParamsEnum, np.ndarray],
                                  base_generate_params: GenerateParams, hold_rvert_at: float,
                                  ratio_xz_to_y_lin_vel_var: float) \
        -> Tuple[List[Tuple[Any, ...]], List[GenerateParams]]:
    """Acts as a wrapper around GenerateParams.multi_generate_params_generator that utilizes a parameter sweeping space
    defined by the parameters in the AltGenerateParamsEnum enumeration. Generates GenerateParams objects
    according to the cartesian product of the contents of alt_param_multiplicands.

    Args:
        alt_param_multiplicands: Dictionary mapping parameters to arrays of values whose cartesian product is
         taken.
        base_generate_params: Supplies every parameter not prescribed by param_multiplicands.
        hold_rvert_at: Because AltGenerateParamsEnum.LIN_TO_ANG_VEL_VAR is a ratio, the rotational part of the odometry
         noise is held constant with this value.
        ratio_xz_to_y_lin_vel_var: Before the X, Y, and Z elements of the unit-magnitude linear velocity variance vector
         are scaled, this sets the X:Y and Z:Y ratios of the vector's elements.

    Returns:
        A list of the outputs from the cartesian product and the corresponding GenerateParams objects.
    """
    # Expand the alt_param_multiplicands argument into a form that can be used in the
    # GenerateParams.generate_params_generator method.
    param_multiplicands: Dict[GenerateParams.GenerateParamsEnum, np.ndarray] = {}
    for key, values in alt_param_multiplicands.items():
        if key == AltGenerateParamsEnum.OBS_NOISE_VAR:
            param_multiplicands[GenerateParams.GenerateParamsEnum.OBS_NOISE_VAR] = values
        elif key == AltGenerateParamsEnum.LIN_TO_ANG_VEL_VAR:
            # Ignore the values provided in alt_param_multiplicands because we are not interested in the cartesian
            # product between each of the X, Y, Z, and rvert elements of the linear and angular velocity variance.
            # Instead, the values provided in alt_param_multiplicands are applied to the result of the cartesian
            # product.
            lin_vel_var_unit = np.ones(3) * np.array([ratio_xz_to_y_lin_vel_var, 1, ratio_xz_to_y_lin_vel_var])
            lin_vel_var_unit /= np.linalg.norm(lin_vel_var_unit)
            param_multiplicands[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X] = np.array(
                [hold_rvert_at * lin_vel_var_unit[0], ])
            param_multiplicands[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_Y] = np.array(
                [hold_rvert_at * lin_vel_var_unit[1], ])
            param_multiplicands[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_Z] = np.array(
                [hold_rvert_at * lin_vel_var_unit[2], ])
            param_multiplicands[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_RVERT] = np.array(
                [hold_rvert_at, ])
        else:
            raise NotImplementedError("Encountered unhandled parameter: " + str(key))

    param_order = sorted(list(param_multiplicands.keys()))
    param_to_param_order_idx: Dict[GenerateParams.GenerateParamsEnum, int] = {}
    for i, param in enumerate(param_order):
        param_to_param_order_idx[param] = i
    products_intermediate, generate_params_intermediate = GenerateParams.generate_params_generator(
        param_multiplicands=param_multiplicands, param_order=param_order, base_generate_params=base_generate_params)

    # Apply the linear and angular velocity variance values provided in alt_param_multiplicands.
    products_orig_space: List[Tuple[Any, ...]] = []
    generate_params_objects: List[GenerateParams] = []
    for product_pre, generate_param_pre in zip(products_intermediate, generate_params_intermediate):
        for value in alt_param_multiplicands[AltGenerateParamsEnum.LIN_TO_ANG_VEL_VAR]:
            new_generate_param: GenerateParams = GenerateParams.copy(generate_param_pre)
            new_generate_param.odometry_noise_var = {
                GenerateParams.OdomNoiseDims.X: generate_param_pre.odometry_noise_var[
                                                    GenerateParams.OdomNoiseDims.X] * value,
                GenerateParams.OdomNoiseDims.Y: generate_param_pre.odometry_noise_var[
                                                    GenerateParams.OdomNoiseDims.Y] * value,
                GenerateParams.OdomNoiseDims.Z: generate_param_pre.odometry_noise_var[
                                                    GenerateParams.OdomNoiseDims.Z] * value,
                GenerateParams.OdomNoiseDims.RVERT: generate_param_pre.odometry_noise_var[
                    GenerateParams.OdomNoiseDims.RVERT],
            }
            generate_params_objects.append(new_generate_param)

            new_product = list(product_pre)
            new_product[param_to_param_order_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X]] = \
                new_generate_param.odometry_noise_var[GenerateParams.OdomNoiseDims.X]
            new_product[param_to_param_order_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X]] = \
                new_generate_param.odometry_noise_var[GenerateParams.OdomNoiseDims.X]
            new_product[param_to_param_order_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X]] = \
                new_generate_param.odometry_noise_var[GenerateParams.OdomNoiseDims.X]
            new_product[param_to_param_order_idx[GenerateParams.GenerateParamsEnum.ODOMETRY_NOISE_VAR_X]] = \
                new_generate_param.odometry_noise_var[GenerateParams.OdomNoiseDims.X]
            products_orig_space.append(tuple(new_product))
    return products_orig_space, generate_params_objects


if __name__ == "__main__":
    # Evaluate the functions and their arguments stored as the values in GENERATE_CONFIG and use that to generate
    # the unique set of GenerateParams objects.
    sweep_arrs_alt_space: Dict[AltGenerateParamsEnum, np.ndarray] = {}
    for generate_key, generate_value in GENERATE_CONFIG.items():
        sweep_arrs_alt_space[generate_key] = generate_value[0](*generate_value[1])
    products, generate_params = alt_generate_params_generator(
        alt_param_multiplicands=sweep_arrs_alt_space, base_generate_params=BASE_GENERATE_PRAMS,
        hold_rvert_at=HOLD_RVERT_AT, ratio_xz_to_y_lin_vel_var=RATIO_XZ_TO_Y_LIN_VEL_VAR)
    num_to_generate = len(generate_params)
    if len(set(generate_params)) == num_to_generate:
        raise Exception(f"Non-unique set of {GenerateParams.__name__} objects created")

    # Acquire the data set from which the generated maps are parsed
    matching_maps = CacheManagerSingleton.find_maps(PATH_FROM, search_only_unprocessed=False)
    if len(matching_maps) == 0:
        print(f"No matches for {PATH_FROM} in recursive search of {CacheManagerSingleton.CACHE_PATH}")
        exit(0)
    elif len(matching_maps) > 1:
        print(f"More than one match for {PATH_FROM} found in recursive search of {CacheManagerSingleton.CACHE_PATH}. "
              f"Will not batch-generate unless only one path is found.")
        exit(0)
    map_info = matching_maps.pop()
    data_set_parsed = UGDataSet(**map_info.map_dct)

    # Batch-generate data sets
    max_num_prefix_digits = int(np.log10(num_to_generate))
    now_str = datetime.datetime.now().strftime(TIME_FORMAT)
    for gen_param_idx, generate_param in enumerate(generate_params):
        num_zeros_to_prefix = max_num_prefix_digits - int(np.log10(gen_param_idx + 1))
        generate_param.map_id = f"batch_generated_{now_str}_{'0' * num_zeros_to_prefix}{gen_param_idx + 1}"
        gg = GraphGenerator(path_from=data_set_parsed, gen_params=generate_param)
        gg.export_to_map_processing_cache()
        print(f"Generated {'0' * num_zeros_to_prefix}{gen_param_idx + 1}/{num_to_generate}: "
              f"{generate_param.dataset_name}")
