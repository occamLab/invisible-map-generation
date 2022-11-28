from map_processing.data_models import *
from map_processing.graph_generator import GraphGenerator
from map_processing import GT_TAG_DATASETS, PrescalingOptEnum
from map_processing.cache_manager import MapInfo
from map_processing.graph_opt_hl_interface import (
    holistic_optimize,
    WEIGHTS_DICT,
    WeightSpecifier,
)

small_data_set_name="robolab_straight_on*"
medium_data_set_name=""
large_data_set_name=""

# This data set name corresponds to a hard-coded set of tag poses from the map_processing module.
tag_data_set = "3line"

# These parameters specify an 8 x 4 elliptical path coplanar with the X-Z plane centered at (0, 0, 0);
# because one loop is completed in t=2pi seconds, the t_max parameter of 6pi specifies an elliptical path
# that loops around 3 times.
path_args = {'e_cp': (0.0, 0.0), 'e_xw': 8.0, 'e_zw': 4.0, 'xzp': 0.0}
gen_params = GenerateParams(dataset_name=tag_data_set, parameterized_path_args=path_args, t_max=6 * np.pi,
                            n_poses=100, tag_size=ASSUMED_TAG_SIZE)

gg = GraphGenerator(path_from=GraphGenerator.PARAMETERIZED_PATH_ALIAS_TO_CALLABLE["e"], gen_params=gen_params,
                    tag_poses_for_parameterized=GT_TAG_DATASETS[tag_data_set])

gen_data_set, gt_data_set = gg.export()
map_info = MapInfo(map_json_name=tag_data_set, map_name=tag_data_set, map_dct=gen_data_set.dict())

fixed_vertices = set()

compute_inf_params = OComputeInfParams(
        lin_vel_var=np.ones(3) * np.sqrt(3),
        tag_sba_var=1.0,
        ang_vel_var=1.0
    )

oconfig = OConfig(is_sba=False, weights=Weights(),
                scale_by_edge_amount=False,
                compute_inf_params=compute_inf_params
            )

opt_result = holistic_optimize(
                map_info=map_info,
                pso=PrescalingOptEnum(1),
                oconfig=oconfig,
                verbose=True,
                visualize=True,
                compare=False,
                upload=False,
                gt_data=GTDataSet.gt_data_set_from_dict_of_arrays(gt_data_set.as_dict_of_se3_arrays) if gt_data_set is not None else None,
            )
