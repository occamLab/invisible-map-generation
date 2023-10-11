"""
Script that runs on the deepthought.olin.edu server to listen to Firebase for
new maps and process and upload them.

Author: Allison Li, Duncan Mazza
"""
import os
import sys
from datetime import datetime

from firebase_admin import credentials

from map_processing.data_models import OConfig
from map_processing.graph_opt_utils import (
    make_processed_map_json,
    make_processed_map_protobuf,
)
import map_processing.throw_out_bad_tags as tag_filter
from map_processing.graph import Graph
from map_processing.graph_opt_hl_interface import (
    PrescalingOptEnum,
    WeightSpecifier,
    WEIGHTS_DICT,
    optimize_graph,
)
from map_processing.cache_manager import CacheManagerSingleton, MapInfo

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"))
cms = CacheManagerSingleton(cred)


def on_event(event):
    """
    Callback for new events on firebase. This sends the for_each_map_info function (which runs the
    entire optimization workflow and uploads processed json to firebase) to the CMS for it to
    extract the unprocessed map json and run for_each_map_info on it.
    """
    cms.get_map_from_unprocessed_map_event(
        event, for_each_map_info, ignore_dict=True, override_all=False
    )


def for_each_map_info(map_info: MapInfo) -> None:
    """
    Full workflow of invisible map optimization. This function is called on every event where a new
    map is uploaded. The workflow is as follows:
        1. Receives map info object from CMS
        2. Converts map_info to graph object.
        3. Runs optimization with pre-defined weights and determines if SBA will be implemented
        4. Makes processed map json from OResult
        5. Uploaded processed map json to firebase under appropriate name
    """
    if map_info is None or map_info.map_dct is None or len(map_info.map_dct) == 0:
        return
    map_info.map_json_blob_name = (
        f"{map_info.map_json_blob_name[:-5]}"
        + f'_{datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    )
    combined_map = cms.combine_shared_maps(map_seed=map_info)
    combined_map.map_dct = tag_filter.throw_out_bad_tags(
        combined_map.map_dct, combined_map.map_name, verbose=False
    )
    graph = Graph.as_graph(combined_map.map_dct, prescaling_opt=PrescalingOptEnum.ONES)
    optimization_config = OConfig(
        is_sba=False, weights=WEIGHTS_DICT[WeightSpecifier.BEST_SWEEP]
    )
    opt_result = optimize_graph(graph=graph, oconfig=optimization_config)
    # json_str = make_processed_map_json(
    #     graph=graph, opt_result=opt_result.map_opt, calculate_intersections=True
    # )
    proto_str = make_processed_map_protobuf(
        graph=graph, opt_result=opt_result.map_opt, calculate_intersections=True
    )
    # cms.upload(map_info, json_str)
    cms.upload(map_info, proto_str)


if __name__ == "__main__":
    # Establishes connection to firebase and runs callback (on_event) on every firebase
    # action (ex: new map is uplaoded by a user)
    cms.firebase_listen(on_event, -1)
