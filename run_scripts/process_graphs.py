"""
Script that runs on the deepthought.olin.edu server to listen to Firebase for
new maps and process and upload them.

Author: Allison Li, Duncan Mazza
"""

import os
import sys
from datetime import datetime

repository_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)
sys.path.append(repository_root)

from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.graph_opt_hl_interface import PrescalingOptEnum, WeightSpecifier, WEIGHTS_DICT, optimize_graph
from map_processing.graph import Graph
from map_processing.graph_opt_utils import make_processed_map_json
from firebase_admin import credentials

from map_processing.data_models import OConfig

# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
cms = CacheManagerSingleton(cred)


def on_event(event):
    print("got a new map to process")
    cms.get_map_from_unprocessed_map_event(event, for_each_map_info, ignore_dict=False, override_all = False)

def for_each_map_info(map_info: MapInfo) -> None:
    if map_info is None or map_info.map_dct is None or len(map_info.map_dct) == 0:
        return
    map_info.map_json_blob_name = f'{map_info.map_json_blob_name[:-5]} {datetime.now().strftime("%Y%m%d%H%M%S")}.json'
    graph = Graph.as_graph(map_info.map_dct, prescaling_opt=PrescalingOptEnum.ONES)
    optimization_config = OConfig(
        is_sba=False, weights=WEIGHTS_DICT[WeightSpecifier.BEST_SWEEP])
    opt_result = optimize_graph(graph=graph, oconfig=optimization_config)
    json_str = make_processed_map_json(opt_result.map_opt, calculate_intersections=True)
    cms.upload(map_info, json_str)


if __name__ == '__main__':
    cms.firebase_listen(on_event, -1)
