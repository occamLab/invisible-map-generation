"""
Script that runs on the deepthought.olin.edu server to listen to Firebase for
new maps and process and upload them.

Author: Allison Li, Duncan Mazza
"""

import os
import sys

# Ensure that the map_processing module is imported
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

from map_processing.cache_manager import CacheManagerSingleton, MapInfo
from map_processing.graph_manager import GraphManager
from firebase_admin import credentials

# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
cms = CacheManagerSingleton(cred)
graph_manager = GraphManager(weights_specifier=4, firebase_manager=cms, pso=3)


def on_event(event):
    cms.get_map_from_unprocessed_map_event(event, for_each_map_info, ignore_dict = True)


def for_each_map_info(map_info: MapInfo) -> None:
    if map_info is None or map_info.map_dct is None or len(map_info.map_dct) == 0:
        return
    json_str = graph_manager.optimize_map_and_get_json(map_info)
    cms.upload(map_info, json_str)


if __name__ == '__main__':
    cms.firebase_listen(on_event, -1)
