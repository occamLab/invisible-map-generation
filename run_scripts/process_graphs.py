"""
Script that runs on the deepthought.olin.edu server to listen to Firebase for
new maps and process and upload them.

Author: Allison Li, Duncan Mazza
"""

from map_processing.firebase_manager import FirebaseManager
from map_processing.graph_manager import GraphManager
from map_processing.graph_utils import MapInfo
from firebase_admin import credentials
import os

# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
firebase = FirebaseManager(cred)
graph_manager = GraphManager(weights_specifier=4, firebase_manager=firebase, pso=3)


def on_event(event):
    firebase.get_map_from_unprocessed_map_event(event, for_each_map_info, ignore_dict = True)


def for_each_map_info(map_info: MapInfo) -> None:
    if map_info is None or map_info.map_dct is None or len(map_info.map_dct) == 0:
        return
    json_str = graph_manager.optimize_map_and_get_json(map_info)
    firebase.upload(map_info, json_str)


if __name__ == '__main__':
    firebase.firebase_listen(on_event, -1)
