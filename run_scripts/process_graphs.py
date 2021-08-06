from map_processing.firebase_manager import FirebaseManager
from map_processing.graph_manager import GraphManager

from firebase_admin import credentials
import os


# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
firebase = FirebaseManager(cred)
graph_manager = GraphManager(weights_specifier=4, firebase_manager=firebase, pso=3)


def on_event(event):
    map_info = firebase.get_map_from_event(event)
    if map_info is None or map_info.map_dct is None or len(map_info.map_dct) == 0:
        return
    json_str = graph_manager.get_json_from_map_info(map_info)
    firebase.upload(map_info, json_str)


if __name__ == '__main__':
    firebase.set_listener(on_event)
