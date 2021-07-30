from map_processing.firebase_manager import FirebaseManager
from map_processing.GraphManager import GraphManager

from firebase_admin import credentials
import os


# Fetch the service account key JSON file contents
cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
firebase = FirebaseManager(cred)
graph_manager = GraphManager(weights_specifier=4, firebase_manager=firebase, pso=1)


def on_event(event):
    map_info = firebase.get_map_from_event(event)
    json_str = graph_manager.get_json_from_map_info(map_info)
    firebase.upload(map_info, json_str)


if __name__ == '__main__':
    firebase.set_listener(on_event)
