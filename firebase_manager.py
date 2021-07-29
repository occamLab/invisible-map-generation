"""
Script that runs on the deepthought.olin.edu server to listen to Firebase for
new maps and process and upload them.

Author: Allison Li
"""
import os
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from GraphManager import GraphManager

class MapsListener:
    """

    """
    def __init__(self, graph_manager: GraphManager, download_only: bool = False, max_wait: int = -1):
        self._graph_manager = graph_manager
        self.download_only = download_only
        self.max_wait = max_wait

    def listener(self, event):
        print(event.event_type)

    def firebase_listen(self):
        if self.max_wait == -1:
            db.reference(f'/{GraphManager.unprocessed_listen_to}').listen(maps_listener.listener)





if __name__ == "__main__":
    # Fetch the service account key JSON file contents
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    graph_manager_local = GraphManager(weights_specifier=4, firebase_creds=cred, pso=1)
    firebase_listen(graph_manager=graph_manager_local)

