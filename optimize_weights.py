"""
Uses a genetic algorithm to optimize the weights for the graph optimization
"""

from firebase_admin import credentials
import os

from GraphManager import GraphManager

CACHE_DIRECTORY = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               ".cache", "unprocessed_maps", "myTestFolder")
MAP_JSON = "2900094388220836-17-21 OCCAM Room.json"

if __name__ == "__main__":
    cred = credentials.Certificate(os.environ.get('GOOGLE_APPLICATION_CREDENTIALS'))
    graph_manager = GraphManager(0, cred)
    print(graph_manager.optimize_weights(os.path.join(CACHE_DIRECTORY, MAP_JSON)))

