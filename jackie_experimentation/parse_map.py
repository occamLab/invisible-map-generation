import numpy as np
import json

class Map_Data:
    ''' 
    Contains map data
    '''
    def __init__(self, filepath):
        self.filepath = filepath
        self.load_map()
    
    def load_map(self):
        ''' Load map from map in json format
        '''
        with open(self.filepath, "r") as read_file:
            mapping_data = json.load(read_file)
            self.odometry_data = mapping_data["odometry_vertices"]
            self.tag_data = mapping_data["tag_vertices"]
            self.waypoints_data = mapping_data["waypoints_vertices"]

    def parse_odometry(self):
        '''
        Parses odometry data and returns numpy translation arrays and poseId array
        '''
        self.trans_x, self.trans_y, self.trans_z, self.poseId, self.adjChi2 = [], [], [], [], []
        for pt in self.odometry_data:
            self.trans_x.append(pt["translation"]["x"])
            self.trans_y.append(pt["translation"]["y"])
            self.trans_z.append(pt["translation"]["z"])
            self.poseId.append(pt["poseId"])
            if "adjChi2" in pt:
                self.adjChi2.append(pt["adjChi2"])

        return  np.array(self.trans_x), np.array(self.trans_y), \
                    np.array(self.trans_z), np.array(self.poseId), \
                    np.array(self.adjChi2)

if __name__ == "__main__":
    map = Map_Data('duncan_test_data.json')
    x, y, z, poseID, adjChi2 = map.parse_odometry()
    print (adjChi2)
