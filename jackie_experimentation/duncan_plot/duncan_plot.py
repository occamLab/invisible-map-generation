import json

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d
import parse_map
import numpy as np


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

class Plot_Chi2_Animation:
    def __init__(self, poseID, adjChi2):
        self.poseID = poseID
        self.adjChi2 = adjChi2
        self.animate()

    def ani_init(self):
        '''resets poseID_text each frame'''
        self.adjChi2_text.set_text('')
        return self.line, self.adjChi2_text

    def update(self, num, poseID, adjChi2):
        ''' Update plotting data '''
        self.line.set_data(self.poseID[:num], self.adjChi2[:num])
        self.adjChi2_text.set_text("chi2:"+str(self.adjChi2[num]))
        return self.line, self.adjChi2_text

    def animate(self):
        # set up graphing
        fig, self.ax = plt.subplots()
        self.ax.set(xlabel='poseID', ylabel='adjChi2',
                    title='adjChi2 Plot')
        self.adjChi2_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.line, = self.ax.loglog(poseID, adjChi2+1)
        # self.line.axes.axis('equal')

        poseID_interval = 100
        # blit: only re-draw the few points that are changing at each frame
        ani = animation.FuncAnimation(fig, lambda x1, x2, x3: self.update(x1, x2, x3), frames=len(x), 
                                fargs=[self.poseID, self.adjChi2_text],
                                interval=poseID_interval, blit=True, init_func = self.ani_init)

        # ani.save('adjChi2plot.mp4', extra_args=['-vcodec', 'libx264'])
        # writervideo = animation.FFMpegWriter(fps=60) 
        # ani.save('adjChi2plot.mp4', writer=writervideo)
        ani.save('adjChi2plot.mp4')

        plt.show()

if __name__ == "__main__":
    map = Map_Data('duncan_test_data.json')
    x, y, z, poseID, adjChi2 = map.parse_odometry()
    # print (x, y, x, adjChi2, poseID)


    # fig, ax = plt.subplots()
    # ax.loglog(poseID, adjChi2+1)
    # ax.set(xlabel='poseID', ylabel='adjChi2',
    #     title='adjChi2 Plot')
    # plt.show()

    # create plots 
    Plot_Chi2_Animation(poseID, adjChi2)

# ffmpeg -i INPUT_1.mp4 -i INPUT_1.mp4 -vsync 2 -filter_complex "[1:v][0:v]scale2ref[wm][base];[base][wm]hstack=2" OUTPUT.mp4
