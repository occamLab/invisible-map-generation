import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import parse_map
import numpy as np


class Plot_3D:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.plot_translation_3D()

    def set_axes_equal(self):
        """Set 3D plot axes to equal scale.

        Make axes of 3D plot have equal scale so that spheres appear as
        spheres and cubes as cubes.  Required since `ax.axis('equal')`
        and `ax.set_aspect('equal')` don't work on 3D.
        """
        limits = np.array([
            self.ax.get_xlim3d(),
            self.ax.get_ylim3d(),
            self.ax.get_zlim3d(),
        ])
        origin = np.mean(limits, axis=1)
        radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
        self.set_axes_radius(origin, radius)

    def set_axes_radius(self, origin, radius):
        x, y, z = origin
        self.ax.set_xlim3d([x - radius, x + radius])
        self.ax.set_ylim3d([y - radius, y + radius])
        self.ax.set_zlim3d([z - radius, z + radius])

    def plot_translation_3D(self):
        '''
        Create 3D plot
        '''
        fig = plt.figure()
        self.ax = fig.gca(projection='3d')
        self.ax.set(xlabel='x', ylabel='z', zlabel='y',
            title='3D Plot of Map')

        self.ax.scatter(self.x, self.z, self.y, s=1**2)

        self.set_axes_equal()
        plt.show()


class Plot_2D:
    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.plot_translation_2D()

    def plot_translation_2D(self):
        fig, self.ax = plt.subplots()
        self.ax.plot(self.x, self.z)
        self.ax.set(xlabel='x', ylabel='z',
            title='2D Plot of Map')
        plt.show()



class Plot_2D_Animation:
    def __init__(self, x, z, poseID):
        self.x = x
        self.y = z
        self.poseID = poseID
        self.animate()

    def ani_init(self):
        '''resets poseID_text each frame'''
        self.poseID_text.set_text('')
        return self.line, self.poseID_text

    def update(self, num, x, y, poseID):
        ''' Update plotting data '''
        self.line.set_data(self.x[:num], self.y[:num])
        self.poseID_text.set_text(self.poseID[num])
        return self.line, self.poseID_text

    def animate(self):
        # set up graphing
        fig, self.ax = plt.subplots()
        self.ax.set(xlabel='x', ylabel='z',
            title='Test Translation Mapping Data 2D Plot')
        self.poseID_text = self.ax.text(0.02, 0.95, '', transform=self.ax.transAxes)
        self.line, = self.ax.plot(x, z)
        self.line.axes.axis('equal')

        poseID_interval = 100
        # blit: only re-draw the few points that are changing at each frame
        ani = animation.FuncAnimation(fig, lambda x1, x2, x3, x4: self.update(x1, x2, x3,x4), frames=len(x), 
                                fargs=[self.x, self.y, self.poseID],
                                interval=poseID_interval, blit=True, init_func = self.ani_init)

        ani.save('marion_map_animation.mp4', extra_args=['-vcodec', 'libx264'])
        # ani.save('test.gif')
        plt.show()

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
    map = parse_map.Map_Data('duncan_test_data.json')
    x, y, z, poseID, adjChi2 = map.parse_odometry()
    # print (x, y, x, adjChi2, poseID)


    # fig, ax = plt.subplots()
    # ax.loglog(poseID, adjChi2+1)
    # ax.set(xlabel='poseID', ylabel='adjChi2',
    #     title='adjChi2 Plot')
    # plt.show()

    # create plots 
    # Plot_3D(x, y, z)
    # Plot_2D(x, z)
    # Plot_2D_Animation(x, z, poseID)
    Plot_Chi2_Animation(poseID, adjChi2)



# ffmpeg -i marion_map_animation.mp4 -i marion_recording.mp4 -vsync 2 -filter_complex "[1:v][0:v]scale2ref[wm][base];[base][wm]hstack=2" final.mp4
