import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d
import numpy as np
import json
import os


# load mapping data
with open('marion_lower_level_map_with_poseID.json', "r") as read_file:
    mapping_data = json.load(read_file)
    odometry_data = mapping_data["odometry_vertices"]
    tag_data = mapping_data["tag_vertices"]

# print(mapping_data)
# print(odometry_data)

def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def parse_odometry():
    '''
    Parses odometry data and returns numpy translation arrays
    '''
    trans_x, trans_y, trans_z, poseID = [], [], [], []
    for pt in odometry_data:
        trans_x.append(pt["translation"]["x"])
        trans_y.append(pt["translation"]["y"])
        trans_z.append(pt["translation"]["z"])
        poseID.append(pt["poseId"])
    return np.array(trans_x), np.array(trans_y), np.array(trans_z), np.array(poseID)


def parse_tag():
    '''
    Parses tag data and returns numpy translation arrays
    '''
    trans_x, trans_y, trans_z = [], [], []
    for pt in tag_data:
        trans_x.append(pt["translation"]["x"])
        trans_y.append(pt["translation"]["y"])
        trans_z.append(pt["translation"]["z"])
    return np.array(trans_x), np.array(trans_y), np.array(trans_z)


def plot_translation_3D(x, y, z):
    '''
    Create 3D plot of the 
    '''
    # plot 3d plot
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(x, z, y)

    ax.set_title("Test Translation Mapping Data 3D Plot")
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.set_zlabel("y")

    ax.set_box_aspect([1,1,1]) # IMPORTANT - this is the new, key line
    # ax.set_proj_type('ortho') # OPTIONAL - default is perspective (shown in image above)
    set_axes_equal(ax) # IMPORTANT - this is also required

    plt.show()
    
def update(num, x, y, line):
    line.set_data(x[:num], y[:num])
    line.axes.axis([-10, 10, -10, 10])
    return line,

def plot_translation_2D(x, z):
    fig, ax = plt.subplots()
    line, = ax.plot(x, z)
    ax.set(xlabel='x', ylabel='z',
        title='Test Translation Mapping Data 2D Plot')

    ani = animation.FuncAnimation(fig, update, len(x), fargs=[x, z, line],
                              interval=100, blit=True, save_count=1490)

    ani.save('test.mp4',fps=10, extra_args=['-vcodec', 'libx264'])
    plt.show()

if __name__ == "__main__":
    x, y, z, poseID = parse_odometry()

    # plot_translation_3D(x, y, z)
    # print(len(x))

# ffmpeg -i marion_recording.mp4 -i test.mp4 -filter_complex \
# "[0:v][1:v]hstack=inputs=2[v]; \
#  [0:a][1:a]amerge[a]" \
# -map "[v]" -map "[a]" -ac 2 output.mp4

# ffmpeg -i marion_recording.mp4 -i test.mp4 -r 30 -filter_complex "[0:v]scale=640:480, setpts=PTS-STARTPTS, pad=1280:720:0:120[left]; [1:v]scale=640:480, setpts=PTS-STARTPTS, pad=640:720:0:120[right]; [left][right]overlay=w; amerge,pan=stereo:c0<c0+c2:c1<c1+c3" -vcodec libx264 -acodec aac -strict experimental output.mp4
# ffmpeg -i marion_recording.mp4 -i test.mp4 -filter_complex hstack output.mp4
