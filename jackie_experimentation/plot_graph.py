import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
import json


# load mapping data
with open("data/marion_lower_level_map.json", "r") as read_file:
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
    trans_x, trans_y, trans_z = [], [], []
    for pt in odometry_data:
        trans_x.append(pt["translation"]["x"])
        trans_y.append(pt["translation"]["y"])
        trans_z.append(pt["translation"]["z"])
    return np.array(trans_x), np.array(trans_y), np.array(trans_z)


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
    

def plot_translation_2D(x, z):
    fig, ax = plt.subplots()
    ax.plot(x, z)
    ax.set(xlabel='x', ylabel='z',
        title='Test Translation Mapping Data 2D Plot')
    plt.show()


if __name__ == "__main__":
    x, y, z = parse_odometry()

    plot_translation_3D(x, y, z)
    plot_translation_2D(x, z)
