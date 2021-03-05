import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mpl_toolkits.mplot3d
import numpy as np
import json
import os
import plot_graph

x, y, z, poseID= plot_graph.parse_odometry()
print(x)
# set up graphing
fig, ax = plt.subplots()
ax.set(xlabel='x', ylabel='z',
    title='Test Translation Mapping Data 2D Plot')
poseID_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
line, = ax.plot(x, z)
print(type(poseID_text), type(line))


def init():
    '''init animation'''
    poseID_text.set_text('')
    return line, poseID_text

def update(num, x, y, poseID):
    line.set_data(x[:num], y[:num])
    print(x[num], y[num])
    poseID_text.set_text(poseID[num])
    line.axes.axis([-10, 10, -10, 10])
    print(type(poseID_text), type(line))

    # return line,
    return line, poseID_text


ani = animation.FuncAnimation(fig, update, frames=len(x), fargs=[x, z, poseID],
                            interval=100, blit=True, init_func = init)
# ani = animation.FuncAnimation(fig, update, frames=len(x), fargs=[x, z, poseID],
#                             interval=100, blit=True)

# ani.save('test.gif')
plt.show()
