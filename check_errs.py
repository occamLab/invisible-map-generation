#!/usr/bin/env python
import pickle
import matplotlib.pyplot as plt
import numpy as np
from graph import isometry_to_pose
from graph_utils import ordered_odometry_edges

with open('converted-data/academic_center.pkl', 'rb') as data:
    graph = pickle.load(data)
# with open('graph.pkl', 'rb') as data:
#     graph = pickle.load(data)


def margin(x):
    return np.log((x / 3)**2)


graph.weights = np.zeros(18)
graph.weights[:3] = margin(.8)
graph.weights[3:6] = margin(20)
graph.weights[6:9] = margin(.4)
graph.weights[9:12] = margin(5)
graph.update_edges()


def plot_errs(errs):
    fig, (translation_ax, rotation_ax) = plt.subplots(2, 3, sharex=True)
    for i, (title, ax) in enumerate(zip(["x", "y", "z"], translation_ax)):
        ax.set_title(r'{} Translation Error $(m)$, $\mu = {:.4g}$, $\sigma^2={:.4g}$'.format(
            title, errs[:, i].mean(), errs[:, i].var()))
        ax.plot(errs[:, i])

    for i, (title, ax) in enumerate(zip(["qx", "qy", "qz"], rotation_ax)):
        ax.set_title(r'{} Translation Error $(m)$, $\mu = {:.4g}$, $\sigma^2={:.4g}$'.format(
            title, errs[:, i + 3].mean(), errs[:, i + 3].var()))
        ax.plot(errs[:, i + 3])
    return fig


def errs_cov(errs, winsz):
    samples = [np.reshape([], (0, winsz)) for _ in range(6)]
    for i in range(winsz, errs.shape[0] - winsz):
        for j in range(6):
            samples[j] = np.vstack([samples[j], errs[i:i+winsz, j]])
    covs = [np.cov(sample.T) for sample in samples]
    return covs


def plot_covs(covs):
    fig, (translation_ax, rotation_ax) = plt.subplots(2, 3)
    for i, (title, ax) in enumerate(zip(["x", "y", "z"], translation_ax)):
        ax.imshow(covs[i])

    for i, (title, ax) in enumerate(zip(["qx", "qy", "qz"], rotation_ax)):
        ax.imshow(covs[i+3])
    return fig


def main():
    graph.generate_unoptimized_graph()
    graph.optimize_graph()
    edges = ordered_odometry_edges(graph)
    errs = np.reshape([], [0, 6])
    edge_lookup = {x.id(): x.error()[:6]
                   for x in graph.optimized_graph.edges()}
    for uid in edges[0]:
        errs = np.vstack([errs, edge_lookup[uid]])

    plot_errs(errs)

    covs = errs_cov(errs, 20)
    plot_covs(covs)

    plt.show()


if __name__ == '__main__':
    main()
