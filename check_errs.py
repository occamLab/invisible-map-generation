#!/usr/bin/env python
"""
Plot the errors between the ARKit measurements and the g2o optimized
measurements.
"""
import pickle
import sys
import matplotlib.pyplot as plt
import numpy as np

if len(sys.argv) < 2:
    FILENAME = 'converted-data/academic_center.pkl'
else:
    FILENAME = sys.argv[1]

with open(FILENAME, 'rb') as data:
    GRAPH = pickle.load(data)


def margin(confidence_interval):
    """Convert a 99.7% confidence interval to a covariance.

    For example, if you are sure that all measurements are within 3e
    units of the mean, the output of the function converts that to a
    variance of 2.

    Args:
        confidence_interval: A distance from the mean you are 99.7%
            sure that all measurements are within.

    Returns:
        A variance corresponding to the input confidence interval.
    """
    return np.log((confidence_interval / 3)**2)


# Set the variance of all the errors and set the edge weights
# accordingly.
GRAPH.weights = np.zeros(18)
GRAPH.weights[:3] = margin(.8)
GRAPH.weights[3:6] = margin(20)
GRAPH.weights[6:9] = margin(.4)
GRAPH.weights[9:12] = margin(5)
GRAPH.update_edges()


def plot_errs(errs):
    """Plot errors.

    Args:
        errs: The errors in a (n, 6) shaped array.
            Each row is formatted as [x, y, z, qx, qy, qz].
    """
    fig, (translation_ax, rotation_ax) = plt.subplots(2, 3, sharex=True)
    for i, (title, axes) in enumerate(zip(["x", "y", "z"], translation_ax)):
        axes.set_title(r'{} Translation Error $(m)$, $\mu = {:.4g}$,'
                       r'$\sigma^2={:.4g}$'
                       .format(title, errs[:, i].mean(), errs[:, i].var()))
        axes.plot(errs[:, i])

    for i, (title, axes) in enumerate(zip(["qx", "qy", "qz"], rotation_ax)):
        axes.set_title(r'{} Translation Error $(m)$, $\mu = {:.4g}$,'
                       r'$\sigma^2={:.4g}$'
                       .format(title, errs[:, i + 3].mean(),
                               errs[:, i + 3].var()))
        axes.plot(errs[:, i + 3])
    return fig


def errs_cov(errs, winsz):
    """Return the covariance matrix of 1-d array of errors encoding a
    single dimension, such as x.

    This is used to see if errors are independent of each other. If
    they are not, off-diagonal entries will be large.

    Args:
        errs: The 1-d list of errors of a single dimension, such as x.
        winsz: How many columns are in the covariance matrix.
           For example, a winsz of 10 will create a covariance matrix
           correlating the first to the second, first to the third,
           and so on for every consequtive group of 10 measurements.

    Returns:
        A covariance matrix showing correlation between adjacent errors.

    """
    samples = [np.reshape([], (0, winsz)) for _ in range(6)]
    for i in range(winsz, errs.shape[0] - winsz):
        for j in range(6):
            samples[j] = np.vstack([samples[j], errs[i:i+winsz, j]])
    covs = [np.cov(sample.T) for sample in samples]
    return covs


def plot_covs(covs):
    """Plot found covariance matrices.

    Args:
        covs: A list of covariance matrices to visualize.
            There should be 6, for x, y, z, qx, qy, and qz respectively.
    """
    fig, (translation_ax, rotation_ax) = plt.subplots(2, 3)
    for i, axes in enumerate(translation_ax):
        axes.imshow(covs[i])

    for i, axes in enumerate(rotation_ax):
        axes.imshow(covs[i+3])
    return fig


def main():
    """Visualize the errors between the raw ARKit positions and the
    g2o optimized positions.
    """
    GRAPH.generate_unoptimized_graph()
    GRAPH.optimize_graph()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    GRAPH.expectation_maximization_once()
    print(GRAPH.weights)
    edges = GRAPH.ordered_odometry_edges()
    errs = np.reshape([], [0, 6])
    edge_lookup = {x.id(): x.error()[:6]
                   for x in GRAPH.optimized_graph.edges()}
    for uid in edges[0]:
        errs = np.vstack([errs, edge_lookup[uid]])


    plot_errs(errs)

    covs = errs_cov(errs, 32)
    plot_covs(covs)

    plt.show()


if __name__ == '__main__':
    main()
