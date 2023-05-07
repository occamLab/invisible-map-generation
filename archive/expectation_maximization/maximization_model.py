"""
Choose values for variance which maximize the likelihood of observed errors.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def likelihood(weights, *args):
    """The function of weights that minimizes the negative likelihood
    of errors by minimizing the negative sum of the log-likelihoods

    The variance of the distribution is assumed to be exp(weights^T
    observations)

    Args:
        weights: a 1-d list of weights.
            Currently, this is an 18-element vector encoding weights
            for x, y, z, qx, qy, qz for odometry, tag, and gravity edges
            respectively.
            For example, the first two elements are odometry x and
            odometry z weights, and the last one is the gravity edge qz
            weight.
        args:
            A tuple of observations (1-d numpy array) and errors
            associated by array index.
            The oservation vector is currently a 1-hot 18-element
            vector encoding if the respective error is an error in x,
            y, z, qx, qy, or qz for odometry, tag, and gravity edges
            respectively.

    Returns:
        The log likelihood of the errors given the input weights.
    """
    observations, errors = args
    return -norm.logpdf(errors, scale=np.sqrt(np.exp(observations.dot(weights)))).sum()


def loglikelihood_gradient(weights, *args):
    """The gradient of the :func: likelihood function.  The input
    parameters are the same.
    """

    observations, errors = args
    return (
        -(np.square(errors) * np.exp(-observations.dot(weights)) - 1).dot(observations)
        / 2
    )


def maxweights(observations, errors, weights):
    """Maximize the likelihood of the errors given a list associated
    observation by tuning transform_vector weights.

    Args:
        observations:
            The oservation array is currently a vertically-stacked
            list 1-hot 18-element vectors encoding if the respective
            error is an error in x, y, z, qx, qy, or qz for odometry,
            tag, and gravity edges respectively. For example, if the
            fifth row of the observation array is [0, 1, 0, ..., 0],
            then the fifth entry in the errors array encodes an
            odometry error in y.
        errors:
            A list of errors.
        weights: a 1-d list of weights.
            Currently, this is an 18-element vector encoding weights
            for x, y, z, qx, qy, qz for odometry, tag, and gravity edges
            respectively.
            For example, the first two elements are odometry x and
            odometry z weights, and the last one is the gravity edge qz
            weight.
    """
    return minimize(
        likelihood, weights, (observations, errors), jac=loglikelihood_gradient
    )
