#!/usr/bin/env python

import numpy as np
from scipy.stats import norm
from scipy.optimize import minimize


def objfun(w, *args):
    x, e = args
    return -norm.logpdf(e, scale=np.sqrt(np.exp(x.dot(w)))).sum()


def gradfun(w, *args):
    x, e = args
    return np.divide(-(np.square(e) * np.exp(-x.dot(w)) - 1).dot(x), 2)


def maxweights(x, e, w):
    # return minimize(objfun, w, (x, e))
    return minimize(objfun, w, (x, e), jac=gradfun)
