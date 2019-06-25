#!/usr/bin/env python
from em import Analysis
from optimization import Optimization
import numpy as np


def cycleOnce(plot=False):
    analysis = Analysis('data_optimized.pkl')
    analysis.getvariance()
    analysis.updateEdges()
    analysis.writePosegraph()

    optimization = Optimization(
        "data_analyzed.pkl", "data_analyzed.pkl")
    optimization.run(plot=plot)
    return optimization


def cycle(tol=10, maxiter=10):
    past = cycleOnce().posegraph.optimization_cost
    present = cycleOnce().posegraph.optimization_cost
    i = 0
    while np.abs(present - past) > tol and i < maxiter:
        past = present
        present = cycleOnce().posegraph.optimization_cost
        print(past, present)
        i += 1

    cycleOnce(plot=True)


def main():
    cycle()


if __name__ == '__main__':
    cycleOnce(plot=True)
