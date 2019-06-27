#!/usr/bin/env python

import pickle
import matplotlib.pyplot as plt

import maximization_model

with open('graph.pkl', 'rb') as data:
    graph = pickle.load(data)


def main():
    graph.optimizeGraph()
    graph.generateUnoptimizedGraph()
    graph.optimizeGraph()

    graph.plotMap()

    plt.show()


if __name__ == '__main__':
    main()
