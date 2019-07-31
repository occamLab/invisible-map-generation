#!/usr/bin/env python
import sys
import pickle
import convert_posegraph

if len(sys.argv) < 2:
    print("Usage: convert_pickle.py SRC DEST")
    sys.exit()


if len(sys.argv) < 3:
    with open(sys.argv[1], 'rb') as data:
        GRAPH = pickle.load(data, encoding='latin1')

    with open(sys.argv[1], 'wb') as data:
        pickle.dump(data, GRAPH)

elif len(sys.argv) < 4:
    with open(sys.argv[1], 'rb') as data:
        GRAPH = pickle.load(data)

    with open(sys.argv[2], 'wb') as data:
        pickle.dump(data, GRAPH)
