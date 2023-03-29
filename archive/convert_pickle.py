#!/usr/bin/env python
import sys
import pickle
from convert_posegraph import convert

if len(sys.argv) < 2:
    print("Usage: convert_pickle.py SRC DEST")
    sys.exit()


if len(sys.argv) < 3:
    with open(sys.argv[1], "rb") as data:
        GRAPH = pickle.load(data, encoding="latin1")

    with open(sys.argv[1], "wb") as data:
        pickle.dump(GRAPH, data)

elif len(sys.argv) < 4:
    with open(sys.argv[1], "rb") as data:
        GRAPH = convert(pickle.load(data))

    with open(sys.argv[2], "wb") as data:
        pickle.dump(GRAPH, data)
