"""
Contains the get_neighbors function and associated helper functions.
"""

from enum import Enum
from typing import Union, List, Tuple, Dict, Any

import numpy as np
import shapely.geometry
from g2o import SE3Quat
from shapely.geometry import LineString

from map_processing.transform_utils import se3_quat_average


class _NeighborType(Enum):
    INTERSECTION = 0,
    CLOSE_DISTANCE = 1


def get_neighbors(vertices: np.ndarray, vertex_ids: Union[List[int], None] = None,
                  neighbor_type: _NeighborType = _NeighborType.INTERSECTION)\
        -> Tuple[List[List[int]], List[Dict[str, Any]]]:
    """TODO: documentation

    Args:
        vertices:
        vertex_ids:
        neighbor_type:

    Returns:

    """
    nvertices = vertices.shape[0]
    if vertex_ids is None:
        vertex_ids = list(range(nvertices))
    neighbors = [[vertex_ids[1]]] + [[vertex_ids[i - 1], vertex_ids[i + 1]] for i in range(1, nvertices - 1)] \
        + [[vertex_ids[-2]]]
    curr_id = max(vertex_ids) + 1
    intersections = []

    for id1 in range(1, nvertices):
        for id2 in range(1, id1):
            if neighbor_type == _NeighborType.INTERSECTION:
                intersection = _get_intersection(vertices, id1, id2, curr_id)
                if intersection is None:
                    continue
                intersections.append(intersection)
                neighbors[id1 - 1][-1] = curr_id
                neighbors[id1][0] = curr_id
                neighbors[id2 - 1][-1] = curr_id
                neighbors[id2][0] = curr_id
                curr_id += 1
            elif neighbor_type == _NeighborType.CLOSE_DISTANCE and _is_close_enough(vertices, id1, id2):
                neighbors[id1].append(id2)
                neighbors[id2].append(id1)
                print(f'Point {id1} and {id2} are close enough, adding neighbors')

    return neighbors, intersections


def _get_intersection(vertices, id1, id2, curr_id):
    """TODO: Documentation

    Args:
        vertices:
        id1:
        id2:
        curr_id:

    Returns:

    """
    line1_yval = (vertices[id1 - 1][1] + vertices[id1][1]) / 2
    line2_yval = (vertices[id2 - 1][1] + vertices[id2][1]) / 2
    if abs(line1_yval - line2_yval) > 1:
        return None

    line1 = LineString([(vertices[id1 - 1][0], vertices[id1 - 1][2]),
                        (vertices[id1][0], vertices[id1][2])])
    line2 = LineString([(vertices[id2 - 1][0], vertices[id2 - 1][2]),
                        (vertices[id2][0], vertices[id2][2])])

    intersect_pt = line1.intersection(line2)
    average = se3_quat_average([SE3Quat(vertices[id1 - 1]), SE3Quat(vertices[id1]),
                                SE3Quat(vertices[id2 - 1]), SE3Quat(vertices[id2])]).to_vector()
    if str(intersect_pt) == "LINESTRING EMPTY" or not isinstance(intersect_pt, shapely.geometry.point.Point):
        return None

    print(f'Intersection at {intersect_pt}, between {id1} and {id2}')
    return {
        "translation": {
            "x": intersect_pt.x,
            "y": average[1],
            "z": intersect_pt.y
        },
        'rotation': {
            "x": average[3],
            "y": average[4],
            "z": average[5],
            "w": average[6]
        },
        "poseId": curr_id,
        "neighbors": [id1 - 1, id1, id2 - 1, id2]
    }


def _is_close_enough(vertices, id1, id2):
    """TODO: Documentation

    Args:
        vertices:
        id1:
        id2:

    Returns:

    """
    v1 = vertices[id1]
    v2 = vertices[id2]
    return abs(v1[1] - v2[1]) < 1 and ((v1[0] - v2[0]) ** 2 + (v1[2] - v2[2]) ** 2) ** 0.5 < 1
