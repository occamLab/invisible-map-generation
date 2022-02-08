"""
Weights class.
"""

from typing import Optional, Union, List, Dict

import numpy as np
from map_processing.graph_vertex_edge_classes import VertexType
from map_processing import ASSUMED_FOCAL_LENGTH


class Weights:
    def __init__(self, odometry: Optional[np.ndarray] = None, tag: Optional[np.ndarray] = None,
                 tag_sba: Optional[np.ndarray] = None, gravity: Optional[np.ndarray] = None,
                 odom_tag_ratio: Optional[Union[np.ndarray, float]] = None):
        self.gravity: np.ndarray = np.array(gravity) if gravity is not None else np.ones(3)
        self.odometry: np.ndarray = np.array(odometry) if odometry is not None else np.ones(6)
        self.tag: np.ndarray = np.array(tag) if tag is not None else np.ones(6)
        self.tag_sba: np.ndarray = np.array(tag_sba) if tag is not None else np.ones(2)

        # Put lower limit of 0.00001 to prevent rounding causing division by 0
        self.odom_tag_ratio: float
        if isinstance(odom_tag_ratio, float):
            self.odom_tag_ratio = max(0.00001, odom_tag_ratio)
        elif isinstance(odom_tag_ratio, np.ndarray):
            self.odom_tag_ratio = max(0.00001, odom_tag_ratio[0])
        else:
            self.odom_tag_ratio = 1

    @property
    def tag_odom_ratio(self):
        return 1 / self.odom_tag_ratio

    @classmethod
    def legacy_from_array(cls, array: Union[np.ndarray, List[float]]) -> "Weights":
        return cls.legacy_from_dict(cls.legacy_weight_dict_from_array(array))

    @classmethod
    def legacy_from_dict(cls, dct: Dict[str, Union[np.ndarray, float]]) -> "Weights":
        return cls(**dct)

    def to_dict(self) -> Dict[str, Union[float, np.ndarray]]:
        return {
            "gravity": np.array(self.gravity),
            "odometry": np.array(self.odometry),
            "tag": np.array(self.tag),
            "tag_sba": np.array(self.tag_sba),
            "odom_tag_ratio": self.odom_tag_ratio
        }

    @staticmethod
    def legacy_weight_dict_from_array(array: Union[np.ndarray, List[float]]) -> Dict[str, Union[float, np.ndarray]]:
        """Construct a normalized weight dictionary from a given array of values using the legacy approach.
        """
        weights = {
            'gravity': np.array([-1, 1e2, -1]),
            'odometry': np.ones(6),
            'tag': np.ones(6),
            'tag_sba': np.ones(2),
            'odom_tag_ratio': 1
        }

        length = array.size if isinstance(array, np.ndarray) else len(array)
        half_len = length // 2
        has_ratio = length % 2 == 1

        if length == 1:  # ratio
            weights['odom_tag_ratio'] = array[0]
        elif length == 2:  # tag/odom pose:rot/tag-sba x:y, ratio
            weights['odometry'] = np.array([array[0]] * 3 + [1] * 3)
            weights['tag'] = np.array([array[0]] * 3 + [1] * 3)
            weights['tag_sba'] = np.array([array[0], 1])
            weights['odom_tag_ratio'] = array[1]
        elif length == 3:  # odom pose:rot, tag pose:rot/tag-sba x:y, ratio
            weights['odometry'] = np.array([array[0]] * 3 + [1] * 3)
            weights['tag'] = np.array([array[1]] * 3 + [1] * 3)
            weights['tag_sba'] = np.array([array[1], 1])
            weights['odom_tag_ratio'] = array[2]
        elif half_len == 2:  # odom pose, odom rot, tag pose/tag-sba x, tag rot/tag-sba y, (ratio)
            weights['odometry'] = np.array([array[0]] * 3 + [array[1]] * 3)
            weights['tag'] = np.array([array[2]] * 3 + [array[3]] * 3)
            weights['tag_sba'] = np.array(array[2:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif half_len == 3:  # odom x y z qx qy, tag-sba x, (ratio)
            weights['odometry'] = np.array(array[:5])
            weights['tag_sba'] = np.array([array[5]])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 4:  # odom, tag-sba, (ratio)
            weights['odometry'] = np.array(array[:6])
            weights['tag_sba'] = np.array(array[6:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 5:  # odom x y z qx qy, tag x y z qx qy, (ratio)
            weights['odometry'] = np.array(array[:5])
            weights['tag'] = np.array(array[5:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        elif length == 6:  # odom, tag, (ratio)
            weights['odometry'] = np.array(array[:6])
            weights['tag'] = np.array(array[6:])
            weights['odom_tag_ratio'] = array[-1] if has_ratio else 1
        else:
            raise Exception(f'Weight length of {length} is not supported')

        w = Weights.legacy_from_dict(weights)
        w.scale_tag_and_odom_weights(normalize=True)
        return w.to_dict()

    def scale_tag_and_odom_weights(self, normalize: bool = False):
        """Apply the odom-to-tag ratio as a scaling factor to the odometry and tag vectors; divide the tag_sba vector
         by the camera's focal length.

        Args:
            normalize: If true, add a multiplicative factor that is the reciprocal of each vector's magnitude.
        """
        if normalize:
            odom_mag = np.linalg.norm(self.odometry)
            if odom_mag == 0:  # Avoid divide by zero error
                odom_mag = 1

            sba_mag = np.linalg.norm(self.tag_sba)
            if sba_mag == 0:
                sba_mag = 1  # Avoid divide by zero error

            tag_mag = np.linalg.norm(self.tag)
            if tag_mag == 0:  # Avoid divide by zero error
                tag_mag = 1
        else:
            odom_mag = 1
            sba_mag = 1
            tag_mag = 1

        self.odometry *= self.odom_tag_ratio / odom_mag
        # TODO: The below implements what was previously in place for SBA weighting. Should it be changed? Why is
        #  such a low weighting so effective?
        self.tag_sba *= 1 / (sba_mag * ASSUMED_FOCAL_LENGTH)
        self.tag *= 1 / tag_mag
    
    def get_weights_from_end_vertex_mode(self, end_vertex_mode: Optional[VertexType]):
        """
        Args:
            end_vertex_mode: Mode of the end vertex of the edge

        Returns:
            A copy of the edge weight vector selected according to the mode of an edge's end vertex. An end vertex mode
             of type waypoint returns a vector of 1s.

        Raises:
            ValueError: If the end_vertex_mode is not recognized
        """
        if end_vertex_mode == VertexType.ODOMETRY:
            return np.array(self.odometry)
        elif end_vertex_mode == VertexType.TAG:
            return np.array(self.tag)
        elif end_vertex_mode == VertexType.TAGPOINT:
            return np.array(self.tag_sba)
        elif end_vertex_mode is None:
            return np.array(self.gravity)
        elif end_vertex_mode == VertexType.WAYPOINT:
            return np.ones(6)  # TODO: set to something other than identity?
        else:
            raise Exception(f"Edge of end type {end_vertex_mode} not recognized")
