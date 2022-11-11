"""
Benchmark RTABmap's effectiveness as a ground truth.
NOTE: only checks positional accuracy, doesn't check rotational accuracy technically.
"""

import numpy as np
import pandas as pd

raw_tags = pd.read_csv("r_b_data/mac_raw_tags.csv")

print(raw_tags)

class Map:
    """
    Turns a list of tags into a map
    """
    
    def __init__(self, tags):
        self.tags = tags
        
    def error(self):
        self.
    
class Tag:
    
    def __init__(self,id, x,y,z):
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        
    def rotate(self, theta):
        """
        Rotate a map of tags by some theta

        Args:
            theta (float): some amount (in radians) to rotate a tag
        """
        self.x = self.x*theta
        self.y = self.y*theta
        self.z = self.z*theta
        
    def translate(self, translation):
        """
        Translate the tag by some translation vector

        Args:
            translation (float): some amount (in meters) to translate a tag
            
        """