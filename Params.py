__author__ = 'ubriela'

import numpy as np
import math

class Params(object):
    """
    Constants
    """
    DATASET = "geolife"
    MIN_SENSITIVITY = 1e-4*5
    PRECISION = 1e-15
    GEOI_GRID_SIZE = 5000
    EARTH_RADIUS = 6378137 # const, in meters
    ONE_KM = 0.0089982311916  # convert km to degree

    def __init__(self, seed):
        self.workerFile = ""
        self.taskFile = ""

        """
        Privacy parameters
        """
        self.radius = 500.0  # default unit is meters
        self.eps = 1.0  # epsilon
        self.confidence = 0.95

        """
        Geocrowd parameters
        """
        self.workerCount = 500
        self.taskCount = 10000
        self.reachableDist = 2000.0

        self.seed = seed # used in generating noisy counts

    def select_dataset(self):
        if Params.DATASET == "geolife":
            self.workerFile = "dataset/geolife/vehicles.txt"
            self.taskFile = "dataset/geolife/passengers.txt"
            self.resdir = "output/"
            self.x_min = 39.1232147
            self.y_min = 115.3879166
            self.x_max = 40.7225952
            self.y_max = 117.3795395

