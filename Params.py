__author__ = 'ubriela'

import numpy as np
import math

class Params(object):
    """
    Constants
    """
    DATASET = "tdrive"
    MIN_SENSITIVITY = 1e-4*5
    PRECISION = 1e-15
    GEOI_GRID_SIZE = 5000
    EARTH_RADIUS = 6378137 # const, in meters
    ONE_KM = 0.0089982311916  # convert km to degree
    STEP = 100  # step of d_prime range
    MAX_DIST = 11900 # max distance used in simulation
    REACHABLE_DIST_STEP = 50.0
    REACHABLE_DIST_RANGE = (1000.0, 3000.0)
    def __init__(self, seed):
        self.workerFile = ""
        self.taskFile = ""

        self.seed = seed # used in generating noisy counts

        """
        Privacy parameters
        """
        self.radius = 1400.0  # default unit is meters
        self.eps = 0.7  # epsilon

        """
        Geocrowd parameters
        """
        self.workerCount = 500
        self.taskCount = 500

        """
        Heuristic parameters:
        @reachabilityThresholdU2U: minimum probability of reachability to a task during U2U,
            used by server to determine the candidate workers
        @reachabilityThresholdU2E: minimum probability of reachability to a task during U2E,
            used by requesters to cancel her task to limit task disclosure
        """
        self.reachabilityThresholdU2U = 0.1
        self.reachabilityThresholdU2E = 0.4

        """
        Parameters of elastic metric (not used)
        """
        self.m = 1000 # granularity of equal-size grid cell

    def select_dataset(self):
        if Params.DATASET == "tdrive":
            self.workerFile = "dataset/tdrive/vehicles.txt"
            self.taskFile = "dataset/tdrive/passengers.txt"
            self.resdir = "output/"
            self.x_min = 39.1232147
            self.y_min = 115.3879166
            self.x_max = 40.7225952
            self.y_max = 117.3795395

