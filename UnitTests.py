from ElasticMetric import locationCount

import unittest
import logging
import csv
import numpy as np

from Params import Params

class MyTestCase(unittest.TestCase):
    def setUp(self):
        # init parameters
        self.p = Params(1000)
        self.p.select_dataset()

        self.log = logging.getLogger("debug.log")

    def testlocationCount(self):
        locs = []
        with open("dataset/weibo/checkins_filtered.txt") as worker_file:
            reader = csv.reader(worker_file, delimiter='\t')
            for row in reader:
                locs.append((float(row[1]), float(row[2]), int(row[3])))  # lat, lon, id
        count = locationCount(self.p, locs)
        print ("number of non-empty cells", len(count))
        print ("average value per cell", np.mean(list(count.values())))