import numpy as np
import Utils
from Differential import Differential
from Params import Params
from collections import defaultdict, OrderedDict

p = Params(1000)
p.select_dataset()
reachable_range = Utils.reachableDistance()
dp = Differential(p.seed)

# Randomly picking location in a small MBR of tdrive dataset.
minLat, maxLat = 39.1232147, 40.7225952
minLon, maxLon = 115.3879166, 117.3795395

diffLat = maxLat - minLat
diffLon = maxLon - minLon

maxLat = maxLat - 0.95*diffLat
maxLon = maxLon - 0.95*diffLon

def variance_from_sampling(p):
    distances = []
    for i in range(samples):
        lat, lon = (minLat+maxLat)/2, (minLon + maxLon)/2  # Fix one location
        noisyLat, noisyLon = dp.addPolarNoise(p.eps, p.radius, (lat, lon))
        distance = Utils.distance(lat, lon, noisyLat, noisyLon)
        distances.append(distance)
    return np.var(distances)

samples = 10000  # sample size
def precomputeVariance():
    """
    Precompute probability of reachability given epsilon
    :param eps:
    :return:
    """
    dict = OrderedDict()
    for radius in [100.0, 400.0, 700.0, 1000.0]:
        for eps in [0.1, 0.4, 0.7, 1.0]:
            p.radius, p.eps = radius, eps
            var = variance_from_sampling(p)
            dict[Utils.RadiusEps2Str(radius, eps)] = var
            sd = Utils.privacyLossToStandardDeviation(eps)
            print (var, sd**2 * radius**2)

    return dict

# print (precomputeVariance())