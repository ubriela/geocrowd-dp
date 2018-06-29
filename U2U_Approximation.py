import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import math
from collections import defaultdict
from scipy.stats import norm

from Differential import Differential
from Params import Params
import Utils

radius = 200.0
eps = 1.0
observed_distace = 2000.0

# radius, eps, dp = 200.0, 1.0, 2000

mu = Utils.mean_U2U(eps, radius, observed_distace)
variance = Utils.variance_U2U(eps, radius, observed_distace)
sigma = math.sqrt(variance)
print "mean %s\t sigma %s" % (mu, sigma)

# s = np.random.normal(mu, sigma, 10000)
# x = np.linspace(mu - 3*sigma, mu + 3*sigma, 100)
# plt.plot(x, mlab.normpdf(x, mu, sigma))
# plt.show()

# norm.stats(loc=mu, scale=sigma, moments="mv")

mu, sigma = 4.26246819779, math.sqrt(3.68211892668)

for i in range(200):
    print math.sqrt(random.gauss(mu, sigma))
    # print norm.pdf(loc=mu, scale=sigma)

# for rd in np.arange(1000, 3001, 50):
#     x2 = (rd/1000.0)**2
#     print "%.1f\t%s" % (x2, norm.pdf(x2, loc=mu, scale=sigma))

p = Params(1000)
p.select_dataset()
dp = Differential(p.seed)

# Randomly picking location in a small MBR of tdrive dataset.
minLat, maxLat = 39.1232147, 40.7225952
minLon, maxLon = 115.3879166, 117.3795395

# diffLat = maxLat - minLat
# diffLon = maxLon - minLon
#
# maxLat = maxLat - 0.95*diffLat
# maxLon = maxLon - 0.95*diffLon

samples = 200000  # sample size
d2_list = []
for i in range(samples):
    # First point
    lat1, lon1 = random.uniform(minLat, maxLat), random.uniform(minLon, maxLon)
    noisyLat1, noisyLon1 = dp.addPolarNoise(eps, radius, (lat1, lon1))

    # Second point
    lat2, lon2 = random.uniform(minLat, maxLat), random.uniform(minLon, maxLon)
    noisyLat2, noisyLon2 = dp.addPolarNoise(eps, radius, (lat2, lon2))

    d = Utils.distance(lat1, lon1, lat2, lon2)
    d_prime = Utils.distance(noisyLat1, noisyLon1, noisyLat2, noisyLon2)
    if abs(d_prime/1000.0 - observed_distace/1000.0) <= 100/1000.0:
        d2_list.append(d**2/1000000.0)
        # print d**2/1000000.0

print np.mean(d2_list), np.var(d2_list)
d2_list.sort()
for d2 in d2_list:
    print math.sqrt(d2)

# hist, bin_edges = np.histogram(d2, bins=10, range=(0,4,0.5))
# print hist, bin_edges