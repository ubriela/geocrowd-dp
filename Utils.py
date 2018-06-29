from collections import defaultdict
import math
from Params import Params
import bisect
import numpy as np
from scipy.stats import rice

#U2U: mean and variance of d^2
def mean_U2U(eps, radius, observed_distance):
    return (4 * radius**2) * (2 + observed_distance**2) / (eps**2)

def variance_U2U(eps, radius, observed_distance):
    return 32 * radius**4 * (1 + observed_distance**2) / (eps**4)

# U2E
def privacy_loss_2_standard_deviation(eps, radius):
    return np.sqrt(2) * radius/eps

def reachable_prob_U2E(reachable_distance, observed_distance, eps, radius):
    sd = privacy_loss_2_standard_deviation(eps, radius)
    return rice.cdf(reachable_distance, observed_distance / sd, scale=sd)

# for q in np.linspace(100,3000,100):
#     print (q, reachable_prob(q, 1000, 0.7,  700))

"""
Return a list of reachable distances, e.g., [1000, 1100,...,5000]
"""
def reachable_distance():
    return np.arange(Params.REACHABLE_DIST_RANGE[0], Params.REACHABLE_DIST_RANGE[1] + 1, Params.REACHABLE_DIST_STEP)

"""
Rounding a reachable distance
"""
def round_reachable_dist(reachable_dist):
    sorted_distances = reachable_distance()
    return int(sorted_distances[bisect.bisect_left(sorted_distances, reachable_dist)])

"""
Generate uniformly and randomly reachable distance
"""
def random_reachable_dist(reachableDistRange, seed):
    np.random.seed(seed)
    return np.random.uniform(reachableDistRange[0], reachableDistRange[1])

def euclidean_2_radian(radian):
    """
    Convert from euclidean scale to radian scale
    :param radian:
    :return:
    """
    return (radian[0] * Params.ONE_KM * 0.001, radian[1] * Params.ONE_KM * 1.2833 * 0.001)


def round_2_grid(point, cell_size, x_offset, y_offset):
    """
    Round the coordinates of a point to the points of a grid.
    :param point: The moint to migrate.
    :param cell_size: Size of the grid to round towards (ndarray)
    :return: The migrated point
    """
    xy = np.array([point[0], point[1]]) - np.array([x_offset, y_offset])
    new_xy = np.round(xy / cell_size) * cell_size + np.array([x_offset, y_offset])

    return new_xy

def distance(lat1, lon1, lat2, lon2):
    """
    Distance between two geographical location (in meters)
    """
    R = Params.EARTH_RADIUS
    dLat = math.radians(abs(lat2 - lat1))
    dLon = math.radians(abs(lon2 - lon1))
    lat1 = math.radians(lat1)
    lat2 = math.radians(lat2)

    a = math.sin(dLat / 2) * math.sin(dLat / 2) + math.sin(dLon / 2) * math.sin(dLon / 2) * math.cos(lat1) * math.cos(
        lat2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = R * c
    return d
# print (distance(34.020412, -118.289936, 34.021969, -118.279983))

"""
convert from workerDict to taskDict
"""
def worker_dict_2_task_dict(workerDict):
    taskDict = defaultdict(set)
    for wid, taskSet in workerDict.items():
        for tid in taskSet:
            taskDict[tid].add(wid)
    return taskDict

# origDict = {}
# origDict[1] = [5,6,7]
# origDict[2] = [5,7]
# print (workerDict2TaskDict(origDict))

# Radix sort for fixed length strings
def radix_sort_passengers(passengers, idx):
    """
    Sort passengers in increasing order of time.
    :param passengers: pass
    :param idx: time index
    :return:
    """
    startIndex = 8 # only consider "hhmmss" in "yyyymmddhhmmss"
    fixedLength = len(passengers[0][idx])
    oa = ord('0'); # First character code
    oz = ord('9'); # Last character code
    n = oz - oa + 1; # Number of buckets
    buckets = [[] for i in range(0, n)] # The buckets
    for position in reversed(range(startIndex, fixedLength)):
        for tuple in passengers:
            string = tuple[2]
            buckets[ord(string[position]) - oa].append(tuple) # Add to bucket
        del passengers[:]
        for bucket in buckets: # Reassemble array in new order
            passengers.extend(bucket)
            del bucket[:]
    return passengers

def radius_eps_2_str(radius, eps):
    return "r" + str(radius) + "_e" + str(eps)

def getParameterizedFile(radius, eps):
    return "output/prob/" + radius_eps_2_str(radius, eps)

"""
Compute range of a distance.
"""
def dist_range(d_prime, step):
    d_prime += 0.001 # make sure distance is greater than 0
    d_prime = min(Params.MAX_DIST, d_prime) # making sure d_prime is not out of simulated range
    result = int(math.ceil(d_prime / step)*step)
    return int(math.ceil(d_prime / step)*step)

"""
Given a set of 'distances', calculate the ratio of distances that are smaller than 'd_threshold'.
"""
def cumulative_prob(distances, d_threshold):
    # print (distances, d_threshold)
    return float(bisect.bisect_left(distances, d_threshold)) / len(distances) if distances else 0

# print cumulative_prob(range(1000), 800)

"""
Compute reachable distance such the coverage probability is greater or equal to a threshold
"""
def reachable_noisy_dist(sortedPairs, recallThreshold):
    idx = bisect.bisect_left(list(sortedPairs.values()), recallThreshold)
    return list(sortedPairs.keys())[min(idx, len(sortedPairs) - 1)]

# sortedPairs = [(1,3),(2,5),(3,6),(4,9),(5,9)]
# print (reachableNoisyDist(sortedPairs, 5))

# str = """
#         20121101001734
#         20121101001730
#         20121101001732
#         20121101001734
#         20121101001730
#         20121101001736
#         20121101001730
#     """
# arr = str.split()
#
# out = radixSortFixedString(arr)
#
# print (out)