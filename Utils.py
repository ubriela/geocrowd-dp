from collections import defaultdict
import math
import numpy as np
from Params import Params
import bisect

def euclideanToRadian(radian):
    """
    Convert from euclidean scale to radian scale
    :param radian:
    :return:
    """
    return (radian[0] * Params.ONE_KM * 0.001, radian[1] * Params.ONE_KM * 1.2833 * 0.001)


def round2Grid(point, cell_size, x_offset, y_offset):
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
def workerDict2TaskDict(workerDict):
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
def radixSortPassengers(passengers, idx):
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

def RadiusEps2Str(radius, eps):
    return "r" + str(radius) + "_e" + str(eps)

def getParameterizedFile(radius, eps):
    return "output/prob/" + RadiusEps2Str(radius, eps)

"""
Compute range of a distance
"""
def dist_range(d_prime, step):
    d_prime = min(Params.max_dist, d_prime) # making sure d_prime is not out of simulated range
    return math.ceil(d_prime / step)*step

def cumulative_prob(distances, d_reachable):
    return bisect.bisect_left(distances, d_reachable) / len(distances) if distances else 0

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