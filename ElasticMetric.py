import collections
import csv
import numpy as np

def cellId2Coord(cellId, p):
    """
    Convert from cell id to lat/lon
    :param cellId:
    :param p:
    :return:
    """
    lat_idx = cellId/p.m
    lon_idx = cellId - lat_idx * p.m
    lat = float(lat_idx)/p.m * (p.x_max - p.x_min) + p.x_min
    lon = float(lon_idx)/p.m * (p.y_max - p.y_min) + p.y_min
    return (lat, lon)

def coord2CellId(point, p):
    """
    Convert from lat/lon to cell id
    :param point:
    :param p:
    :return:
    """
    lat, lon = point[0], point[1]
    lat_idx = int((lat - p.x_min) / (p.x_max - p.x_min) * p.m)
    lon_idx = int((lon - p.y_min) / (p.y_max - p.y_min) * p.m)
    cellId = lat_idx * p.m + lon_idx
    return cellId

def locationCount(locs, p):
    """
    partition space into an equal-size grid, then count the number of locations within each grid
    :param p:
    :return: a count for each cell id
    """
    count = collections.defaultdict(int)
    for lat, lon, checkins in iter(locs):
        cellId = coord2CellId((lat, lon), p)
        count[cellId] = count.get(cellId, 0) + checkins # this equation can be tailored to specific application

    return count

# Compute location quality from Vemo
def locationQuality(p, file="dataset/weibo/checkins_filtered.txt"):
    locs = []
    with open(file) as worker_file:
        reader = csv.reader(worker_file, delimiter='\t')
        for row in reader:
            locs.append((float(row[1]), float(row[2]), int(row[3])))  # lat, lon, id
    count = locationCount(locs, p)
    print("number of non-empty cells", len(count))
    print("average value per cell", np.mean(list(count.values())))

# Given cell x, return the closest unvisited cell (vertex) x' and connect them.
def next_by_geodistance(x, ):
    """
    :param x: cell id
    :return:
    """



# Create elastic metric from Vemo (constructing a weighted graph)
def indistinguishabilityMetric(p):
    """
    The goal is to build an indistinguishability metric, approximated by a graph.
    The distance is then the shortest path on this graph.
    The algorithm to connect the graph so that it satisfies a certain privacy requirement.
    The graph starts with all vertices disconnected (thus at infinite distance) and then iteratively adds edges.
    From a vertex x you pick a closest unvisited vertex x' and connect (x,x').
    To chose a weight for the edge you compute the mass of the reachable set from x.
    Then using the inverse of the requirement you obtain a what distance you need to have that
    much mass and you use that distance for the new edge from x to x’.
    """

# Implement exponential mechanism given the weighted graph.
def exponentialAlgorithm():
    """
    From the theory of dx-privacy, define a metric and use it to scale the noise addition,
    then the mechanism satisfies the definition. Formal proofs are in paper
    Broadening the Scope of Differential Privacy Using Metrics. PETS'13
    :return:
    """
