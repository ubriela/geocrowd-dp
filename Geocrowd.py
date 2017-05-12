import csv
import copy
import random
import Utils
import networkx as nx
import numpy as np

from collections import defaultdict, Counter
from Params import Params

"""
Counting the number of satisfiable matches, which have worker-task distance smaller than a reachable distance
"""
def satisfiableMatches(matches, wLoc, tLoc, reachableDist):
    c = defaultdict(list)
    count = 0
    for tid, wid in matches:
        c[wid].append(tid)

    for wid, taskSet in c.items():
        for tid in taskSet:
            # any satisfiable worker-task --> break
            if Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[wid][0], wLoc[wid][1]) <= reachableDist:
                count += 1
                break
    return count

"""
Add noise to worker or task list
"""
def perturbedData(wtList, p, dp):
    """

    :param wtList: worker or task list [(lat, lon, id), ...]
    :param p: parameters
    :param dp: differential privacy object
    :return: perturbed list
    """
    perturbedList = []

    u = Utils.distance(p.x_min, p.y_min, p.x_max, p.y_min) / Params.GEOI_GRID_SIZE
    v = Utils.distance(p.x_min, p.y_min, p.x_min, p.y_max) / Params.GEOI_GRID_SIZE
    rad = Utils.euclideanToRadian((u, v))
    cell_size = np.array([rad[0], rad[1]])

    for lat, lon, id in wtList:
        noisyLat, noisyLon = dp.addPolarNoise(p.eps, p.radius, (lat, lon))  # perturbed noisy location
        roundedLoc = Utils.round2Grid((noisyLat, noisyLon), cell_size, p.x_min, p.y_min) # rounded to grid
        perturbedList.append((roundedLoc[0], roundedLoc[1], id))

    return perturbedList

# DG = nx.DiGraph()
# for tuple in [("s",0,2), ("s",1,9), (0,1,1), (1,0,2), (0,3,5), (1,3,4)]:
# print (DG.out_degree(2,weight='weight'))
# print (DG.successors(2))
# print (DG.neighbors(0))
# print(nx.maximum_flow_value(DG, "s", 3))

"""
Create max-flow from dict
"""
def maxFlowValue(workers, taskids, b=1):
    """
    Compute max-flow between source s and destination d
    :param workers:
    :param taskids:
    :param b:
    :return:
    """
    DG = nx.DiGraph()
    # create edges from source to workers' nodes
    for wid in workers.keys():
        DG.add_edge("s", wid, capacity=b)

    # create edges from tasks' nodes to destination
    for tid in taskids:
        DG.add_edge(tid, "d", capacity=1)

    # create edges between workers and tasks
    for wid, taskSet in workers.items():
        for tid in taskSet:
            DG.add_edge(wid, tid, capacity=1)

    return nx.maximum_flow_value(DG, "s", "d")

"""
Sample workers and tasks data
"""
def sampleWorkersTasks(workerFile, taskFile, wCount, tCount):
    """

    :param workerFile:
    :param taskFile:
    :param wCount: worker count
    :param tCount: task count
    :return: worker list: [(lat, lon, id), ...] and task list [(lat, lon, id), ...]
    """
    wList, taskList = [], []
    wLoc, tLoc = {}, {}
    w, t = 0, 0
    with open(workerFile) as worker_file:
        reader = csv.reader(worker_file, delimiter=',')
        for row in reader:
            lat, lon, wid = float(row[0]), float(row[1]), int(row[2])
            wList.append((lat, lon, wid)) # lat, lon, id
            wLoc[wid] = (lat, lon)
            w += 1
            if w == wCount:
                break
    with open(taskFile) as task_file:
        reader = csv.reader(task_file, delimiter=',')
        for row in reader:
            lat, lon, tid = float(row[0]), float(row[1]), t
            taskList.append((lat, lon, tid)) # lat, lon, id (use counter as taskid)
            tLoc[tid] = (lat, lon)
            t += 1
            if t == tCount:
                break
    worker_file.close()
    task_file.close()
    return wList, taskList, wLoc, tLoc

# print (sampleWorkersTasks("./dataset/geolife/vehicles.txt", "./dataset/geolife/passengers.txt", 10, 5))


"""
Crete bipartite graph from workers and tasks list
"""
def createBipartiteGraph(wList, tList, reachableDist):
    """

    :param wList: array of workers
    :param tList: array of tasks
    :param reachableDist: in meters
    :return:
    """
    workers = defaultdict(set)
    """
    for each worker, find a set of reachable tasks
    """
    for wlat, wlon, wid in wList:
        for tlat, tlon, tid in tList:
            dist = Utils.distance(wlat, wlon, tlat, tlon)
            # print (dist)
            if dist <= reachableDist:
                workers[wid].add(tid)
    return workers

# wList, tList = sampleWorkersTasks("./dataset/geolife/vehicles.txt", "./dataset/geolife/passengers.txt", 1000, 1000)
# print (createBipartiteGraph(wList, tList, 1000))

"""
Implementation of Ranking algorithm for online bipartite matching.
Citation: Karp et al. An Optimal Algorithm for On-line Bipartite Matching
"""
def rankingAlgo(orgWorkers, taskids):
    """

    :param orgWorkers: map each workerid to a list of nearby tasks
    :param taskids: list of taskids arriving in order
    :return: a list matching pairs
    """
    matches = []
    workers = copy.deepcopy(orgWorkers)
    tasks = Utils.workerDict2TaskDict(workers) # create dict with key = taskid

    # random permutation of workers' ranks
    randomRanks = list(range(len(workers)))
    random.shuffle(randomRanks)
    workerRank = dict([(wid, randomRanks[i]) for i, wid in enumerate(workers.keys())])
    # sortedWorkerRank = sorted(workerRank.items(), key=lambda x:x[1], reversed=False)

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            if len(eligibleWids) > 0:
                # find the worker of highest rank
                highestRankWid = max(eligibleWids, key=lambda x:workerRank[x])
                matches.append((tid, highestRankWid))
                # matches += 1

                # affected tasks
                affectedTids = workers[highestRankWid]

                # delete highestRankWid from workers, tid from tasks
                del workers[highestRankWid]
                del tasks[tid]

                # delete from workerRank & tasks
                # del workerRank[highestRankWid]

                for _tid in affectedTids:
                    if _tid != tid:
                        tasks[_tid].discard(highestRankWid)
                        if len(tasks[_tid]) == 0:
                            del tasks[_tid]

    return matches

# workers = {0:set([1,2]), 2:set([0,3]), 3:set([2]), 4:set([2,3]), 5:set([5])}
# taskids = [0,1,2,3,4,5]
# print (rankingAlgo(workers, taskids))

"""
Implementation of balance algorithm for online b-matching.
Citation: Kalyanasundaram and Kruhs. An optimal deterministic algorithm for online b-matching
"""
def balanceAlgo(orgWorkers, taskids, b):
    """

    :param orgWorkers: map each workerid to a list of nearby tasks
    :param taskids: list of taskids arriving in order
    :param b: at most b tasks can be matched to one worker
    :return: a list matching pairs
    """
    matches = []
    workers = copy.deepcopy(orgWorkers)
    tasks = Utils.workerDict2TaskDict(workers) # create dict with key = taskid

    # initialize maximum number of tasks matched to each worker
    workerCapacity = Counter(dict([(wid, b) for wid in workers.keys()]))

    for tid in taskids:     # iterate through task list
        if tid in tasks:    # if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            if len(eligibleWids) > 0:
                # find the worker of highest remaining capacity
                maxCapacityWid = max(eligibleWids, key=lambda x:workerCapacity[x])
                matches.append((tid, maxCapacityWid))
                # matches += 1

                affectedTids = workers[maxCapacityWid]  # affected tasks

                del tasks[tid] # remove tid from tasks

                # update workerCapacity and workers
                workerCapacity[maxCapacityWid] -= 1
                if workerCapacity[maxCapacityWid] == 0: # disable this worker
                    del workerCapacity[maxCapacityWid]
                    del workers[maxCapacityWid]
                else:
                    workers[maxCapacityWid].discard(tid)
                    if len(workers[maxCapacityWid]) == 0:
                        del workers[maxCapacityWid]
                        del workerCapacity[maxCapacityWid]

                for _tid in affectedTids:
                    # remove from tasks
                    if _tid != tid and maxCapacityWid not in workerCapacity:
                        tasks[_tid].discard(maxCapacityWid)
                        if len(tasks[_tid]) == 0:
                            del tasks[_tid]

    return matches

# workers = {0:set([1,2,6]), 2:set([0,3]), 3:set([2]), 4:set([2,3]), 5:set([5])}
# taskids = [0,1,2,3,4,5,6]
# print (balanceAlgo(workers, taskids, 1))
