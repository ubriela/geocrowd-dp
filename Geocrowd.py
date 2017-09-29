import csv
import copy
import random
import Utils
import networkx as nx
import numpy as np

from collections import defaultdict, Counter
from Params import Params

reachable_range = Utils.reachableDistance()
def flowDict2Matches(flowDict):
    """
    Create a list of worker-task matches from flow_dict
    :param flowDict:
    :return:
    """
    matches = []
    for wid in flowDict.keys():
        if wid != "s":
            for tid in flowDict[wid].keys():
                if tid != "d":
                    if flowDict[wid][tid] == 1:
                        matches.append((tid, wid))
    return matches

"""
Counting the number of satisfiable matches, which have worker-task distance smaller than a reachable distance
"""
def satisfiableMatches(matches, wLoc, tLoc, reachable_distance_map):
    c = defaultdict(list)
    count = 0
    total_travel_dist = 0.0
    for tid, wid in matches:
        c[wid].append(tid)

    for wid, taskSet in c.items():
        for tid in taskSet:
            # any satisfiable worker-task --> break
            dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[wid][0], wLoc[wid][1])
            if dist <= reachable_distance_map[wid]:
                count += 1
                total_travel_dist += dist
                break
    return count, total_travel_dist/count

def perturbedData(wtList, p, dp):
    """
    Add noise to worker or task list
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
        perturbedList.append([roundedLoc[0], roundedLoc[1], id])

    return perturbedList

# DG = nx.DiGraph()
# for tuple in [("s",0,2), ("s",1,9), (0,1,1), (1,0,2), (0,3,5), (1,3,4)]:
# print (DG.out_degree(2,weight='weight'))
# print (DG.successors(2))
# print (DG.neighbors(0))
# print(nx.maximum_flow_value(DG, "s", 3))

def createDG(workers, taskids, b=1):
    """
    Create directed graph between worker nodes and task nodes
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
    return DG

"""
Compute max-flow from dict
"""
def maxFlowValue(workers, taskids, b=1):
    """
    Compute max-flow between source s and destination d
    :param workers:
    :param taskids:
    :param b:
    :return:
    """
    DG = createDG(workers, taskids, b)
    return nx.maximum_flow_value(DG, "s", "d")

"""
Compute max-flow from dict
"""
def maxFlow(workers, taskids, b=1):
    """
    Compute max-flow between source s and destination d
    :param workers:
    :param taskids:
    :param b:
    :return:
    """
    DG = createDG(workers, taskids, b)
    return nx.maximum_flow(DG, "s", "d")

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

# print (sampleWorkersTasks("./dataset/tdrive/vehicles.txt", "./dataset/tdrive/passengers.txt", 10, 5))

"""
Crete bipartite graph from workers and tasks list
"""
def createBipartiteGraph(wList, tList, reachable_distance_map):
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
            noisy_dist = Utils.distance(wlat, wlon, tlat, tlon)
            reachable_distance = reachable_distance_map[wid]
            if noisy_dist <= reachable_distance:
                workers[tid].add(wid)
    return workers

def createBipartiteGraphU2U(wList, tList, reachable_distance_map, reachable_prob_U2U, p):
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
            reachable_distance = reachable_distance_map[wid]
            noisy_dist = Utils.distance(wlat, wlon, tlat, tlon)
            reachable_prob = reachable_prob_U2U[
                str(Utils.round_reachable_dist(reachable_distance)) + ":" +
                str(Utils.dist_range(noisy_dist, Params.step))
            ]
            # reachability >= a threshold
            if reachable_prob >= p.reachabilityThresholdU2U:
                workers[tid].add(wid)
    return workers

# wList, tList = sampleWorkersTasks("./dataset/tdrive/vehicles.txt", "./dataset/tdrive/passengers.txt", 1000, 1000)
# print (createBipartiteGraph(wList, tList, 1000))

"""
Implementation of Ranking algorithm for online bipartite matching.

Citation: Karp et al. An Optimal Algorithm for On-line Bipartite Matching

Ranking based on random rank.
"""
def rankingRandom(candidateWorkers, taskids, wLoc, tLoc, reachable_distance_map):
    """
    :param candidateWorkers: map each workerid to a list of nearby tasks
    :param taskids: list of taskids arriving in order
    :return: a list matching pairs
    """
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    # random permutation of workers' ranks
    randomRanks = list(range(len(workers)))
    random.shuffle(randomRanks)
    workerRank = dict([(wid, randomRanks[i]) for i, wid in enumerate(workers.keys())])

    total_notified_workers = 0
    total_travel_dist = 0.0

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            if len(eligibleWids) > 0:
                # find the worker of highest rank
                highestRankWid = min(eligibleWids, key=lambda w:workerRank[w])
                total_notified_workers += 1
                actual_dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0],
                                             wLoc[highestRankWid][1])
                if actual_dist <= reachable_distance_map[highestRankWid]:
                    matches.append((tid, highestRankWid))
                    total_travel_dist += actual_dist
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

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist / matched_count
    return matched_count, false_hits, average_travel_dist, total_notified_workers

"""
Similar to above function but ranking based on distance.
"""
def rankingNearest(candidateWorkers, taskids, wLoc, tLoc, reachable_distance_map):
    """
    :param candidateWorkers: map each workerid to a list of nearby tasks
    :param taskids: list of taskids arriving in order
    :return: a list matching pairs
    """
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    total_notified_workers = 0
    total_travel_dist = 0.0

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            if len(eligibleWids) > 0:
                # find the worker of highest rank
                highestRankWid = min(eligibleWids, key=lambda w:Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[w][0], wLoc[w][1]))
                total_notified_workers += 1

                actual_dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0],
                                             wLoc[highestRankWid][1])
                if actual_dist <= reachable_distance_map[highestRankWid]:
                    matches.append((tid, highestRankWid))
                    total_travel_dist += actual_dist

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

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist / matched_count
    return matched_count, false_hits, average_travel_dist, total_notified_workers

# workers = {0:set([1,2]), 2:set([0,3]), 3:set([2]), 4:set([2,3]), 5:set([5])}
# taskids = [0,1,2,3,4,5]
# print (rankingAlgo(workers, taskids))

"""
Modify ranking algorithm such that each worker is selected by the highest reachable probability rather than
the highest random rank. This technique not only increases the number of performed tasks but also reduces the
travel distance.
"""
def rankingByReachability(candidateWorkers, taskids, wLoc, tLoc, reachable_prob_U2E, reachable_distance_map):
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workers

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            if len(eligibleWids) > 0:
                # find the worker of highest probability of reachability
                highestRankWid = max(eligibleWids, key=lambda wid:reachable_prob_U2E[
                    str(Utils.round_reachable_dist(reachable_distance_map[wid])) + ":" +
                    str(Utils.dist_range(Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[wid][0], wLoc[wid][1]), Params.step))
                ])
                matches.append((tid, highestRankWid))

                # affected tasks
                affectedTids = workers[highestRankWid]

                # delete highestRankWid from workers, tid from tasks
                del workers[highestRankWid]
                del tasks[tid]

                for _tid in affectedTids:
                    if _tid != tid:
                        tasks[_tid].discard(highestRankWid)
                        if len(tasks[_tid]) == 0:
                            del tasks[_tid]

    return matches

"""
Modify ranking algorithm such that in each iteration task is sent to a matched worker,
the worker responses to the requester if the task is reachable. Otherwise, the task will be sent
to the next matched worker. This stragegy helps to increase the number of performed tasks.

Note: We assume that workers would accept their matched tasks as long as the tasks are reachable.
"""
def rankingResendRandom(candidateWorkers, taskids, wLoc, tLoc, reachable_distance_map):
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    # random permutation of workers' ranks
    randomRanks = list(range(len(workers)))
    random.shuffle(randomRanks)
    workerRank = dict([(wid, randomRanks[i]) for i, wid in enumerate(workers.keys())])

    # total disclosure
    total_notified_workers = 0
    total_travel_dist = 0.0

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            matched = False # if matched is True, go to the next task
            while len(eligibleWids) > 0 and not matched:
                # find the worker of the highest rank.
                highestRankWid = max(eligibleWids, key=lambda x:workerRank[x])
                actual_dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0], wLoc[highestRankWid][1])
                total_notified_workers += 1

                if actual_dist <= reachable_distance_map[highestRankWid]: # satisfiable match
                    matches.append((tid, highestRankWid))
                    total_travel_dist += actual_dist
                    matched = True
                else:
                    eligibleWids.remove(highestRankWid)

                if matched:
                    # affected tasks
                    affectedTids = workers[highestRankWid]

                    # delete highestRankWid from workers, tid from tasks
                    del workers[highestRankWid]
                    del tasks[tid]

                    for _tid in affectedTids:
                        if _tid != tid:
                            tasks[_tid].discard(highestRankWid)
                            if len(tasks[_tid]) == 0:
                                del tasks[_tid]

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist/matched_count if matched_count != 0 else 0
    return matched_count, false_hits, average_travel_dist, total_notified_workers

"""
Similar to above function except ranking by distance.
"""
def rankingResendNearest(candidateWorkers, taskids, wLoc, tLoc, reachable_distance_map):
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    # total disclosure
    total_notified_workers = 0
    total_travel_dist = 0.0

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            matched = False # if matched is True, go to the next task
            while len(eligibleWids) > 0 and not matched:
                # find the worker of the highest rank.
                highestRankWid = min(eligibleWids, key=lambda x:Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[x][0], wLoc[x][1]))
                actual_dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0], wLoc[highestRankWid][1])
                total_notified_workers += 1

                if actual_dist <= reachable_distance_map[highestRankWid]: # satisfiable match
                    matches.append((tid, highestRankWid))
                    total_travel_dist += actual_dist
                    matched = True
                else:
                    eligibleWids.remove(highestRankWid)

                if matched:
                    # affected tasks
                    affectedTids = workers[highestRankWid]

                    # delete highestRankWid from workers, tid from tasks
                    del workers[highestRankWid]
                    del tasks[tid]

                    for _tid in affectedTids:
                        if _tid != tid:
                            tasks[_tid].discard(highestRankWid)
                            if len(tasks[_tid]) == 0:
                                del tasks[_tid]

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist/matched_count if matched_count != 0 else 0
    return matched_count, false_hits, average_travel_dist, total_notified_workers

"""
Modify ranking algorithm that considers both reachable probability and resend strategy.
"""
def rankingByReachabilityResend(candidateWorkers, taskids, wLoc, tLoc, reachable_prob_U2E, reachable_distance_map):
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    # total disclosure
    total_notified_workers = 0
    total_travel_dist = 0.0
    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            matched = False # if matched is True, go to the next task
            while len(eligibleWids) > 0 and not matched:
                # find the worker of highest probability of reachability
                highestRankWid = max(eligibleWids, key=lambda wid:reachable_prob_U2E[
                    str(Utils.round_reachable_dist(reachable_distance_map[wid])) + ":" +
                    str(Utils.dist_range(Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[wid][0], wLoc[wid][1]), Params.step))
                ])
                dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0], wLoc[highestRankWid][1])
                total_notified_workers += 1

                if dist <= reachable_distance_map[highestRankWid]: # satisfiable match
                    matches.append((tid, highestRankWid))
                    total_travel_dist += dist
                    matched = True
                else:
                    eligibleWids.remove(highestRankWid)

                if matched:
                    # affected tasks
                    affectedTids = workers[highestRankWid]

                    # delete highestRankWid from workers, tid from tasks
                    del workers[highestRankWid]
                    del tasks[tid]

                    for _tid in affectedTids:
                        if _tid != tid:
                            tasks[_tid].discard(highestRankWid)
                            if len(tasks[_tid]) == 0:
                                del tasks[_tid]

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist/matched_count if matched_count != 0 else 0
    return matched_count, false_hits, average_travel_dist

"""
Modify ranking algorithm that considers reachable probability, resend strategy and worker's acceptance policy.
The policy is that a worker can reject a task without knowing the task location. This helps to decrease the amount
of extra disclosure.
"""
def rankingByReachabilityEmpirical(candidateWorkers, taskids, wLoc, tLoc, reachable_prob_U2E, reachable_distance_map, reachability_threshold):
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    # total disclosure
    total_notified_workers = 0
    total_travel_dist = 0.0

    # false dismissals/false hits during U2E
    # false_hits equals to total_notified_workers - number of matches
    false_dismissals = 0 # dismiss a reachable worker.

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            matched = False # if matched is True, go to the next task
            rejected = False # if rejected is True, go to the next task
            while len(eligibleWids) > 0 and not matched and not rejected:
                # find the worker of highest probability of reachability (U2E)
                highestRankWid = max(eligibleWids, key=lambda wid:reachable_prob_U2E[
                    str(Utils.round_reachable_dist(reachable_distance_map[wid])) + ":" +
                    str(Utils.dist_range(Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[wid][0], wLoc[wid][1]),
                                         Params.step))
                ])

                actual_dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0], wLoc[highestRankWid][1])
                highest_reachable_prob = reachable_prob_U2E[
                    str(Utils.round_reachable_dist(reachable_distance_map[highestRankWid])) + ":" +
                    str(Utils.dist_range(actual_dist, Params.step))] # U2E.
                if highest_reachable_prob < reachability_threshold:
                    rejected = True
                    if actual_dist <= reachable_distance_map[highestRankWid]:  # if this worker is reachable.
                        false_dismissals += 1
                    continue
                total_notified_workers += 1

                if actual_dist <= reachable_distance_map[highestRankWid]: # satisfiable match
                    matched = True
                    matches.append((tid, highestRankWid))
                    total_travel_dist += actual_dist
                else:
                    eligibleWids.remove(highestRankWid)

                if matched:
                    # affected tasks
                    affectedTids = workers[highestRankWid]

                    # delete highestRankWid from workers, tid from tasks
                    del workers[highestRankWid]
                    del tasks[tid]

                    for _tid in affectedTids:
                        if _tid != tid:
                            tasks[_tid].discard(highestRankWid)
                            if len(tasks[_tid]) == 0:
                                del tasks[_tid]

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist/matched_count if matched_count != 0 else 0
    return matched_count, false_hits, average_travel_dist, false_dismissals, total_notified_workers

"""
Similar to above function except using analytical results.
"""
def rankingByReachabilityAnalytical(candidateWorkers, taskids, wLoc, tLoc, reachable_distance_map, reachability_threshold, p):
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

    # total disclosure
    total_notified_workers = 0
    total_travel_dist = 0.0

    # false dismissals/false hits during U2E
    # false_hits equals to total_notified_workers - number of matches
    false_dismissals = 0 # dismiss a reachable worker.

    for tid in taskids:  # iterate through task list
        if tid in tasks: # check if tid has eligible nearby workers
            eligibleWids = list(tasks[tid])
            matched = False # if matched is True, go to the next task
            rejected = False # if rejected is True, go to the next task
            while len(eligibleWids) > 0 and not matched and not rejected:
                # find the worker of highest probability of reachability (U2E)
                highestRankWid = max(eligibleWids, key=lambda wid:Utils.reachable_prob_U2E(
                    reachable_distance_map[wid],
                    Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[wid][0], wLoc[wid][1]),
                    p.eps, p.radius))

                actual_dist = Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0], wLoc[highestRankWid][1])
                highest_reachable_prob = Utils.reachable_prob_U2E(
                    reachable_distance_map[highestRankWid],
                    Utils.distance(tLoc[tid][0], tLoc[tid][1], wLoc[highestRankWid][0], wLoc[highestRankWid][1]),
                    p.eps, p.radius) # U2E.
                if highest_reachable_prob < reachability_threshold:
                    rejected = True
                    if actual_dist <= reachable_distance_map[highestRankWid]:  # if this worker is reachable.
                        false_dismissals += 1
                    continue
                total_notified_workers += 1

                if actual_dist <= reachable_distance_map[highestRankWid]: # satisfiable match
                    matched = True
                    matches.append((tid, highestRankWid))
                    total_travel_dist += actual_dist
                else:
                    eligibleWids.remove(highestRankWid)

                if matched:
                    # affected tasks
                    affectedTids = workers[highestRankWid]

                    # delete highestRankWid from workers, tid from tasks
                    del workers[highestRankWid]
                    del tasks[tid]

                    for _tid in affectedTids:
                        if _tid != tid:
                            tasks[_tid].discard(highestRankWid)
                            if len(tasks[_tid]) == 0:
                                del tasks[_tid]

    matched_count = len(matches)
    false_hits = total_notified_workers - matched_count
    average_travel_dist = total_travel_dist/matched_count if matched_count != 0 else 0
    return matched_count, false_hits, average_travel_dist, false_dismissals, total_notified_workers

"""
Implementation of balance algorithm for online b-matching.
Citation: Kalyanasundaram and Kruhs. An optimal deterministic algorithm for online b-matching
"""
def balance(candidateWorkers, taskids, b):
    """
    :param candidateWorkers: map each workerid to a list of nearby tasks
    :param taskids: list of taskids arriving in order
    :param b: at most b tasks can be matched to one worker
    :return: a list matching pairs
    """
    matches = []
    tasks = copy.deepcopy(candidateWorkers)
    workers = Utils.workerDict2TaskDict(tasks) # create dict with key = workerid

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