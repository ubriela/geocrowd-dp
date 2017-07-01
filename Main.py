__author__ = 'ubriela'
import math
import logging
import time
import sys
import random
import numpy as np
from Differential import Differential
from Params import Params
import Geocrowd
from collections import defaultdict, Counter
import sys
import Simulation
import Utils

seed_list = [9110]
# seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

eps_list = [0.5]
# eps_list = [0.1, 0.4, 0.7, 1.0]

radius_list = [500.0]
# radius_list = [100.0, 400.0, 700.0, 1000.0]

"""
Compute #assigned tasks
"""
def evalOnline(p):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_travel = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachable = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_travel = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_resend = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_disclosure = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_travel = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachable_resend = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_disclosure = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_travel = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    wList, tList, wLoc, tLoc = Geocrowd.sampleWorkersTasks(p.workerFile, p.taskFile, p.workerCount, p.taskCount)

    reachableProbDict= Simulation.getProbability(radius_list, eps_list, "reachable")
    coverageProbDict = Simulation.getProbability(radius_list, eps_list, "coverage")
    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            for k in range(len(radius_list)):
                p.seed = seed_list[j]
                p.eps = eps_list[i]
                p.radius = radius_list[k]
                dp = Differential(p.seed)

                perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, p.reachableDist)

                # ranking by random rank
                matches = Geocrowd.ranking(workers, range(p.taskCount))

                # ranking by reachable probability
                reachableProb = reachableProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                matches_reachable = Geocrowd.rankingByReachableProb(workers, range(p.taskCount), wLoc, tLoc, reachableProb)

                # ranking with resend strategy
                matches_resend, extra_disclosure, average_travel_dist = \
                    Geocrowd.rankingResend(workers, range(p.taskCount), wLoc, tLoc, p.reachableDist)

                # ranking with both reachability probability and resend strategy
                matches_reachable_resend, extra_reachable_disclosure, average_reachable_travel_dist = \
                    Geocrowd.rankingByReachableProbResend(workers, range(p.taskCount), wLoc, tLoc, reachableProb, p.reachableDist)

                res_cube[i, j, k], res_cube_travel[i, j, k] = \
                    Geocrowd.satisfiableMatches(matches, wLoc, tLoc, p.reachableDist)
                res_cube_reachable[i, j, k], res_cube_reachable_travel[i, j, k] = \
                    Geocrowd.satisfiableMatches(matches_reachable, wLoc, tLoc, p.reachableDist)
                res_cube_resend[i, j, k], res_cube_resend_disclosure[i, j, k], res_cube_resend_travel[i, j, k] = \
                    matches_resend, extra_disclosure, average_travel_dist
                res_cube_reachable_resend[i, j, k], res_cube_reachable_resend_disclosure[i, j, k], res_cube_reachable_resend_travel[i, j, k] = \
                    matches_reachable_resend, extra_reachable_disclosure, average_reachable_travel_dist

    res_summary, res_summary_travel = np.average(res_cube, axis=1), np.average(res_cube_travel, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name, res_summary,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_travel", res_summary_travel,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

    res_summary_reachable, res_summary_reachable_travel = np.average(res_cube_reachable, axis=1), np.average(res_cube_reachable_travel, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachable", res_summary_reachable,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachable_travel", res_summary_reachable_travel,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

    res_summary_resend, res_summary_resend_disclosure, res_summary_resend_travel = \
        np.average(res_cube_resend, axis=1), np.average(res_cube_resend_disclosure, axis=1), \
        np.average(res_cube_resend_travel, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_resend", res_summary_resend,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_resend_disclosure", res_summary_resend_disclosure,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_resend_travel", res_summary_resend_travel,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

    res_summary_reachable_resend, res_summary_reachable_resend_disclosure, res_summary_reachable_resend_travel = \
        np.average(res_cube_reachable_resend, axis=1), np.average(res_cube_reachable_resend_disclosure, axis=1), \
        np.average(res_cube_reachable_resend_travel, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachable_resend", res_summary_reachable_resend,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachable_resend_disclosure", res_summary_reachable_resend_disclosure,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachable_resend_travel", res_summary_reachable_resend_travel,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

p = Params(1000)
p.select_dataset()

# without privacy
wList, tList, wLoc, tLoc = Geocrowd.sampleWorkersTasks(p.workerFile, p.taskFile, p.workerCount, p.taskCount)
workers = Geocrowd.createBipartiteGraph(wList, tList, p.reachableDist)
onlineMatches = Geocrowd.ranking(workers, range(p.taskCount))
offlineMatches = Geocrowd.maxFlowValue(workers, range(p.taskCount))
print ("Non-privacy online/offline: ", len(onlineMatches), offlineMatches)

# with privacy
evalOnline(p)


# onlineMatches = Geocrowd.balanceAlgo(workers, range(p.taskCount), 2)
# offlineMatches = Geocrowd.maxFlowValue(workers, range(p.taskCount), 2)
# print (onlineMatches, offlineMatches)


"""
Offline
"""
# _, flowDict = Geocrowd.maxFlow(workers, range(p.taskCount))
# matches = Geocrowd.flowDict2Matches(flowDict)

# matches = Geocrowd.balance(workers, range(p.taskCount), 10)

