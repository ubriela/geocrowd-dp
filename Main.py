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

# seed_list = [9110]
seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

eps_list = [0.5]
# eps_list = [0.1, 0.4, 0.7, 1.0]

radius_list = [500.0]
# radius_list = [100.0, 400.0, 700.0, 1000.0]

"""
Publish location entropy using Smoooth sensitivity
"""
def evalOnline(p):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    wList, tList, wLoc, tLoc = Geocrowd.sampleWorkersTasks(p.workerFile, p.taskFile, p.workerCount, p.taskCount)

    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            for k in range(len(radius_list)):
                p.seed = seed_list[j]
                p.eps = eps_list[i]
                p.radius = radius_list[k]
                dp = Differential(p.seed)

                perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                # radiusOfRetrieval = dp.radiusOfRetrieval(p.reachableDist, p.eps, p.radius, p.confidence)
                # print (radiusOfRetrieval)
                # print (radiusOfRetrieval/p.reachableDist)
                workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, 2*p.reachableDist)

                matches = Geocrowd.balanceAlgo(workers, range(p.taskCount), 2)
                res_cube[i, j, k] = Geocrowd.satisfiableMatches(matches, wLoc, tLoc, p.reachableDist)

    res_summary = np.average(res_cube, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name, res_summary, header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')


p = Params(1000)
p.select_dataset()

# without privacy
wList, tList, wLoc, tLoc = Geocrowd.sampleWorkersTasks(p.workerFile, p.taskFile, p.workerCount, p.taskCount)
workers = Geocrowd.createBipartiteGraph(wList, tList, p.reachableDist)
onlineMatches = Geocrowd.rankingAlgo(workers, range(p.taskCount))
offlineMatches = Geocrowd.maxFlowValue(workers, range(p.taskCount))
print (len(onlineMatches), offlineMatches)

# with privacy
evalOnline(p)


# onlineMatches = Geocrowd.balanceAlgo(workers, range(p.taskCount), 2)
# offlineMatches = Geocrowd.maxFlowValue(workers, range(p.taskCount), 2)
# print (onlineMatches, offlineMatches)

