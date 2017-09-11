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
import Simulation, Simulation2
import Utils

# seed_list = [9110]
seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# eps_list = [0.7]
eps_list = [0.1, 0.4, 0.7, 1.0]

# radius_list = [700.0]
radius_list = [100.0, 400.0, 700.0, 1000.0]

at_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] # acceptance threshold.
ct_list = [0.6,0.7,0.8,0.9] # coverge threshold.
#
def evalOnline(p, wList, tList, wLoc, tLoc, reach_dist_map):
    exp_name = ""
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
    res_cube_reachable_resend_precision = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_recall = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_f1score = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachable_resend_acceptance = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_acceptance_disclosure = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_acceptance_travel = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_acceptance_false_dismissals = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    reachableProbDict = Simulation.getProbability(radius_list, eps_list, "reachable") # U2U stage.
    reachableProbDict2 = Simulation2.getProbability(radius_list, eps_list, "reachable2") # U2E stage.
    # TODO: consider expanded stragegy.
    # coverageProbDict = Simulation.getProbability(radius_list, eps_list, "coverage")
    for expanded in [False]:
        for j in range(len(seed_list)):
            for i in range(len(eps_list)):
                for k in range(len(radius_list)):
                    p.seed = seed_list[j]
                    p.eps = eps_list[i]
                    p.radius = radius_list[k]
                    dp = Differential(p.seed)

                    # perturb locations of workers and tasks using Geo-I.
                    perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                    perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                    """
                    precision/recall during U2U
                    """

                    # Obtain workerMap, taskMap.
                    # wDict = dict([(id, (lat, lon)) for lat, lon, id in wList])
                    # tDict = dict([(id, (lat, lon)) for lat, lon, id in tList])

                    # for each task, find a set of reachable workers
                    precisions, recalls = [], []
                    for tNoisyLat, tNoisyLon, tid in perturbedTList: # [lat, lon, id]
                        selected_workers, reachable_workers, reachable_selected_workers = 0, 0, 0
                        for wNoisyLat, wNoisyLon, wid in perturbedWList:
                            noisy_dist = Utils.distance(wNoisyLat, wNoisyLon, tNoisyLat, tNoisyLon)
                            actual_dist = Utils.distance(wLoc[wid][0], wLoc[wid][1], tLoc[tid][0], tLoc[tid][1])
                            if noisy_dist <= reach_dist_map[wid]:
                                selected_workers += 1 # noisy domain
                            if actual_dist <= reach_dist_map[wid]:
                                reachable_workers += 1 # actual domain
                            if noisy_dist <= reach_dist_map[wid] and actual_dist <= reach_dist_map[wid]:
                                reachable_selected_workers += 1
                        if selected_workers > 0:
                            precisions.append(reachable_selected_workers/selected_workers)
                        if reachable_workers > 0:
                            recalls.append(reachable_selected_workers/reachable_workers)

                    mean_precision, mean_recall = np.mean(precisions), np.mean(recalls)
                    # print ("average precision/recall: ", mean_precision, mean_recall)
                    mean_f1_score = np.mean([2*(p * r)/(p + r) for p in precisions for r in recalls if p != 0 and r != 0])

                    # compute reachable distance such that a matched worker-task pair is
                    # reachable in the actual domain.
                    # coverageProb = coverageProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                    if expanded:
                        # reachableNoisyDist = Utils.reachableNoisyDist(coverageProb, p.coverageThreshold)
                        exp_name = sys._getframe().f_code.co_name + "_expanded"
                    else:
                        exp_name = sys._getframe().f_code.co_name

                    workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, reach_dist_map)

                    # ranking by random rank
                    matches = Geocrowd.ranking(workers, range(p.taskCount), wLoc, tLoc)

                    # ranking by reachable probability
                    reachableProb = reachableProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                    matches_reachable = Geocrowd.rankingByReachability(workers, range(p.taskCount), wLoc, tLoc,
                                                                       reachableProb, reach_dist_map)

                    # ranking with resend strategy (Baseline)
                    matches_resend, extra_disclosure, average_travel_dist = \
                        Geocrowd.rankingResend(workers, range(p.taskCount), wLoc, tLoc, reach_dist_map)

                    # ranking with both reachable probability and resend strategy (Empirical-Reachability)
                    matches_reachable_resend, extra_reachable_disclosure, average_reachable_travel_dist = \
                        Geocrowd.rankingByReachabilityResend(workers, range(p.taskCount), wLoc, tLoc, reachableProb,
                                                             reach_dist_map)

                    # Ranking with reachable probability, resend strategy and worker's acceptance policy (Analytical-Reachability-Threshold)
                    reachableProb2 = reachableProbDict2[Utils.RadiusEps2Str(p.radius, p.eps)]
                    matches_reachable_resend_acceptance, extra_reachable_acceptance_disclosure, average_reachable_acceptance_travel_dist, \
                    average_reachable_acceptance_false_dismissals = \
                        Geocrowd.rankingByReachabilityResendAcceptance(workers, range(p.taskCount), wLoc, tLoc,
                                                                       reachableProb, reachableProb2,
                                                                       reach_dist_map, p.acceptanceThreshold)
                    # print ("false_dismissals", average_reachable_acceptance_false_dismissals)

                    res_cube[i, j, k], res_cube_travel[i, j, k] = \
                        Geocrowd.satisfiableMatches(matches, wLoc, tLoc, reach_dist_map)
                    res_cube_reachable[i, j, k], res_cube_reachable_travel[i, j, k] = \
                        Geocrowd.satisfiableMatches(matches_reachable, wLoc, tLoc, reach_dist_map)
                    res_cube_resend[i, j, k], res_cube_resend_disclosure[i, j, k], res_cube_resend_travel[i, j, k] = \
                        matches_resend, extra_disclosure, average_travel_dist
                    res_cube_reachable_resend[i, j, k], res_cube_reachable_resend_disclosure[i, j, k], \
                    res_cube_reachable_resend_travel[i, j, k], res_cube_reachable_resend_f1score[i, j, k], \
                    res_cube_reachable_resend_precision[i, j, k], res_cube_reachable_resend_recall[i, j, k] = \
                        matches_reachable_resend, extra_reachable_disclosure, average_reachable_travel_dist, mean_f1_score, mean_precision, mean_recall
                    res_cube_reachable_resend_acceptance[i, j, k], res_cube_reachable_resend_acceptance_disclosure[i, j, k], \
                    res_cube_reachable_resend_acceptance_travel[i, j, k], res_cube_reachable_resend_acceptance_false_dismissals[i, j, k] = \
                        matches_reachable_resend_acceptance, extra_reachable_acceptance_disclosure, average_reachable_acceptance_travel_dist, \
                        average_reachable_acceptance_false_dismissals

        res_summary, res_summary_travel = np.average(res_cube, axis=1), np.average(res_cube_travel, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name, res_summary,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_travel", res_summary_travel,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_reachable, res_summary_reachable_travel = np.average(res_cube_reachable, axis=1), np.average(
            res_cube_reachable_travel, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reach", res_summary_reachable,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reach_travel", res_summary_reachable_travel,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_resend, res_summary_resend_disclosure, res_summary_resend_travel = \
            np.average(res_cube_resend, axis=1), np.average(res_cube_resend_disclosure, axis=1), \
            np.average(res_cube_resend_travel, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Baseline", res_summary_resend,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Baseline_disclosure", res_summary_resend_disclosure,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Baseline_travel", res_summary_resend_travel,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_reachable_resend, res_summary_reachable_resend_disclosure, res_summary_reachable_resend_travel, res_summary_reachable_resend_f1score, \
        res_summary_reachable_resend_precision, res_summary_reachable_resend_recall = \
            np.average(res_cube_reachable_resend, axis=1), np.average(res_cube_reachable_resend_disclosure, axis=1), \
            np.average(res_cube_reachable_resend_travel, axis=1), np.average(res_cube_reachable_resend_f1score, axis=1), \
            np.average(res_cube_reachable_resend_precision, axis=1), np.average(res_cube_reachable_resend_recall, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachresend", res_summary_reachable_resend,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachresend_disclosure",
                   res_summary_reachable_resend_disclosure,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachresend_travel",
                   res_summary_reachable_resend_travel,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_f1score",
                   res_summary_reachable_resend_f1score,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_precision",
                   res_summary_reachable_resend_precision,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_recall",
                   res_summary_reachable_resend_recall,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_reachable_resend_acceptance, res_summary_reachable_resend_acceptance_disclosure, res_summary_reachable_resend_acceptance_travel, \
        res_summary_reachable_resend_acceptance_false_dismissals = \
            np.average(res_cube_reachable_resend_acceptance, axis=1), np.average(res_cube_reachable_resend_acceptance_disclosure, axis=1), \
            np.average(res_cube_reachable_resend_acceptance_travel, axis=1), np.average(res_cube_reachable_resend_acceptance_false_dismissals, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_SampleMatch", res_summary_reachable_resend_acceptance,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_SampleMatch_disclosure",
                   res_summary_reachable_resend_acceptance_disclosure,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_SampleMatch_travel",
                   res_summary_reachable_resend_acceptance_travel,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_SampleMatch_false_dismissals",
                   res_summary_reachable_resend_acceptance_false_dismissals,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

def evalOnline_vary_at(p, wList, tList, wLoc, tLoc, reach_dist_map):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube_reachable_resend_acceptance = np.zeros((len(eps_list), len(seed_list), len(at_list)))
    res_cube_reachable_resend_acceptance_disclosure = np.zeros((len(eps_list), len(seed_list), len(at_list)))
    res_cube_reachable_resend_acceptance_travel = np.zeros((len(eps_list), len(seed_list), len(at_list)))
    res_cube_reachable_resend_acceptance_false_dismissals = np.zeros((len(eps_list), len(seed_list), len(at_list)))

    reachableProbDict = Simulation.getProbability(radius_list, eps_list, "reachable") # U2U stage.
    reachableProbDict2 = Simulation2.getProbability(radius_list, eps_list, "reachable2") # U2E stage.
    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            for k in range(len(at_list)):
                p.seed = seed_list[j]
                p.eps = eps_list[i]
                p.acceptanceThreshold = at_list[k]
                dp = Differential(p.seed)

                perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                # compute reachable distance such that a matched worker-task pair is
                # reachable in the actual domain.
                workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, reach_dist_map)

                reachableProb = reachableProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]

                # Ranking with reachable probability, resend strategy and worker's acceptance policy
                reachableProb2 = reachableProbDict2[Utils.RadiusEps2Str(p.radius, p.eps)]
                matches_reachable_resend_acceptance, extra_reachable_acceptance_disclosure, average_reachable_acceptance_travel_dist, \
                    average_reachable_acceptance_false_dismissals = \
                    Geocrowd.rankingByReachabilityResendAcceptance(workers, range(p.taskCount), wLoc, tLoc,
                                                                   reachableProb, reachableProb2,
                                                                   reach_dist_map, p.acceptanceThreshold)
                print("false_dismissals", average_reachable_acceptance_false_dismissals)

                res_cube_reachable_resend_acceptance[i, j, k], res_cube_reachable_resend_acceptance_disclosure[i, j, k], \
                res_cube_reachable_resend_acceptance_travel[i, j, k], res_cube_reachable_resend_acceptance_false_dismissals[i, j, k] = \
                    matches_reachable_resend_acceptance, extra_reachable_acceptance_disclosure, average_reachable_acceptance_travel_dist, \
                    average_reachable_acceptance_false_dismissals

    res_summary_reachable_resend_acceptance, res_summary_reachable_resend_acceptance_disclosure, res_summary_reachable_resend_acceptance_travel, \
    res_summary_reachable_resend_acceptance_false_dismissals = \
        np.average(res_cube_reachable_resend_acceptance, axis=1), np.average(res_cube_reachable_resend_acceptance_disclosure, axis=1), \
        np.average(res_cube_reachable_resend_acceptance_travel, axis=1), np.average(res_cube_reachable_resend_acceptance_false_dismissals, axis=1)
    np.savetxt(p.resdir + "/vary_at/" + Params.DATASET + "_" + exp_name + "_SampleMatch", res_summary_reachable_resend_acceptance,
               header="\t".join([str(r) for r in at_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + "/vary_at/" + Params.DATASET + "_" + exp_name + "_SampleMatch_disclosure",
               res_summary_reachable_resend_acceptance_disclosure,
               header="\t".join([str(r) for r in at_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + "/vary_at/" + Params.DATASET + "_" + exp_name + "_SampleMatch_travel",
               res_summary_reachable_resend_acceptance_travel,
               header="\t".join([str(r) for r in at_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + "/vary_at/" + Params.DATASET + "_" + exp_name + "_SampleMatch_false_dismissals",
               res_summary_reachable_resend_acceptance_false_dismissals,
               header="\t".join([str(r) for r in at_list]), fmt='%.4f\t')

# NOTE: expanded stragegy should be enabled.
def evalOnline_vary_ct(p, wList, tList, wLoc, tLoc, reach_dist_map):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    res_cube_reachable_resend_acceptance = np.zeros((len(eps_list), len(seed_list), len(ct_list)))
    res_cube_reachable_resend_acceptance_disclosure = np.zeros((len(eps_list), len(seed_list), len(ct_list)))
    res_cube_reachable_resend_acceptance_travel = np.zeros((len(eps_list), len(seed_list), len(ct_list)))

    reachableProbDict = Simulation.getProbability(radius_list, eps_list, "reachability") # U2U stage.
    reachableProbDict2 = Simulation2.getProbability(radius_list, eps_list, "reachability_2") # U2E stage.
    coverageProbDict = Simulation.getProbability(radius_list, eps_list, "precision_recall")
    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            for k in range(len(ct_list)):
                p.seed = seed_list[j]
                p.eps = eps_list[i]
                p.coverageThreshold = ct_list[k]
                dp = Differential(p.seed)

                perturbedWList = Geocrowd.perturbedData(wList, p, dp) # (lat, lon, id)
                perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                # compute reachable distance such that a matched worker-task pair is
                # reachable in the actual domain.
                # coverageProb = coverageProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                # reachableNoisyDist = Utils.reachableNoisyDist(coverageProb, p.coverageThreshold)
                workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, reach_dist_map)

                reachableProb = reachableProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]

                # Ranking with reachable probability, resend strategy and worker's acceptance policy
                reachableProb2 = reachableProbDict2[Utils.RadiusEps2Str(p.radius, p.eps)]
                matches_reachable_resend_acceptance, extra_reachable_acceptance_disclosure, average_reachable_acceptance_travel_dist, \
                average_reachable_acceptance_false_dismissals = \
                    Geocrowd.rankingByReachabilityResendAcceptance(workers, range(p.taskCount), wLoc, tLoc,
                                                                   reachableProb, reachableProb2,
                                                                   reach_dist_map, p.acceptanceThreshold)
                print("false_dismissals", average_reachable_acceptance_false_dismissals)

                res_cube_reachable_resend_acceptance[i, j, k], res_cube_reachable_resend_acceptance_disclosure[i, j, k], \
                res_cube_reachable_resend_acceptance_travel[i, j, k] = \
                    matches_reachable_resend_acceptance, extra_reachable_acceptance_disclosure, average_reachable_acceptance_travel_dist


    res_summary_reachable_resend_acceptance, res_summary_reachable_resend_acceptance_disclosure, res_summary_reachable_resend_acceptance_travel = \
        np.average(res_cube_reachable_resend_acceptance, axis=1), np.average(res_cube_reachable_resend_acceptance_disclosure, axis=1), \
        np.average(res_cube_reachable_resend_acceptance_travel, axis=1)
    np.savetxt(p.resdir + "/vary_ct/" + Params.DATASET + "_" + exp_name + "_reachable_resend_acceptance", res_summary_reachable_resend_acceptance,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + "/vary_ct/" + Params.DATASET + "_" + exp_name + "_reachable_resend_acceptance_disclosure",
               res_summary_reachable_resend_acceptance_disclosure,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + "/vary_ct/" + Params.DATASET + "_" + exp_name + "_reachable_resend_acceptance_travel",
               res_summary_reachable_resend_acceptance_travel,
               header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

def evalOffline(p, wList, tList, wLoc, tLoc, reach_dist_map):
    exp_name = ""
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_travel = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    # coverageProbDict = Simulation.getProbability(radius_list, eps_list, "coverage")
    for expanded in [False]:
        for j in range(len(seed_list)):
            for i in range(len(eps_list)):
                for k in range(len(radius_list)):
                    p.seed = seed_list[j]
                    p.eps = eps_list[i]
                    p.radius = radius_list[k]
                    dp = Differential(p.seed)

                    perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                    perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                    # associate a reachable distance to each worker.
                    reach_dist_map = {} # <wid, reachable_distance>
                    for wid in wLoc.keys():
                        reach_dist_map[wid] = Utils.randomReachableDist(p.reachableDistRange, p.seed)

                    # compute reachable distance such that a matched worker-task pair is
                    # reachable in the actual domain.
                    # coverageProb = coverageProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                    if expanded:
                        # reachableNoisyDist = Utils.reachableNoisyDist(coverageProb, p.coverageThreshold)
                        exp_name = sys._getframe().f_code.co_name + "_expanded"
                    else:
                        exp_name = sys._getframe().f_code.co_name

                    workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, reach_dist_map)

                    # max flow
                    _, flowDict = Geocrowd.maxFlow(workers, range(p.taskCount))
                    matches = Geocrowd.flowDict2Matches(flowDict)


                    res_cube[i, j, k], res_cube_travel[i, j, k] = \
                        Geocrowd.satisfiableMatches(matches, wLoc, tLoc, reach_dist_map)

            res_summary, res_summary_travel = np.average(res_cube, axis=1), np.average(res_cube_travel, axis=1)
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name, res_summary,
                       header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_travel", res_summary_travel,
                       header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')


p = Params(1000)
p.select_dataset()

"""
Online
without privacy
"""
# without privacy
wList, tList, wLoc, tLoc = Geocrowd.sampleWorkersTasks(p.workerFile, p.taskFile, p.workerCount, p.taskCount)
reach_dist_map = dict([(wid, Utils.randomReachableDist(p.reachableDistRange, p.seed)) for wid in wLoc.keys()]) # <wid, reachable_distance>
workers = Geocrowd.createBipartiteGraph(wList, tList, reach_dist_map)

onlineMatches = Geocrowd.ranking(workers, range(p.taskCount), wLoc, tLoc)
onlineCount, onlineTravelCost = Geocrowd.satisfiableMatches(onlineMatches, wLoc, tLoc, reach_dist_map)
print("Non-private online (utility/travel cost): ", onlineCount, onlineTravelCost)

# with privacy
evalOnline(p, wList, tList, wLoc, tLoc, reach_dist_map)
# evalOnline_vary_at(p, wList, tList, wLoc, tLoc, reach_dist_map)
# evalOnline_vary_ct(p, wList, tList, wLoc, tLoc, reach_dist_map)

# onlineMatches = Geocrowd.balanceAlgo(workers, range(p.taskCount), 2)
# offlineMatches = Geocrowd.maxFlowValue(workers, range(p.taskCount), 2)
# print (onlineMatches, offlineMatches)


"""
Offline
"""
# Without privacy
offlineMatches = Geocrowd.maxFlowValue(workers, range(p.taskCount))
print("Non-private offline: ", offlineMatches)

# with privacy
# TODO: fix offline.
# evalOffline(p, wList, tList, wLoc, tLoc, reach_dist_map)
