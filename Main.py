__author__ = 'ubriela'
import math
import logging
from time import time
import sys
import random
import numpy as np
from Differential import Differential
from Params import Params
import Geocrowd
from collections import defaultdict, Counter
import sys
import U2U_Simulation, U2E_Simulation
import Utils
from multiprocessing import Pool

# seed_list = [9110]
seed_list = [9110, 4064, 6903]
# seed_list = [9110, 4064, 6903, 7509, 5342, 3230, 3584, 7019, 3564, 6456]

# eps_list = [0.7]
eps_list = [0.1, 0.4, 0.7, 1.0]

# radius_list = [1000.0]
radius_list = [200.0, 800.0, 1400.0, 2000.0]

U2U_threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
U2E_threshold_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
#
def eval(params):
    exp_name = ""
    logging.info(exp_name)

    p = params[0]
    wList = params[1]
    tList = params[2]
    wLoc = params[3]
    tLoc = params[4]
    reachable_distance_map = params[5]
    eps = params[6]

    eps_list = [eps]

    res_cube_rank_nearest_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_rank_nearest_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_rank_nearest_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_rank_nearest_NNW = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_rank_random_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_rank_random_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_rank_random_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_rank_random_NNW = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachable_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_resend_random_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_random_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_random_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_random_NNW = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_resend_nearest_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_nearest_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_nearest_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_resend_nearest_NNW = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachable_resend_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachable_resend_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_precision = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_recall = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_CAN = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachability_empirical_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_empirical_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_empirical_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_empirical_false_dismissals = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_empirical_NNW = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    res_cube_reachability_analytical_NAT = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_analytical_false_hits = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_analytical_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_analytical_false_dismissals = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_reachability_analytical_NNW = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

    reachabilityMapU2U = U2U_Simulation.getProbability(radius_list, eps_list, "reachability") # U2U stage.
    reachabilityMapU2E = U2E_Simulation.getProbability(radius_list, eps_list, "reachability_2") # U2E stage.
    for U2U_optimization in [True, False]: #
        for j in range(len(seed_list)):
            for i in range(len(eps_list)):
                for k in range(len(radius_list)):
                    p.seed = seed_list[j]
                    p.eps = eps_list[i]
                    p.radius = radius_list[k]
                    print ("seed/eps/radius", p.seed, p.eps, p.radius)
                    dp = Differential(p.seed)

                    # perturb locations of candidate_workers and tasks using Geo-I.
                    perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                    perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                    # compute reachable distance such that a matched worker-task pair is
                    # reachable in the actual domain.
                    # coverageProb = coverageProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                    reachable_prob_U2U = reachabilityMapU2U[Utils.radius_eps_2_str(p.radius, p.eps)]
                    reachable_prob_U2E = reachabilityMapU2E[Utils.radius_eps_2_str(p.radius, p.eps)]

                    # for each task, find a set of reachable candidate_workers
                    candidate_workers = None # tid --> [wid]
                    if U2U_optimization:
                        exp_name = sys._getframe().f_code.co_name + "_U2U_Opt"
                        candidate_workers = Geocrowd.createBipartiteGraphU2U(perturbedWList, perturbedTList,
                                                                          reachable_distance_map, reachable_prob_U2U, p)
                    else:
                        exp_name = sys._getframe().f_code.co_name
                        candidate_workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList,
                                                                      reachable_distance_map)
                    """
                    precision/recall during U2U
                    """
                    # wNoisyLoc = dict({wid, (lat, lon)} for lat, lon, wid in perturbedWList) # [lat, lon, id]
                    # tNoisyLoc = dict({tid, (lat, lon)} for lat, lon, tid in perturbedTList)

                    precisions, recalls = [], []
                    for tid, wids in candidate_workers.items():
                        selected_workers = len(wids)
                        reachable_workers, reachable_workers_other = 0, 0
                        for wid in wids:
                            # noisy_dist = Utils.distance(wNoisyLoc[wid][0], wNoisyLoc[wid][1], tNoisyLoc[tid][0], tNoisyLoc[tid][1])
                            actual_dist = Utils.distance(wLoc[wid][0], wLoc[wid][1], tLoc[tid][0], tLoc[tid][1])
                            if actual_dist <= reachable_distance_map[wid]:
                                reachable_workers += 1  # actual domain
                            # if noisy_dist <= reachable_distance_map[wid] and actual_dist <= reachable_distance_map[wid]:
                            #     reachable_selected_workers += 1
                        # number of reachable workers in total
                        for wid, loc in wLoc.items():
                            if wid not in wids:
                                actual_dist_other = Utils.distance(loc[0], loc[1], tLoc[tid][0], tLoc[tid][1])
                                if actual_dist_other <= reachable_distance_map[wid]:
                                    reachable_workers_other += 1  # actual domain
                        if selected_workers > 0:
                            precisions.append(float(reachable_workers) / selected_workers)
                        if reachable_workers > 0 or reachable_workers_other > 0:
                            recalls.append(float(reachable_workers) / (reachable_workers + reachable_workers_other))
                    mean_precision, mean_recall, candidate_worker_size = np.mean(precisions), np.mean(recalls), len(candidate_workers)
                    # print ("average precision/recall: ", mean_precision, mean_recall)

                    # ranking by nearest neighbor
                    ranking_nearest_NAT, ranking_nearest_false_hits, ranking_nearest_WTD, ranking_nearest_NNW = \
                        Geocrowd.rankingNearest(candidate_workers, range(p.taskCount), wLoc, tLoc, reachable_distance_map)

                    # ranking by nearest neighbor
                    ranking_random_NAT, ranking_random_false_hits, ranking_random_WTD, ranking_random_NNW = \
                        Geocrowd.rankingRandom(candidate_workers, range(p.taskCount), wLoc, tLoc, reachable_distance_map)

                    # ranking by reachable probability
                    matches_reachable = Geocrowd.rankingByReachability(candidate_workers, range(p.taskCount), wLoc, tLoc,
                                                                       reachable_prob_U2E, reachable_distance_map)

                    # ranking with resend strategy (Baseline-Random)
                    resend_random_NAT, resend_random_false_hits, resend_random_WTD, resend_random_NNW = \
                        Geocrowd.rankingResendRandom(candidate_workers, range(p.taskCount), wLoc, tLoc, reachable_distance_map)

                    # ranking with resend strategy (Baseline-Nearest)
                    resend_nearest_NAT, resend_nearest_false_hits, resend_nearest_WTD, resend_nearest_NNW = \
                        Geocrowd.rankingResendNearest(candidate_workers, range(p.taskCount), wLoc, tLoc, reachable_distance_map)

                    # ranking with both reachable probability and resend strategy
                    reachable_resend_NAT, reachable_resend_false_hits, reachable_resend_WTD = \
                        Geocrowd.rankingByReachabilityResend(candidate_workers, range(p.taskCount), wLoc, tLoc, reachable_prob_U2E, reachable_distance_map)

                    # Ranking with reachable probability, resend strategy and worker's acceptance policy (Reachability-Empirical)
                    reachability_empirical_NAT, reachability_empirical_false_hits, reachability_empirical_WTD, \
                    reachability_empirical_false_dismissals, reachability_empirical_NNW = \
                        Geocrowd.rankingByReachabilityEmpirical(candidate_workers, range(p.taskCount), wLoc, tLoc,
                                                                reachable_prob_U2E,
                                                                reachable_distance_map, p.reachabilityThresholdU2E)

                    reachability_analytical_NAT, reachability_analytical_false_hits, reachability_analytical_WTD, \
                    reachability_analytical_false_dismissals, reachability_analytical_NNW = \
                        Geocrowd.rankingByReachabilityAnalytical(candidate_workers, range(p.taskCount), wLoc, tLoc,
                                                                reachable_distance_map, p.reachabilityThresholdU2E, p)

                    res_cube_rank_nearest_NAT[i, j, k], res_cube_rank_nearest_false_hits[i, j, k], res_cube_rank_nearest_WTD[i, j, k], \
                    res_cube_rank_nearest_NNW[i, j, k] = \
                        ranking_nearest_NAT, ranking_nearest_false_hits, ranking_nearest_WTD, ranking_nearest_NNW
                    res_cube_rank_random_NAT[i, j, k], res_cube_rank_random_false_hits[i, j, k], res_cube_rank_random_WTD[i, j, k], \
                    res_cube_rank_random_NNW[i, j, k] = \
                        ranking_random_NAT, ranking_random_false_hits, ranking_random_WTD, ranking_random_NNW
                    res_cube_reachable_NAT[i, j, k], res_cube_reachable_WTD[i, j, k] = \
                        Geocrowd.satisfiableMatches(matches_reachable, wLoc, tLoc, reachable_distance_map)
                    res_cube_resend_random_NAT[i, j, k], res_cube_resend_random_false_hits[i, j, k], res_cube_resend_random_WTD[i, j, k], \
                    res_cube_resend_random_NNW[i, j, k]= \
                        resend_random_NAT, resend_random_false_hits, resend_random_WTD, resend_random_NNW
                    res_cube_resend_nearest_NAT[i, j, k], res_cube_resend_nearest_false_hits[i, j, k], res_cube_resend_nearest_WTD[i, j, k], \
                    res_cube_resend_nearest_NNW[i, j, k] = \
                        resend_nearest_NAT, resend_nearest_false_hits, resend_nearest_WTD, resend_nearest_NNW
                    res_cube_reachable_resend_NAT[i, j, k], res_cube_reachable_resend_false_hits[i, j, k], \
                    res_cube_reachable_resend_WTD[i, j, k] = \
                        reachable_resend_NAT, reachable_resend_false_hits, reachable_resend_WTD
                    res_cube_reachability_empirical_NAT[i, j, k], res_cube_reachability_empirical_false_hits[i, j, k], \
                    res_cube_reachability_empirical_WTD[i, j, k], res_cube_reachability_empirical_false_dismissals[i, j, k], \
                    res_cube_reachability_empirical_NNW[i, j, k] = \
                        reachability_empirical_NAT, reachability_empirical_false_hits, reachability_empirical_WTD, \
                        reachability_empirical_false_dismissals, reachability_empirical_NNW
                    res_cube_reachability_analytical_NAT[i, j, k], res_cube_reachability_analytical_false_hits[i, j, k], \
                    res_cube_reachability_analytical_WTD[i, j, k], res_cube_reachability_analytical_false_dismissals[i, j, k], \
                    res_cube_reachability_analytical_NNW[i, j, k] = \
                        reachability_analytical_NAT, reachability_analytical_false_hits, reachability_analytical_WTD, \
                        reachability_analytical_false_dismissals, reachability_analytical_NNW
                    res_cube_precision[i, j, k], res_cube_recall[i, j, k], res_cube_CAN[i, j, k] = mean_precision, mean_recall, candidate_worker_size

        res_summary_rank_nearest_NAT, res_summary_rank_nearest_false_hits, res_summary_rank_nearest_WTD, res_summary_rank_nearest_NNW = \
            np.average(res_cube_rank_nearest_NAT, axis=1), np.average(res_cube_rank_nearest_false_hits, axis=1), \
            np.average(res_cube_rank_nearest_WTD, axis=1), np.average(res_cube_rank_nearest_NNW, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_nearest_NAT" + '_' + str(eps), res_summary_rank_nearest_NAT,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name  + "_rank_nearest_false_hits" + '_' + str(eps), res_summary_rank_nearest_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_nearest_WTD" + '_' + str(eps), res_summary_rank_nearest_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_nearest_NNW" + '_' + str(eps), res_summary_rank_nearest_NNW,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_rank_random_NAT, res_summary_rank_random_false_hits, res_summary_rank_random_WTD, res_summary_rank_random_NNW = \
            np.average(res_cube_rank_random_NAT, axis=1), np.average(res_cube_rank_random_false_hits, axis=1), \
            np.average(res_cube_rank_random_WTD, axis=1), np.average(res_cube_rank_random_NNW, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_random_NAT" + '_' + str(eps), res_summary_rank_random_NAT,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_random_false_hits" + '_' + str(eps), res_summary_rank_random_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_random_WTD" + '_' + str(eps), res_summary_rank_random_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_rank_random_NNW" + '_' + str(eps), res_summary_rank_random_NNW,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_reachable, res_summary_reachable_WTD = np.average(res_cube_reachable_NAT, axis=1), np.average(
            res_cube_reachable_WTD, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reach_NAT" + '_' + str(eps), res_summary_reachable,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reach_WTD" + '_' + str(eps), res_summary_reachable_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_resend_random, res_summary_resend_random_false_hits, res_summary_resend_random_WTD, \
        res_summary_resend_random_NNW = \
            np.average(res_cube_resend_random_NAT, axis=1), np.average(res_cube_resend_random_false_hits, axis=1), \
            np.average(res_cube_resend_random_WTD, axis=1), np.average(res_cube_resend_random_NNW, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineRandom_NAT" + '_' + str(eps), res_summary_resend_random,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineRandom_false_hits" + '_' + str(eps), res_summary_resend_random_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineRandom_WTD" + '_' + str(eps), res_summary_resend_random_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineRandom_NNW" + '_' + str(eps), res_summary_resend_random_NNW,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_resend_nearest, res_summary_resend_nearest_false_hits, res_summary_resend_nearest_WTD, \
        res_summary_resend_nearest_NNW = \
            np.average(res_cube_resend_nearest_NAT, axis=1), np.average(res_cube_resend_nearest_false_hits, axis=1), \
            np.average(res_cube_resend_nearest_WTD, axis=1), np.average(res_cube_resend_nearest_NNW, axis=1),
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineNearest_NAT" + '_' + str(eps), res_summary_resend_nearest,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineNearest_false_hits" + '_' + str(eps), res_summary_resend_nearest_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineNearest_WTD" + '_' + str(eps), res_summary_resend_nearest_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_BaselineNearest_NNW" + '_' + str(eps), res_summary_resend_nearest_NNW,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_reachable_resend, res_summary_reachable_resend_false_hits, res_summary_reachable_resend_WTD = \
            np.average(res_cube_reachable_resend_NAT, axis=1), np.average(res_cube_reachable_resend_false_hits, axis=1), \
            np.average(res_cube_reachable_resend_WTD, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachresend_NAT" + '_' + str(eps), res_summary_reachable_resend,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachresend_false_hits" + '_' + str(eps),
                   res_summary_reachable_resend_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_reachresend_WTD" + '_' + str(eps),
                   res_summary_reachable_resend_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_reachability_empirical_NAT, res_summary_reachability_empirical_false_hits, res_summary_reachability_empirical_WTD, \
        res_summary_reachability_empirical_false_dismissals, res_summary_reachability_empirical_NNW = \
            np.average(res_cube_reachability_empirical_NAT, axis=1), np.average(res_cube_reachability_empirical_false_hits, axis=1), \
            np.average(res_cube_reachability_empirical_WTD, axis=1), np.average(res_cube_reachability_empirical_false_dismissals, axis=1), \
            np.average(res_cube_reachability_empirical_NNW, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Empirical_NAT" + '_' + str(eps), res_summary_reachability_empirical_NAT,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Empirical_false_hits" + '_' + str(eps),
                   res_summary_reachability_empirical_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Empirical_WTD" + '_' + str(eps),
                   res_summary_reachability_empirical_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Empirical_false_dismissals" + '_' + str(eps),
                   res_summary_reachability_empirical_false_dismissals,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Empirical_NNW" + '_' + str(eps),
                   res_summary_reachability_empirical_NNW,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')


        res_summary_reachability_analytical_NAT, res_summary_reachability_analytical_false_hits, res_summary_reachability_analytical_WTD, \
        res_summary_reachability_analytical_false_dismissals, res_summary_reachability_analytical_NNW = \
            np.average(res_cube_reachability_empirical_NAT, axis=1), np.average(res_cube_reachability_analytical_false_hits,
                axis=1), np.average(res_cube_reachability_analytical_WTD, axis=1), np.average(res_cube_reachability_analytical_false_dismissals, axis=1), \
            np.average(res_cube_reachability_analytical_NNW, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_NAT" + '_' + str(eps), res_summary_reachability_analytical_NAT,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_false_hits" + '_' + str(eps),
                   res_summary_reachability_analytical_false_hits,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_WTD" + '_' + str(eps),
                   res_summary_reachability_analytical_WTD,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_false_dismissals" + '_' + str(eps),
                   res_summary_reachability_analytical_false_dismissals,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_NNW" + '_' + str(eps),
                   res_summary_reachability_analytical_NNW,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')

        res_summary_precision, res_summary_recall, res_summary_CAN = \
            np.average(res_cube_precision, axis=1), np.average(res_cube_recall, axis=1), np.average(res_cube_CAN, axis=1)
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_precision" + '_' + str(eps),
                   res_summary_precision,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_recall" + '_' + str(eps),
                   res_summary_recall,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
        np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_CAN" + '_' + str(eps),
                   res_summary_CAN,
                   header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')


def eval_U2E_threshold(params):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    p = params[0]
    wList = params[1]
    tList = params[2]
    wLoc = params[3]
    tLoc = params[4]
    reachable_distance_map = params[5]
    eps = params[6]

    eps_list = [eps]

    res_cube_reachability_NAT = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))
    res_cube_reachability_false_hits = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))
    res_cube_reachability_WTD = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))
    res_cube_reachability_false_dismissals = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))
    res_cube_reachability_NNW = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))
    res_cube_reachability_precision = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))
    res_cube_reachability_recall = np.zeros((len(eps_list), len(seed_list), len(U2E_threshold_list)))

    reachabilityMapU2U = U2U_Simulation.getProbability(radius_list, eps_list, "reachability") # U2U stage.
    # reachabilityMapU2E = Simulation2.getProbability(radius_list, eps_list, "reachability_2") # U2E stage.
    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            for k in range(len(U2E_threshold_list)):
                p.seed = seed_list[j]
                p.eps = eps_list[i]
                print ("seed/eps/radius", p.seed, p.eps, p.radius)
                p.reachabilityThresholdU2E = U2E_threshold_list[k]
                dp = Differential(p.seed)

                perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                reachable_prob_U2U = reachabilityMapU2U[Utils.radius_eps_2_str(p.radius, p.eps)]

                candidate_workers = Geocrowd.createBipartiteGraphU2U(perturbedWList, perturbedTList,
                                                                     reachable_distance_map, reachable_prob_U2U, p)

                """
                precision/recall during U2U
                """
                precisions, recalls = [], []
                for tid, wids in candidate_workers.items():
                    selected_workers = len(wids)
                    reachable_workers, reachable_workers_other = 0, 0
                    for wid in wids:
                        # noisy_dist = Utils.distance(wNoisyLoc[wid][0], wNoisyLoc[wid][1], tNoisyLoc[tid][0], tNoisyLoc[tid][1])
                        actual_dist = Utils.distance(wLoc[wid][0], wLoc[wid][1], tLoc[tid][0], tLoc[tid][1])
                        if actual_dist <= reachable_distance_map[wid]:
                            reachable_workers += 1  # actual domain
                            # if noisy_dist <= reachable_distance_map[wid] and actual_dist <= reachable_distance_map[wid]:
                            #     reachable_selected_workers += 1
                    # number of reachable workers in total
                    for wid, loc in wLoc.items():
                        if wid not in wids:
                            actual_dist_other = Utils.distance(loc[0], loc[1], tLoc[tid][0], tLoc[tid][1])
                            if actual_dist_other <= reachable_distance_map[wid]:
                                reachable_workers_other += 1  # actual domain
                    if selected_workers > 0:
                        precisions.append(float(reachable_workers) / selected_workers)
                    if reachable_workers > 0 or reachable_workers_other > 0:
                        recalls.append(float(reachable_workers) / (reachable_workers + reachable_workers_other))
                reachability_precision, reachability_recall = np.mean(precisions), np.mean(recalls)

                # Ranking with reachable probability, resend strategy and worker's acceptance policy
                reachability_NAT, reachability_false_hits, reachability_WTD, reachability_false_dismissals, reachability_NNW = \
                    Geocrowd.rankingByReachabilityAnalytical(candidate_workers, range(p.taskCount), wLoc, tLoc,
                                                            reachable_distance_map, p.reachabilityThresholdU2E, p)

                res_cube_reachability_NAT[i, j, k], res_cube_reachability_false_hits[i, j, k], \
                res_cube_reachability_WTD[i, j, k], res_cube_reachability_false_dismissals[i, j, k], \
                res_cube_reachability_NNW[i, j, k], res_cube_reachability_precision[i, j, k], \
                res_cube_reachability_recall[i, j, k] = \
                    reachability_NAT, reachability_false_hits, reachability_WTD, reachability_false_dismissals, \
                    reachability_NNW, reachability_precision, reachability_recall

    res_summary_reachability_NAT, res_summary_reachability_false_hits, res_summary_reachability_WTD, \
    res_summary_reachability_false_dismissals, res_summary_reachability_NNW, \
    res_summary_reachability_precision, res_summary_reachability_recall = \
        np.average(res_cube_reachability_NAT, axis=1), np.average(res_cube_reachability_false_hits, axis=1), \
        np.average(res_cube_reachability_WTD, axis=1), np.average(res_cube_reachability_false_dismissals, axis=1), \
        np.average(res_cube_reachability_NNW, axis=1), np.average(res_cube_reachability_precision, axis=1), \
        np.average(res_cube_reachability_recall, axis=1)
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_NAT" + '_' + str(eps), res_summary_reachability_NAT,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_false_hits" + '_' + str(eps),
               res_summary_reachability_false_hits,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_WTD" + '_' + str(eps),
               res_summary_reachability_WTD,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_false_dismissals" + '_' + str(eps),
               res_summary_reachability_false_dismissals,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_NNW" + '_' + str(eps),
               res_summary_reachability_NNW,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_precision" + '_' + str(eps),
               res_summary_reachability_precision,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
    np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_recall" + '_' + str(eps),
               res_summary_reachability_recall,
               header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')

def eval_U2U_threshold(params):
    exp_name = sys._getframe().f_code.co_name
    logging.info(exp_name)

    p = params[0]
    wList = params[1]
    tList = params[2]
    wLoc = params[3]
    tLoc = params[4]
    reachable_distance_map = params[5]
    eps = params[6]

    eps_list = [eps]

    res_cube_reachability_NAT = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_false_hits = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_WTD = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_false_dismissals = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_NNW = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_precision = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_recall = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_u2u_runtime = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_u2e_runtime = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))
    res_cube_reachability_CAN = np.zeros((len(eps_list), len(seed_list), len(U2U_threshold_list)))

    reachabilityMapU2U = U2U_Simulation.getProbability(radius_list, eps_list, "reachability") # U2U stage.
    # reachabilityMapU2E = Simulation2.getProbability(radius_list, eps_list, "reachability_2") # U2E stage.
    for j in range(len(seed_list)):
        for i in range(len(eps_list)):
            for k in range(len(U2U_threshold_list)):
                p.seed = seed_list[j]
                p.eps = eps_list[i]
                print ("seed/eps/radius", p.seed, p.eps, p.radius)
                p.reachabilityThresholdU2U = U2U_threshold_list[k]
                dp = Differential(p.seed)

                perturbedWList = Geocrowd.perturbedData(wList, p, dp)
                perturbedTList = Geocrowd.perturbedData(tList, p, dp)

                reachable_prob_U2U = reachabilityMapU2U[Utils.radius_eps_2_str(p.radius, p.eps)]
                # reachable_prob_U2E = reachabilityMapU2E[Utils.RadiusEps2Str(p.radius, p.eps)]

                before_u2u = time()
                candidate_workers = Geocrowd.createBipartiteGraphU2U(perturbedWList, perturbedTList,
                                                                     reachable_distance_map, reachable_prob_U2U, p)
                duration_u2u = time() - before_u2u

                """
                precision/recall during U2U
                """
                precisions, recalls = [], []
                for tid, wids in candidate_workers.items():
                    selected_workers = len(wids)
                    reachable_workers, reachable_workers_other = 0, 0
                    for wid in wids:
                        # noisy_dist = Utils.distance(wNoisyLoc[wid][0], wNoisyLoc[wid][1], tNoisyLoc[tid][0], tNoisyLoc[tid][1])
                        actual_dist = Utils.distance(wLoc[wid][0], wLoc[wid][1], tLoc[tid][0], tLoc[tid][1])
                        if actual_dist <= reachable_distance_map[wid]:
                            reachable_workers += 1  # actual domain
                            # if noisy_dist <= reachable_distance_map[wid] and actual_dist <= reachable_distance_map[wid]:
                            #     reachable_selected_workers += 1
                    # number of reachable workers in total
                    for wid, loc in wLoc.items():
                        if wid not in wids:
                            actual_dist_other = Utils.distance(loc[0], loc[1], tLoc[tid][0], tLoc[tid][1])
                            if actual_dist_other <= reachable_distance_map[wid]:
                                reachable_workers_other += 1  # actual domain
                    if selected_workers > 0:
                        precisions.append(float(reachable_workers) / selected_workers)
                    if reachable_workers > 0 or reachable_workers_other > 0:
                        recalls.append(float(reachable_workers) / (reachable_workers + reachable_workers_other))
                reachability_precision, reachability_recall, reachability_CAN = np.mean(precisions), np.mean(recalls), len(candidate_workers)

                # Ranking with reachable probability, resend strategy and worker's acceptance policy

                before_u2e = time()
                reachability_NAT, reachability_false_hits, reachability_WTD, reachability_false_dismissals, reachability_NNW = \
                    Geocrowd.rankingByReachabilityAnalytical(candidate_workers, range(p.taskCount), wLoc, tLoc,
                                                             reachable_distance_map, p.reachabilityThresholdU2E, p)
                duration_u2e = time() - before_u2e

                res_cube_reachability_NAT[i, j, k], res_cube_reachability_false_hits[i, j, k], \
                res_cube_reachability_WTD[i, j, k], res_cube_reachability_false_dismissals[i, j, k], \
                res_cube_reachability_NNW[i, j, k], res_cube_reachability_precision[i, j, k], \
                res_cube_reachability_recall[i, j, k], res_cube_reachability_u2u_runtime[i, j, k], res_cube_reachability_u2e_runtime[i, j, k], \
                res_cube_reachability_CAN[i, j, k] = \
                    reachability_NAT, reachability_false_hits, reachability_WTD, reachability_false_dismissals, \
                    reachability_NNW, reachability_precision, reachability_recall, duration_u2u, duration_u2e, reachability_CAN

            res_summary_reachability_NAT, res_summary_reachability_false_hits, res_summary_reachability_WTD, \
            res_summary_reachability_false_dismissals, res_summary_reachability_NNW, \
            res_summary_reachability_precision, res_summary_reachability_recall, res_summary_reachability_u2u_runtime, \
            res_summary_reachability_u2e_runtime, res_summary_reachability_CAN = \
                np.average(res_cube_reachability_NAT, axis=1), np.average(res_cube_reachability_false_hits, axis=1), \
                np.average(res_cube_reachability_WTD, axis=1), np.average(res_cube_reachability_false_dismissals, axis=1), \
                np.average(res_cube_reachability_NNW, axis=1), np.average(res_cube_reachability_precision, axis=1), \
                np.average(res_cube_reachability_recall, axis=1), np.average(res_cube_reachability_u2u_runtime, axis=1), \
                np.average(res_cube_reachability_u2e_runtime, axis=1), np.average(res_cube_reachability_CAN, axis=1)

            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_NAT" + '_' + str(eps),
                       res_summary_reachability_NAT,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_false_hits" + '_' + str(eps),
                       res_summary_reachability_false_hits,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_WTD" + '_' + str(eps),
                       res_summary_reachability_WTD,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_false_dismissals" + '_' + str(eps),
                       res_summary_reachability_false_dismissals,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_NNW" + '_' + str(eps),
                       res_summary_reachability_NNW,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_precision" + '_' + str(eps),
                       res_summary_reachability_precision,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_recall" + '_' + str(eps),
                       res_summary_reachability_recall,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_runtime_u2u" + '_' + str(eps),
                       res_summary_reachability_u2u_runtime,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_runtime_u2e" + '_' + str(eps),
                       res_summary_reachability_u2e_runtime,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_Reachability_Analytical_CAN" + '_' + str(eps),
                       res_summary_reachability_CAN,
                       header="\t".join([str(r) for r in U2E_threshold_list]), fmt='%.4f\t')

def evalOffline(p, wList, tList, wLoc, tLoc, reachable_distance_map):
    exp_name = ""
    logging.info(exp_name)

    res_cube = np.zeros((len(eps_list), len(seed_list), len(radius_list)))
    res_cube_WTD = np.zeros((len(eps_list), len(seed_list), len(radius_list)))

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
                    reachable_distance_map = {} # <wid, reachable_distance>
                    for wid in wLoc.keys():
                        reachable_distance_map[wid] = Utils.random_reachable_dist(p.reachableDistRange, p.seed)

                    # compute reachable distance such that a matched worker-task pair is
                    # reachable in the actual domain.
                    # coverageProb = coverageProbDict[Utils.RadiusEps2Str(p.radius, p.eps)]
                    if expanded:
                        exp_name = sys._getframe().f_code.co_name + "_expanded"
                    else:
                        exp_name = sys._getframe().f_code.co_name

                    workers = Geocrowd.createBipartiteGraph(perturbedWList, perturbedTList, reachable_distance_map)

                    # max flow
                    _, flowDict = Geocrowd.maxFlow(workers, range(p.taskCount))
                    matches = Geocrowd.flowDict2Matches(flowDict)


                    res_cube[i, j, k], res_cube_WTD[i, j, k] = \
                        Geocrowd.satisfiableMatches(matches, wLoc, tLoc, reachable_distance_map)

            res_summary, res_summary_WTD = np.average(res_cube, axis=1), np.average(res_cube_WTD, axis=1)
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name, res_summary,
                       header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')
            np.savetxt(p.resdir + Params.DATASET + "_" + exp_name + "_WTD", res_summary_WTD,
                       header="\t".join([str(r) for r in radius_list]), fmt='%.4f\t')


p = Params(1000)
p.select_dataset()

"""
Online
without privacy
"""
# without privacy
wList, tList, wLoc, tLoc = Geocrowd.sampleWorkersTasks(p.workerFile, p.taskFile, p.workerCount, p.taskCount)
reachable_distance_map = dict([(wid, Utils.random_reachable_dist(p.REACHABLE_DIST_RANGE, p.seed)) for wid in wLoc.keys()]) # <wid, reachable_distance>
workers = Geocrowd.createBipartiteGraph(wList, tList, reachable_distance_map)

ranking_nearest_NAT, ranking_nearest_false_hits, ranking_nearest_WTD, ranking_nearest_NNW = Geocrowd.rankingNearest(workers, range(p.taskCount), wLoc, tLoc, reachable_distance_map)
ranking_random_NAT, ranking_random_false_hits, ranking_random_WTD, ranking_random_NNW= Geocrowd.rankingRandom(workers, range(p.taskCount), wLoc, tLoc, reachable_distance_map)
print("Non-private Ranking-Nearest (NAT/False hits/WTD/NNW): ", ranking_nearest_NAT, ranking_nearest_false_hits, ranking_nearest_WTD, ranking_nearest_NNW)
print("Non-private Ranking-Random (NAT/False hits/WTD/NNW): ", ranking_random_NAT, ranking_random_false_hits, ranking_random_WTD, ranking_random_NNW)

# with privacy
pool = Pool(processes=len(eps_list))
params = []
for eps in eps_list:
    params.append((p, wList, tList, wLoc, tLoc, reachable_distance_map, eps))
pool.map(eval, params)
pool.close()
pool.join()
# eval(p, wList, tList, wLoc, tLoc, reachable_distance_map)

pool = Pool(processes=len(eps_list))
params = []
for eps in eps_list:
    params.append((p, wList, tList, wLoc, tLoc, reachable_distance_map, eps))
pool.map(eval_U2U_threshold, params)
pool.close()
pool.join()
# eval_U2U_threshold(p, wList, tList, wLoc, tLoc, reachable_distance_map)

pool = Pool(processes=len(eps_list))
params = []
for eps in eps_list:
    params.append((p, wList, tList, wLoc, tLoc, reachable_distance_map, eps))
pool.map(eval_U2E_threshold, params)
pool.close()
pool.join()
# eval_U2E_threshold(p, wList, tList, wLoc, tLoc, reachable_distance_map)

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
# evalOffline(p, wList, tList, wLoc, tLoc, reachable_distance_map)
