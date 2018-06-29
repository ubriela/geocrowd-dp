import math
import random
import numpy as np
import Utils
from Differential import Differential
from Params import Params
from collections import defaultdict, OrderedDict
import bisect

random.seed(1000)
def dist(x, y):
    return math.sqrt((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2)

# Square
if False:

    total = 0
    N = 100000
    for i in range(N):
        x = [random.uniform(0, 1), random.uniform(0, 1)]
        y = [random.uniform(0, 1), random.uniform(0, 1)]
        d = dist(x, y)
        total += d

    print ("Expected dist: ", total/N)

if False:
    total = 0
    N = 100000
    for i in range(N):
        r, theta = [math.sqrt(random.uniform(0, 1)) * math.sqrt(1), 2 * math.pi * random.uniform(0, 1)]
        x = [math.cos(theta) * r, math.sin(theta) * r]
        r, theta = [math.sqrt(random.uniform(0, 1)) * math.sqrt(1), 2 * math.pi * random.uniform(0, 1)]
        y = [math.cos(theta) * r, math.sin(theta) * r]
        d = dist(x, y)
        total += d

    print ("Expected dist: ", total/N)


"""
Simulated dataset
"""
p = Params(1000)
p.select_dataset()
reachable_range = Utils.reachable_distance()
dp = Differential(p.seed)

# Randomly picking location in a small MBR of tdrive dataset.
minLat, maxLat = 39.1232147, 40.7225952
minLon, maxLon = 115.3879166, 117.3795395

diffLat = maxLat - minLat
diffLon = maxLon - minLon

maxLat = maxLat - 0.95*diffLat
maxLon = maxLon - 0.95*diffLon

# print ("diagonal dist: ", Utils.distance(minLat, minLon, maxLat, maxLon))

def probs_from_sampling(samples, step, d_prime_values, d_matches_values):
    """
    For all worker-task pairs, compute d=dist(w,t) and d'=dist(w',t').
    We discretize d' into ranges [u,2*u,3*u,..,n*u] where x-range means d' <= x.

    We want to compute a map from d'-range to values in actual domain.
    For each d'-range, we can compute the PDF of d values.
    """
    d_prime_range_to_d_values = defaultdict(list)  # (d'-ranges, d values)
    # we want to sorted during constructing this list due to performance
    reachable_d_to_d_primes = defaultdict(list) # key is reachable distance d
    reachable_d_prime_to_d_values = defaultdict(list) # key is reachable distance d_prime
    for i in range(samples):
        # First point
        lat1, lon1 = random.uniform(minLat, maxLat), random.uniform(minLon, maxLon)
        noisyLat1, noisyLon1 = dp.addPolarNoise(p.eps, p.radius, (lat1, lon1))

        # Second point
        lat2, lon2 = random.uniform(minLat, maxLat), random.uniform(minLon, maxLon)
        noisyLat2, noisyLon2 = dp.addPolarNoise(p.eps, p.radius, (lat2, lon2))

        d = Utils.distance(lat1, lon1, lat2, lon2)
        d_prime = Utils.distance(noisyLat1, noisyLon1, noisyLat2, noisyLon2)

        # d_prime range
        d_prime_range = Utils.dist_range(d_prime, step)
        d_prime_range_to_d_values[d_prime_range].append(d)

        # insert d_prime values of all reachable pairs
        for reachable_dist in reachable_range:
            if d <= reachable_dist:
                bisect.insort(reachable_d_to_d_primes[reachable_dist], d_prime)
            if d_prime <= reachable_dist:
                bisect.insort(reachable_d_prime_to_d_values[reachable_dist], d)


    # print ("\n".join(map(str, d_prime_range_to_d_values[1000.0])))

    """
    Given worker-task distance in the perturbed domain d'=dist(w',t'),
    compute the reachable probability in the actual domain: P_reachable = Pr(dist(w, t) < d_reachable).
    """
    reachable_prob = defaultdict(list)
    for d_prime in d_prime_values:
        d_prime_range = Utils.dist_range(d_prime, step)
        d_values = d_prime_range_to_d_values[d_prime_range] # actual d_values that has noisy distance in a certain range.
        d_values.sort() # Sort values in map
        for reachable_dist in reachable_range:
            cum_prob = Utils.cumulative_prob(d_values, reachable_dist)
            reachable_prob[reachable_dist].append((d_prime, cum_prob))
    """
    Compute the upper-bound matching distance match_dist(w',t') in the perturbed domain
    (that SC-server would match a worker to a task) such that with high probability
    a reachable pair in the actual domain (dist(w, t) < d_reachable) are matched in the noisy domain: P_recall.
    """
    precision_recall_prob = defaultdict(list)
    for d_match in d_matches_values:
        for reachable_dist in reachable_range:
            precision = Utils.cumulative_prob(reachable_d_prime_to_d_values[d_match], reachable_dist)
            recall = Utils.cumulative_prob(reachable_d_to_d_primes[reachable_dist], d_match)
            precision_recall_prob[reachable_dist].append((d_match, precision, recall))

    return reachable_prob, precision_recall_prob

samples = 50000  # sample size
d_prime_values = range(100, Params.MAX_DIST + 1, 100) # range of noisy distance
d_matches_values = range(100, Params.MAX_DIST + 1, 100) # range of matching distance
def precomputeProbability():
    """
    Precompute probability of reachability given epsilon
    :param eps:
    :return:
    """
    for radius in [1000.0]:
        for eps in [1.0]:
            print ("radius/eps: ", radius, eps)
            p.radius, p.eps = radius, eps
            outputFile = Utils.getParameterizedFile(radius, eps)
            with open(outputFile + "_reachability.txt", "w") as f_reachable, \
                    open(outputFile + "_precision_recall.txt", "w") as f_precision_recall:
                lines = ""
                reachable_prob, precision_recall_prob = probs_from_sampling(samples, Params.STEP, d_prime_values, d_matches_values)
                for reachable_dist in reachable_range:
                    for d_prime, prob in reachable_prob[reachable_dist]:
                        lines += str(reachable_dist) + "\t" + str(d_prime) + "\t" + str(prob) + "\n"
                f_reachable.write(lines)

                lines = ""
                for reachable_dist in reachable_range:
                    for d_match, precision, recall in precision_recall_prob[reachable_dist]:
                        lines += str(reachable_dist) + "\t" + str(d_match) + "\t" + str(precision) + "\t" + str(recall) + "\n"

                f_precision_recall.write(lines)
            f_reachable.close()
            f_precision_recall.close()

# precomputeProbability()

def getProbability(radius_list, eps_list, suffix):
    """
    Get precomputed smooth sensitivity
    :param C_list:
    :param eps_lis:
    :return: mapping from C and eps to sensitivity list
    """
    dict = defaultdict(OrderedDict)
    for radius in radius_list:
        for eps in eps_list:
            inputFile = Utils.getParameterizedFile(radius, eps) + "_" + suffix + ".txt"
            # print inputFile
            data = np.loadtxt(inputFile, dtype=float, delimiter="\t")
            k = Utils.radius_eps_2_str(radius, eps)
            v = OrderedDict({str(int(kv[0])) + ":" + str(int(kv[1])) : kv[2] for kv in data[:, :]})
            # print k, v
            dict[k] = v
    return dict

radius, eps, dp = 200.0, 1.0, 2000.0
key = Utils.radius_eps_2_str(radius, eps)
print getProbability([radius], [eps], "reachability")[key]
di = dict()
for k, v in (getProbability([radius], [eps], "reachability")[key]).items():
    rd, d_prime = map(int, k.split(':'))
    if d_prime == dp:
        di[rd] = v

for rd in sorted(di.iterkeys()):
    print "%s\t%s" % (((rd + 0.0)/1000)**2, di[rd])

