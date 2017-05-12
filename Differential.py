import numpy as np
import math
from Params import Params

class Differential(object):

    def __init__(self, seed):
        np.random.seed(seed)

    def radiusOfRetrieval(self, radiusInt, eps, radius, c):
        """

        :param radiusInt: radius of interest
        :param eps:
        :param c: confidence (prob)
        :return:
        """
        return radiusInt + self.inverseCumulativeGamma(eps, c) * radius

    def addPolarNoise(self, eps, radius, pos):
        # random number in [0, 2 * PI)
        theta = np.random.rand() * 2 * math.pi

        # random variable in [0, 1)
        z = np.random.rand()
        r = self.inverseCumulativeGamma(eps, z) * radius
        return self.addVectorToPos(pos, r, theta)

    def addPolarNoiseCartesian(self, eps, pos):
        pos = self.getCartesian(pos)

        # random number in [0, 2 * PI)
        theta = np.random.rand() * 2 * math.pi

        # random variable in [0, 1)
        z = np.random.rand()
        r = self.inverseCumulativeGamma(eps, z)

        return self.getLatLon((pos[0] + r * math.cos(theta), pos[1] + r * math.sin(theta)))


    def rad_of_deg(self, ang):
        return ang * math.pi / 180

    def deg_of_rad(self, ang):
        return ang * 180 / math.pi

    # http://www.movable-type.co.uk/scripts/latlong.html
    def addVectorToPos(self, pos, distance, angle):
        ang_distance = distance / Params.EARTH_RADIUS

        lat1 = self.rad_of_deg(pos[0])
        lon1 = self.rad_of_deg(pos[1])

        lat2 = math.asin(math.sin(lat1) * math.cos(ang_distance) + math.cos(lat1) * math.sin(ang_distance) * math.cos(angle))
        lon2 = lon1 + math.atan2(math.sin(angle) * math.sin(ang_distance) * math.cos(lat1),
                                 math.cos(ang_distance) - math.sin(lat1) * math.sin(lat2))

        lon2 = (lon2 + 3*math.pi) % (2*math.pi) - math.pi # normalize to -180 to +180
        return (self.deg_of_rad(lat2), self.deg_of_rad(lon2))

    # LamberW function on branch -1 (http://en.wikipedia.org/wiki/Lambert_W_function)
    def LambertW(self, x):
        min_diff = 1e-10
        if x == -1/math.e:
            return -1
        elif 0 > x > -1/math.e:
            q = np.log(-x)
            p = 1
            while abs(p-q) > min_diff:
                p = (q*q + x/math.exp(q))/(q+1)
                q = (p*p + x/math.exp(p))/(p+1)
            # This line decides the precision of the float number that would be returned
            return round(q, 6)
        else:
            return 0

    def inverseCumulativeGamma(self, eps, z):
        x = (z-1)/math.e
        return -(self.LambertW(x) + 1)/eps

    def getLatLon(self, cart):
        rLon = cart[0] / Params.EARTH_RADIUS
        rLat = 2 * (math.atan(math.exp(cart[1]/Params.EARTH_RADIUS)))  - math.pi/2
        return (self.deg_of_rad(rLat), self.deg_of_rad(rLon))

    def getCartesian(self, ll):
        # latitude and longitude are converted in radiants
        return (Params.EARTH_RADIUS * self.rad_of_deg(ll[1]), Params.EARTH_RADIUS * math.log(math.tan(math.pi/4 + self.rad_of_deg(ll[0])/2)))



    def getNoise(self, sens, eps):
        """Get simple Laplacian noise"""
        return np.random.laplace(0, sens / eps, 1)[0]