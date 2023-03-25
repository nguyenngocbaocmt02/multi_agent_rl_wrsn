from environments.Extension import *
import numpy as np
from scipy.spatial.distance import euclidean

class MobileCharger:
    def __init__(self, location, mc_phy_spe):
        """
        The initialization for a MC.
        :param env: the time management system of this MC
        :param location: the initial coordinate of this MC, usually at the base station
        """
        self.env = None
        self.net = None
        self.id = None

        self.location = np.array(location)
        self.energy = mc_phy_spe['capacity']
        self.capacity = mc_phy_spe['capacity']

        self.alpha = mc_phy_spe['alpha']
        self.beta = mc_phy_spe['beta']
        self.threshold = mc_phy_spe['threshold']
        self.velocity = mc_phy_spe['velocity']
        self.pm = mc_phy_spe['pm']
        self.chargingRate = 0
        self.chargingRange = mc_phy_spe['charging_range']
        self.epsilon = mc_phy_spe['epsilon']
        self.status = 1
        self.checkStatus()
        self.log = []

    def charge_step(self, nodes, t=1):
        """
        The charging process to nodes in 'nodes' within simulateTime
        :param nodes: the set of charging nodes
        :param t: the status of MC is updated every t(s)
        """
        for node in nodes:
            node.charger_connection(self)

        if self.chargingRate != 0:
            charge_time = min(t, (self.energy - self.threshold) / self.chargingRate)
        else:
            charge_time = t

        yield self.env.timeout(charge_time)
        self.energy = self.energy - self.chargingRate * charge_time
        self.checkStatus()
        print(self.env.now, ": MC ", self.id, " charge: ", self.energy, self.chargingRate, self.location)

        for node in nodes:
            node.charger_disconnection(self)
        self.chargingRate = 0
        return

    def charge(self, chargingTime):
        tmp = chargingTime
        nodes = []
        for node in self.net.listNodes:
            if euclideanDistance(node.location, self.location) <= self.chargingRange:
                nodes.append(node)
        while True:
            span = min(tmp, 1.0)
            yield self.env.process(self.charge_step(nodes, t=span))
            tmp -= span
            if tmp == 0 or self.status == 0:
                break
        return

    def move_step(self, vector, t):
        yield self.env.timeout(t)
        self.location = self.location + vector
        self.energy -= self.pm * t * self.velocity
        print(self.env.now, ": MC ", self.id, " move: ", self.energy, self.chargingRate, self.location)
        self.checkStatus()

    def move(self, destination):
        print(destination, self.location, self.velocity)
        moving_time = euclidean(destination, self.location) / self.velocity
        moving_vector = destination - self.location
        print(moving_time)
        tmp = moving_time
        while True:
            span = min(tmp, 1.0)
            yield self.env.process(self.move_step(moving_vector / moving_time * span, t=span))
            tmp -= span
            if tmp == 0:
                self.location = destination
                break
            if self.status == 0:
                break
        return

    def self_charge(self):
        print(self.env.now, ": MC ", self.id, " self charge: ", self.energy, self.chargingRate, self.location)
        print(self.location, self.net.baseStation.location)
        if euclideanDistance(self.location, self.net.baseStation.location) <= self.epsilon:
            for i in range(len(self.location)):
                self.location[i] = self.net.baseStation.location[i]
            self.energy = self.capacity
        print(self.env.now, ": MC ", self.id, " self charge: ", self.energy, self.chargingRate, self.location)
        return

    def checkStatus(self):
        """
        check the status of MC
        """
        if self.energy <= self.threshold:
            self.status = 0
            self.energy = self.threshold
