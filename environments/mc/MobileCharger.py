import numpy as np
from scipy.spatial.distance import euclidean
import copy

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
        #print("MC " + str(self.id) + " Charging", self.location, self.energy, self.chargingRate)
        yield self.env.timeout(t)
        self.energy = self.energy - self.chargingRate * t

        for node in nodes:
            node.charger_disconnection(self)
        self.chargingRate = 0
        return

    def charge(self, chargingTime):
        tmp = chargingTime
        nodes = []
        for node in self.net.listNodes:
            if euclidean(node.location, self.location) <= self.chargingRange:
                nodes.append(node)
        while True:
            span = min(tmp, 1.0)
            if self.chargingRate != 0:
                span = min(span, (self.energy - self.threshold) / self.chargingRate)
            yield self.env.process(self.charge_step(nodes, t=span))
            tmp -= span
            self.checkStatus()
            if tmp == 0 or self.status == 0:
                break
        return

    def move_step(self, vector, t):
        yield self.env.timeout(t)
        self.location = self.location + vector
        self.energy -= self.pm * t * self.velocity
        

    def move(self, destination):
        moving_time = euclidean(destination, self.location) / self.velocity
        if moving_time == 0:
            return
        moving_vector = destination - self.location

        while True:
            #print("MC " + str(self.id) + " Moving", self.location, self.energy, self.chargingRate)
            span = min(min(moving_time, 1.0), (self.energy - self.threshold) / (self.pm * self.velocity))
            yield self.env.process(self.move_step(moving_vector / moving_time * span, t=span))
            moving_time -= span
            if moving_time == 0 or self.status == 0:
                break
        return

    def recharge(self):
        if euclidean(self.location, self.net.baseStation.location) <= self.epsilon:
            self.location = copy.deepcopy(self.net.baseStation.location)
            self.energy = self.capacity
        yield self.env.timeout(0)
    
    def operate_step(self, action):
        print(action)
        destination = np.array([action[0], action[1]])
        chargingTime = action[2]

        usedEnergy = euclidean(destination, self.location) * self.pm
        tmp = 0
        for node in self.net.listNodes:
            if euclidean(node.location, self.location) <= self.chargingRange and node.status == 1:
               tmp += self.alpha / (euclidean(self.location, node.location) + self.beta) ** 2
        usedEnergy += tmp * chargingTime
        usedEnergy += euclidean(destination, self.net.baseStation.location) * self.pm

        if usedEnergy > self.energy - self.threshold + self.capacity / 200.0:
            yield self.env.process(self.move(destination=self.net.baseStation.location))
            yield self.env.process(self.recharge())
            return
                
        yield self.env.process(self.move(destination=destination))
        yield self.env.process(self.charge(chargingTime=chargingTime))
    
    def operate(self, controller):
        while True:
            action = controller.make_action(self.id)
            yield self.env.process(self.operate_step(action))

    def checkStatus(self):
        """
        check the status of MC
        """
        if self.energy <= self.threshold:
            self.status = 0
            self.energy = self.threshold
