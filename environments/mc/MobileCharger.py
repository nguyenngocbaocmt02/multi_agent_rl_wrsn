from environments.Extension import *


class MobileCharger:

    def __init__(self, location, mc_specification):
        """
        The initialization for a MC.
        :param env: the time management system of this MC
        :param location: the initial coordinate of this MC, usually at the base station
        """
        self.env = None
        self.net = None
        self.id = None

        self.location = location
        self.energy = mc_specification.mc_capacity
        self.capacity = mc_specification.mc_capacity

        self.alpha = mc_specification.mc_alpha
        self.beta = mc_specification.mc_beta
        self.threshold = mc_specification.mc_threshold
        self.velocity = mc_specification.mc_velocity
        self.pm = mc_specification.mc_pm
        self.chargingRate = 0
        self.chargingRange = mc_specification.mc_charging_range
        self.epsilon = mc_specification.epsilon
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

    def move_step(self, vector, t=1):
        move_time = min(t, (self.energy - self.threshold) / self.pm)
        yield self.env.timeout(move_time)
        for i in range(len(vector)):
            self.location[i] += (vector[i] * self.velocity * move_time)
        self.energy -= self.pm * move_time
        print(self.env.now, ": MC ", self.id, " move: ", self.energy, self.chargingRate, self.location)
        self.checkStatus()

    def move(self, vector, moving_time, to_base):
        if to_base:
            moving_time = min(moving_time,
                              euclideanDistance(self.net.baseStation.location, self.location) / self.velocity)
        tmp = moving_time
        vector = regularize(vector)
        if vector is not None:
            while True:
                span = min(tmp, 1.0)
                yield self.env.process(self.move_step(vector, t=span))
                tmp -= span
                if tmp == 0 or self.status == 0:
                    break
        if to_base:
            self.self_charge()
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
