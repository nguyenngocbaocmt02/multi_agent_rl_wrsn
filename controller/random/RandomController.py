import random
import numpy as np
class RandomController:
    def __init__(self, net, mcs):
        self.net = net
        self.mcs = mcs
        self.max_charging_time = (self.net.listNodes[0].capacity - self.net.listNodes[0].threshold) /  (self.mcs[0].alpha / (self.mcs[0].chargingRange + self.mcs[0].beta) ** 2)


    def make_action(self, id):
        return self.translate([random.random(), random.random(), random.random()])
    
    def translate(self, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                self.max_charging_time * action[2]])
