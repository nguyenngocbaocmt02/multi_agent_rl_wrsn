import yaml
import sys
import os
import random
import copy
import numpy as np
from collections import namedtuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from environments.iostream.NetworkIO import NetworkIO
from environments.mc.MobileCharger import MobileCharger

class RandomController:
    def __init__(self, net, mc):
        self.net = net
        self.mc = mc
        self.max_charging_time = (self.net.listNodes[0].capacity - self.net.listNodes[0].threshold) /  (mc.alpha / (mc.chargingRange + mc.beta) ** 2)


    def make_action(self):
        return self.translate([random.random(), random.random(), random.random()])
    
    def translate(self, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                self.max_charging_time * action[2]])


with open("environments/para.yaml", "r") as f:
    testcase = yaml.safe_load(f)
print(testcase)
netIO = NetworkIO(testcase["default_net"], testcase["node_phy_spe"])
random.seed(testcase['seed'])

env, net = netIO.makeNetwork()
mc = MobileCharger(copy.deepcopy(net.baseStation.location), testcase["mc_phy_spe"])
mc.env = env
mc.net = net
mc.id = 1
x = env.process(net.operate(max_time=100000))
env.process(mc.operate(controller=RandomController(net, mc)))
env.run(until=x)