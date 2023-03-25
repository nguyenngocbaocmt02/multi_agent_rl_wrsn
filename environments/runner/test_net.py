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
env.process(mc.move(destination=np.array([100000.0, 10000.0])))
env.run(until=x)