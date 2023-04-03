import yaml
import sys
import os
import random
import copy
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from environments.iostream.NetworkIO import NetworkIO
from environments.mc.MobileCharger import MobileCharger
from controller.bao.BaoController import BaoController
from controller.random.RandomController import RandomController

with open("environments/para.yaml", "r") as f:
    testcase = yaml.safe_load(f)
print(testcase)

netIO = NetworkIO(testcase["default_net"], testcase["node_phy_spe"])
random.seed(testcase['seed'])

env, net = netIO.makeNetwork()
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), testcase["mc_phy_spe"]) for _ in range(3)]
#controller = BaoController(net, mcs)
controller = RandomController(net, mcs)
for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    env.process(mc.operate(controller=controller))
x = env.process(net.operate(max_time=100000))
env.run(until=x)