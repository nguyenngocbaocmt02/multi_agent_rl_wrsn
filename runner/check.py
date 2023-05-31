import yaml
import sys
import os
import random
from simpy.events import AnyOf
import copy
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from controller.bao.BaoController import BaoController
from controller.random.RandomController import RandomController
import matplotlib.pyplot as plt

class WRSN:
    def __init__(self, phy_para, topology):
        self.phy_para = phy_para
        self.topology = topology
        self.netIO = NetworkIO("environments/data/test.yaml", phy_para["node_phy_spe"])


random.seed(0)

netIO = NetworkIO("../physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = netIO.makeNetwork()

node_x = [node.location[0] for node in net.listNodes]
node_y = [node.location[1] for node in net.listNodes]
target_x = [target.location[0] for target in net.listTargets]
target_y = [target.location[1] for target in net.listTargets]

plt.scatter(np.array(node_x), np.array(node_y))
plt.scatter(np.array([net.baseStation.location[0]]), np.array([net.baseStation.location[1]]), c="red")
plt.scatter(np.array(target_x), np.array(target_y), c="green")
plt.scatter(np.array(target_x[0]), np.array(target_y[0]), c="purple")
print(target_x[0], target_y[0])
plt.show()