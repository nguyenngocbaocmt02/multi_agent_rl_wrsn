import yaml
import sys
import os
import random
from simpy.events import AnyOf
import copy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from environments.network.NetworkIO import NetworkIO
from environments.mc.MobileCharger import MobileCharger
from controller.bao.BaoController import BaoController
from controller.random.RandomController import RandomController


class WRSN:
    def __init__(self, phy_para, topology):
        self.phy_para = phy_para
        self.topology = topology
        self.netIO = NetworkIO("environments/data/test.yaml", phy_para["node_phy_spe"])


with open("environments/para.yaml", "r") as f:
    phy_para = yaml.safe_load(f)
print(phy_para)


random.seed(0)

netIO = NetworkIO("E:/gym-tc-wrsn/environments/network/network_samples/test.yaml")
env, net = netIO.makeNetwork()
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), phy_para["mc_phy_spe"]) for _ in range(3)]
controller = RandomController(net, mcs)

processes = []
processes.append(env.process(net.operate(max_time=100000)))
# Loop through each controller and add its process to the list
for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    process = env.process(mc.operate(controller=controller))
    processes.append(process)

