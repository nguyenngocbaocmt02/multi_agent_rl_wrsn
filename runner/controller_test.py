import yaml
import sys
import os
import random
from simpy.events import AnyOf
import copy
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from controller.bao.BaoController import BaoController
from controller.random.RandomController import RandomController


class WRSN:
    def __init__(self, phy_para, topology):
        self.phy_para = phy_para
        self.topology = topology
        self.netIO = NetworkIO("environments/data/test.yaml", phy_para["node_phy_spe"])


random.seed(0)

netIO = NetworkIO("../physical_env/network/network_scenarios/test.yaml")
env, net = netIO.makeNetwork()

with open("../physical_env/mc/mc_types/default.yaml", "r") as file:
    mc_phy_para = yaml.safe_load(file)
 
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_para) for _ in range(3)]
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
env.run(until=processes[0])

