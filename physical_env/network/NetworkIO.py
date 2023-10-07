import copy
import simpy
import yaml
import random
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__))
from BaseStation import BaseStation
from Network import Network
from Node import Node
from Target import Target

class NetworkIO:
    def __init__(self, file_data):
        with open(file_data, 'r') as file:
            self.net_argc = yaml.safe_load(file)

    def makeNetwork(self):
        net_argc = copy.deepcopy(self.net_argc)
        self.node_phy_spe = net_argc["node_phy_spe"]
        self.seed = net_argc["seed"]
        np.random.seed(self.seed)
        random.seed(self.seed)
        listNodes = []
        listTargets = []
        for tmp in net_argc["nodes"]:
            listNodes.append(Node(location=tmp, phy_spe=copy.deepcopy(self.node_phy_spe)))
        for tmp in self.net_argc["targets"]:
            listTargets.append(Target(location=tmp))

        baseStation = BaseStation(location=net_argc["base_station"])
        env = simpy.Environment()
        return env, Network(env, listNodes, baseStation, listTargets, net_argc["max_time"])

