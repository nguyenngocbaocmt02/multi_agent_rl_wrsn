import copy
import simpy
import yaml
import random
import numpy as np
from environments.network.BaseStation import BaseStation
from environments.network.Network import Network
from environments.network.Node import Node
from environments.network.Target import Target

class NetworkIO:
    def __init__(self, file_data="environments/network/network_samples/test.yaml"):
        with open(file_data, 'r') as file:
            self.net_argc = yaml.safe_load(file)

    def makeNetwork(self):
        net_argc = copy.deepcopy(self.net_argc)
        node_phy_spe = net_argc["node_phy_spe"]
        seed = net_argc["seed"]
        np.random.seed(seed)
        random.seed(seed)
        listNodes = []
        listTargets = []
        for tmp in net_argc["nodes"]:
            listNodes.append(Node(location=tmp, phy_spe=copy.deepcopy(node_phy_spe)))
        for tmp in self.net_argc["targets"]:
            listTargets.append(Target(location=tmp))

        baseStation = BaseStation(location=net_argc["base_station"])
        env = simpy.Environment()
        return env, Network(env, listNodes, baseStation, listTargets)

