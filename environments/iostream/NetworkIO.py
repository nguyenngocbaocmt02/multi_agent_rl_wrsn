import copy
from collections import namedtuple
import simpy
import yaml
from environments.network.BaseStation import BaseStation
from environments.network.Network import Network
from environments.network.Node import Node
from environments.network.Target import Target

class NetworkIO:
    def __init__(self, file_data, node_phy_spe):
        with open(file_data, 'r') as file:
            net_argc = yaml.safe_load(file)

        self.listNodes = []
        self.listTargets = []
        for tmp in net_argc["nodes"]:
            self.listNodes.append(Node(location=copy.deepcopy(tmp), phy_spe=node_phy_spe))
        for tmp in net_argc["targets"]:
            self.listTargets.append(Target(location=copy.deepcopy(tmp)))

        self.baseStation = BaseStation(location=copy.deepcopy(net_argc["base_station"]))

    def makeNetwork(self):
        env = simpy.Environment()
        return env, Network(env, copy.deepcopy(self.listNodes),
                            copy.deepcopy(self.baseStation), copy.deepcopy(self.listTargets))

