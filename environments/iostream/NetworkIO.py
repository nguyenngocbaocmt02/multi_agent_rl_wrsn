import copy
from collections import namedtuple

import simpy
import yaml

from environments.network.BaseStation import BaseStation
from environments.network.Network import Network
from environments.network.Node import Node
from environments.network.Target import Target


def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


class NetworkIO:
    def __init__(self, file_data, args):
        with open(file_data, 'r') as file:
            net_argc = yaml.safe_load(file)
        if isinstance(args, dict):
            args = convert(args)
        self.listNodes = []
        self.listTargets = []
        for tmp in net_argc["nodes"]:
            self.listNodes.append(Node(location=copy.deepcopy(tmp), energy=args.node_capacity
                                       , threshold=args.node_threshold, capacity=args.node_capacity
                                       , phy_spe=convert(args.node_phy_spe)))

        for tmp in net_argc["targets"]:
            self.listTargets.append(Target(location=copy.deepcopy(tmp)))

        self.baseStation = BaseStation(location=copy.deepcopy(net_argc["base_station"]))

    def makeNetwork(self):
        env = simpy.Environment()
        return env, Network(env, copy.deepcopy(self.listNodes),
                            copy.deepcopy(self.baseStation), copy.deepcopy(self.listTargets))

'''
with open("C:/gym-tc-wrsn/args/env_args/wrsn.yaml", "r") as f:
    env_config_dict = convert(yaml.safe_load(f)["env_args"])
    x = NetworkIO(file_data="D:/gym-tc-wrsn/environments/data/test.yaml", args=env_config_dict)
    print(x.baseStation.location)
    print(x.listTargets[0].location)
    print(x.listNodes[0].location)
'''