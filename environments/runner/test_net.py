import simpy
import yaml
import sys
import os
import random
from collections import namedtuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from environments.iostream.NetworkIO import NetworkIO

def convert(dictionary):
    return namedtuple('GenericDict', dictionary.keys())(**dictionary)


node_phy_spe = dict()
node_phy_spe["node_com_range"] = 100
node_phy_spe["node_prob_gp"] = 0.8
node_phy_spe["package_size"] = 400.0
node_phy_spe["er"] = 0.0001
node_phy_spe["et"] = 0.00005
node_phy_spe["efs"] = 0.00000001
node_phy_spe["emp"] = 0.0000000000013


env_config_dict = dict()
env_config_dict["node_capacity"] = 10800
env_config_dict["node_threshold"] = 580

# env_config_dict["reset_action"] = True
env_config_dict["n_mcs"] = 0
env_config_dict["degree"] = 10
env_config_dict["node_phy_spe"] = node_phy_spe


netIO = NetworkIO("environments/data/test.yaml", env_config_dict)
env, net = netIO.makeNetwork()

env.process(net.operate(max_time=500))
env.run(until=505)
print(net.listNodes[231].energy)
print(env.now)