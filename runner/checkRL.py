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
from rl_env.WRSN import WRSN
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import euclidean
import pandas as pd

network = WRSN(scenario_path="../physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,mc_type_path="../physical_env/mc/mc_types/default.yaml"
               ,num_mc=2, map_size=100)
controller = RandomController()
request = network.reset()
while not request["terminal"]:
    print(request["agent_id"], request["prev_fitness"], request["action"], request["fitness"], request["terminal"])
    request = network.step(request["agent_id"], controller.make_action(request["state"]))
print(network.net.check_nodes())
