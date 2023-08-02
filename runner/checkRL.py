import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.random.RandomController import RandomController
from rl_env.WRSN import WRSN
from utils import draw_heatmap_state


def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        if net.env.now % 100 == 0:
            print(net.env.now)
        yield net.env.timeout(1.0)

network = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=3, map_size=100,density_map=True)
controller = RandomController()

request = network.reset()
for id, _ in enumerate(network.net.targets_active):
    if _ == 0:
        print(id)

#network.env.process(log(network.net, network.agents))   
while not request["terminal"]:
    print(request["agent_id"], request["action"], request["detailed_rewards"], request["terminal"])
    action = controller.make_action(request["agent_id"], request["state"], request["info"], network)
    request = network.step(request["agent_id"], action)
    
print(network.net.env.now)