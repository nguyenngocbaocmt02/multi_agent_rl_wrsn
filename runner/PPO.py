import sys
import os
import torch
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.ppo.PPO import PPO
import numpy as np
from rl_env.WRSN import WRSN
import yaml

def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        print(net.env.now, net.listNodes[0].energy)
        yield net.env.timeout(1.0)

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


env = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=3, map_size=100, density_map=True)

with open("alg_args/ppo.yaml", 'r') as file:
    args = yaml.safe_load(file)
controller = PPO(args['alg_args'], device)


controller.train(env, 1000, save_folder="save_model/ppo")