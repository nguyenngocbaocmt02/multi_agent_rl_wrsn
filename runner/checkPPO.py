import sys
import os
import torch
import random
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.ppo.PPO2 import PPO
import numpy as np
from rl_env.WRSN import WRSN
import yaml

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=2, map_size=100)

with open("alg_args/ppo.yaml", 'r') as file:
    args = yaml.safe_load(file)
controller = PPO(args['alg_args'])
controller.train(env, 1000, save_folder="save_model/ppo")