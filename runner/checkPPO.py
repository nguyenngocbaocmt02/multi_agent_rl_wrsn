import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.ppo.PPO import PPO
import numpy as np
from rl_env.WRSN import WRSN
import yaml

np.random.seed(0)
env = WRSN(scenario_path="physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,agent_type_path="physical_env/mc/mc_types/default.yaml"
               ,num_agent=2, map_size=100)

with open("alg_args/ppo.yaml", 'r') as file:
    args = yaml.safe_load(file)
controller = PPO(args['alg_args'])
controller.train(env, 1000000, 100, save_folder="save_model/ppo")


