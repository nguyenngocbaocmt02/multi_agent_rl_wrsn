import argparse
import random
import torch
import yaml
from environments.env.wrsn.WRSN_ver1 import WRSN

parser = argparse.ArgumentParser(description="Train rl agent.")
parser.add_argument("--save-path", type=str, nargs="?", default="./", help="Please enter the directory of saving model.")
parser.add_argument("--alg", type=str, nargs="?", default="maddpg", help="Please enter the alg name.")
parser.add_argument("--env", type=str, nargs="?", default="wrsn", help="Please enter the env name.")
parser.add_argument("--alias", type=str, nargs="?", default="", help="Please enter the alias for exp control.")
parser.add_argument("--mode", type=str, nargs="?", default="decentralized", help="Please enter the mode: distributed or decentralised.")
parser.add_argument("--scenario", type=str, nargs="?", default="test.yaml", help="Please input the valid name of an node scenario.")
parser.add_argument("--render", action="store_true", default= False, help="Activate the rendering of the environment.")
parser.add_argument("--n-mcs", type=int, nargs="?", default=2, help="Please input the number off agents")
parser.add_argument("--degree", type=int, nargs="?", default=10, help="Please input the degree of Taylor")
parser.add_argument("--test-mode", type=str, nargs="?", default="single", help="Please input the valid test mode: single or batch.")
argv = parser.parse_args()

print(parser)
# load env args
with open("C:/gym-tc-wrsn/args/env_args/wrsn.yaml", "r") as f:
    env_config_dict = yaml.safe_load(f)["env_args"]
scenario = env_config_dict["scenario"].split("/")
scenario[-1] = argv.scenario
env_config_dict["scenario"] = "C:/gym-tc-wrsn/environments/data/test.yaml"
env_config_dict["n_mcs"] = argv.n_mcs
env_config_dict["degree"] = argv.degree

assert argv.mode in ['distributed', 'decentralized'], "Please input the correct mode, e.g. distributed or decentralised."
env_config_dict["mode"] = argv.mode

# for one-day test
env_config_dict["episode_limit"] = 240

# load default args
with open("C:/gym-tc-wrsn/args/default.yaml", "r") as f:
    default_config_dict = yaml.safe_load(f)
default_config_dict["max_steps"] = 240
wrsn = WRSN(env_config_dict)
for t in range(10):
    actions = torch.Tensor([random.random() for i in range(6)]) # a vector involving all agents' actions
    print(actions)
    actions[2] = (1 - actions[1] ** 2) ** 0.5
    actions[5] = (1 - actions[4] ** 2) ** 0.5
    reward, done, info = wrsn.step(actions)
    next_state = wrsn.get_obs()
    state = next_state
