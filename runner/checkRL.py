import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from controller.random.RandomController import RandomController
from rl_env.WRSN import WRSN

network = WRSN(scenario_path="../physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,mc_type_path="../physical_env/mc/mc_types/default.yaml"
               ,num_mc=2, map_size=100)
controller = RandomController()
request = network.reset()
while not request["terminal"]:
    print(request["agent_id"], request["prev_fitness"], request["action"], request["fitness"], request["terminal"])
    request = network.step(request["agent_id"], controller.make_action(request["state"]))
print(network.net.check_nodes())
