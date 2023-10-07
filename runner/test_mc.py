import sys
import os
import time
import copy
import yaml
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger

def log(net, mcs):
    # If you want to print something, just put it here. Do not revise the core code.
    while True:
        print(net.env.now, net.listNodes[0].energy)
        yield net.env.timeout(1.0)

netIO = NetworkIO("physical_env/network/network_scenarios/hanoi1000n50.yaml")
env, net = netIO.makeNetwork()

with open("physical_env/mc/mc_types/default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(3)]

for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
    mc.log = [net.baseStation.location[0], net.baseStation.location[1], 0]
mc0_process = env.process(mcs[0].operate_step([0.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.8 * (net.frame[3] - net.frame[2]) + net.frame[2], 75]))
mc1_process = env.process(mcs[1].operate_step([0.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.8 * (net.frame[3] - net.frame[2]) + net.frame[2], 50]))   
mc2_process = env.process(mcs[2].operate_step([0.35 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.6 * (net.frame[3] - net.frame[2]) + net.frame[2], 100]))    
net_process = env.process(net.operate())

general_process = mc0_process | mc1_process | mc2_process | net_process

env.process(log(net, mcs))
env.run(until = net_process)
#env.run(until = general_process)