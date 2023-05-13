import sys
import os
import copy
import yaml
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger


netIO = NetworkIO("../physical_env/network/network_scenarios/test.yaml")
env, net = netIO.makeNetwork()

with open("../physical_env/mc/mc_types/default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(3)]

for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id
print([2.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 2.8 * (net.frame[3] - net.frame[2]) + net.frame[2]])
env.process(mcs[0].charge(200))
env.process(mcs[1].move([0.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 0.8 * (net.frame[3] - net.frame[2]) + net.frame[2]]))   
env.process(mcs[2].move([2.7 * (net.frame[1] - net.frame[0]) + net.frame[0], 2.8 * (net.frame[3] - net.frame[2]) + net.frame[2]]))    
x = env.process(net.operate(max_time=10000))
env.run(until=x)