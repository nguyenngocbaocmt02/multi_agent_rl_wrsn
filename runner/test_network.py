import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO


netIO = NetworkIO("../physical_env/network/network_scenarios/test.yaml")
env, net = netIO.makeNetwork()

x = env.process(net.operate(max_time=10000))
env.run(until=100)