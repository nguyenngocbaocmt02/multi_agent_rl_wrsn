import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO


netIO = NetworkIO("../physical_env/network/network_scenarios/test.yaml")
env, net = netIO.makeNetwork()

x = env.process(net.operate())
env.run(until=1000)