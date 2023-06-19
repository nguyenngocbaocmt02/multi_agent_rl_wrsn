import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO


def log(net, mcs):
    # If you want to print something, just put it here. Do not fix the core code.
    while True:
        print(net.env.now, net.check_nodes())
        yield net.env.timeout(1.0)

netIO = NetworkIO("physical_env/network/network_scenarios/test_50.yaml")
env, net = netIO.makeNetwork()
x = env.process(net.operate())
env.process(log(net, None))
env.run(until=x)