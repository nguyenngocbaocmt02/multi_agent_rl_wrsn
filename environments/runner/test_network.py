import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from environments.network.NetworkIO import NetworkIO

netIO = NetworkIO("environments/network/network_samples/test.yaml")
env, net = netIO.makeNetwork()
net.listNodes[0].location[0] = 1000
print(net.listNodes[0].location, net.listNodes[1].location)
x = env.process(net.operate(max_time=10000))
env.run(until=x)