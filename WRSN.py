import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from environments.network.NetworkIO import NetworkIO
from environments.mc.MobileCharger import MobileCharger
import yaml
import copy
class WRSN:
    def __init__(self, network_path, mc_config_path, mc_num):
        self.netIO = NetworkIO(network_path)
        with open(mc_config_path, 'r') as file:
            self.mc_argc = yaml.safe_load(file)
        self.mc_config_path = mc_config_path
        self.mc_num = mc_num
    
    def reset(self):
        self.env, self.net = self.netIO.makeNetwork()
        self.mcs = [MobileCharger(copy.deepcopy(self.net.baseStation.location), mc_phy_spe=copy.deepcopy(self.mc_argc)) for _ in range(self.mc_num)]

        for id, mc in enumerate(self.mcs):
            mc.env = self.env
            mc.net = self.net
            mc.id = id
        tmp = self.env.process(self.net.operate(max_time=50))
        self.env.run(until=tmp)    

    def step(actions):
        for i, action in enumerate(actions):
            if action is None:
                continue
            else:
            
