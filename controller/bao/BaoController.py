import random
import numpy as np
import torch.nn as nn
import torch
from controller.bao.actor.CNNActor import CNNActor
from controller.bao.critic.CNNCritic import CNNCritic

class BaoController:
    def __init__(self, net, mcs):
        self.net = net
        self.mcs = mcs
        self.max_charging_time = (self.net.listNodes[0].capacity - self.net.listNodes[0].threshold) /  (mcs[0].alpha / (mcs[0].chargingRange + mcs[0].beta) ** 2)
        self.actors = [CNNActor() for _ in mcs]
        self.critic = CNNCritic()

        for actor in self.actors:
            actor.apply(lambda x: nn.init.xavier_uniform_(x.weight) if type(x) == nn.Conv2d or type(x) == nn.Linear else None)
        self.critic.apply(lambda x: nn.init.xavier_uniform_(x.weight) if type(x) == nn.Conv2d or type(x) == nn.Linear else None)

    def make_action(self, id):
        input = torch.randn(1, 3, 100, 100)
        output = self.actors[id](input).squeeze().detach().numpy()
        return self.translate(output)
    
    
    def translate(self, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                self.max_charging_time * action[2]])
