import random
import numpy as np
from scipy.optimize import minimize, basinhopping, differential_evolution
from scipy.spatial.distance import euclidean
from controller.ppo.actor.UnetActor import UNet
import matplotlib.pyplot as plt
import torch
class RandomController:
    def __init__(self):
        self.unet = UNet()

    def make_action(self, id, state, info, wrsn):
        if id == None:
            return None
        return np.copy(state[0] + state[1] - 10 * state[2] + state[3])

