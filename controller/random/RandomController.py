import random
import numpy as np
class RandomController:
    def __init__(self):
        pass

    def make_action(self, state):
        return [random.random(), random.random(), random.random() / 20]
    

