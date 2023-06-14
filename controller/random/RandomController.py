import random
import numpy as np
class RandomController:
    def __init__(self):
        pass

    def make_action(self, state):
        max_index = np.unravel_index(np.argmax(state[0]), state[0].shape)
        return [1 / 100 * max_index[1] + 1/200, 1 / 200 + 1 / 100 * max_index[0],random.random() / 4]
    

