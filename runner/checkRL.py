import yaml
import sys
import os
import random
from simpy.events import AnyOf
import copy
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from controller.bao.BaoController import BaoController
from controller.random.RandomController import RandomController
from rl_env.WRSN import WRSN
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
import pandas as pd

network = WRSN(scenario_path="../physical_env/network/network_scenarios/hanoi1000n50.yaml"
               ,mc_type_path="../physical_env/mc/mc_types/default.yaml"
               ,num_mc=3, map_size=100)
hi = network.get_state()
df = pd.DataFrame(hi[0])
sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
plt.show()