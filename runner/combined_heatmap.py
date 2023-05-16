import sys
import os
import copy
import yaml
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from multi_agent_rl_wrsn.physical_env.network.NetworkIO import NetworkIO
from multi_agent_rl_wrsn.physical_env.mc.MobileCharger import MobileCharger
from tqdm import tqdm
from scipy.integrate import dblquad
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import time

def first_heatmap_function(x,y, a, b):
    return np.exp(-((x-a)**2 + (y-b)**2))


def second_heatmap_function(x,y, rct, cr, a, b):
        return rct*0.0001/((((x-a)**2 + (y-b)**2)**(1/2)))


def third_heatmap_function(x,y,destination, location, velocity, cr):
    esp = 0.1
    y_0,x_0 = destination
    a,b = location
    dis_to_des = ((x_0-a)**2 + (y_0-b)**2)**(1/2) + esp
    dis_from_des_to_point = ((x-x_0)**2 + (y-y_0)**2)**(1/2) + esp
    return velocity/(dis_to_des*dis_from_des_to_point)


netIO = NetworkIO("C:/GIT/multi_agent_rl_wrsn/environments/network/network_samples/test.yaml")
env, net = netIO.makeNetwork()

with open("C:/GIT/multi_agent_rl_wrsn/environments/mc/mc_samples/default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(10)]

start = time.time()
for id, mc in enumerate(mcs):
    mc.env = env
    mc.net = net
    mc.id = id

for i in range(5):
    env.process(mcs[i].charge(200))
env.run(until=100)
print(mcs[1].chargingTime)
size = 100
slot_size = 1000/ size

x_grid = np.linspace(0, 1000, num = size +1)
y_grid = np.linspace(0, 1000, num = size +1)


start = time.time()
# 1st heatmap
first_heatmap_encode_precalculate = np.load("C:/GIT/multi_agent_rl_wrsn/addtional_files/first_heatmap.npy")
first_heatmap_encode_precalculate = np.transpose(first_heatmap_encode_precalculate, (1, 2, 0))
first_heatmap_encode = np.zeros((size,size), dtype = np.float64)
sensors_energy = np.zeros((len(net.listNodes)))
for id, sensor in enumerate(net.listNodes):
    sensors_energy[id] = net.listNodes[id].energy


for i in range(size):
    for j in range(size):
        first_heatmap_encode[i][j] = np.sum(np.divide(first_heatmap_encode_precalculate[i][j], sensors_energy))


# df = pd.DataFrame(first_heatmap_encode)
# sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
# plt.show()




#2nd heatmap
second_heatmap_encode = np.zeros((size, size))
for id, mc in enumerate(mcs):
    print("AT M", id, "======================================")
    if mc.chargingTime != 0:
        print("CHARGING")
        # print(id, mc.chargingTime)
        for i in range(45,55):
            # print("DONE AT I =", i)
            for j in range(45,55):
                # print ("DONE AT I, J", i, j)
                x_upper_bound = i* slot_size
                x_lower_bound = (i+1) * slot_size
                y_upper_bound = j * slot_size
                y_lower_bound = (j+1) * slot_size

                value, _ = dblquad(second_heatmap_function, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, args =(mc.chargingTime, mc.chargingRange, mc.location[0], mc.location[1]))
                second_heatmap_encode[i][j] += value

# df = pd.DataFrame(second_heatmap_encode)
# sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
# plt.show()

end = time.time()


# Third heatmap

env.process(mcs[6].move([500, 900]))
env.process(mcs[7].move([900, 100]))
env.process(mcs[8].move([900, 500]))
env.process(mcs[9].move([900, 900]))
env.run(until = 101)
third_heatmap_encode = np.zeros((size,size), dtype = np.float64)
tmp_res = np.zeros((size,size), dtype = np.float64)
start = time.time()
for id, mc in enumerate(mcs):
    if mc.movingTime != 0:
        tmp_res = np.zeros((size, size), dtype=np.float64)
        if mc.destination[0] > 1000 or mc.destination[1] > 1000:
            continue
        print("MC", id,'=================================================================================================')
        mc_destination_id_x = np.searchsorted(x_grid, mc.destination[0], side = "right") -1
        mc_destination_id_y = np.searchsorted(y_grid, mc.destination[1], side = 'right') -1
        left_id_x, right_id_x = max(0, mc_destination_id_x-7), min(size, mc_destination_id_x + 9)
        left_id_y, right_id_y = max(0, mc_destination_id_y-7), min(size, mc_destination_id_y + 9)

        for i in range(left_id_x, right_id_x):
            for j in range(left_id_y, right_id_y):
                x_lower_bound = i*slot_size
                x_upper_bound = (i+1)*slot_size
                y_lower_bound = j * slot_size
                y_upper_bound = (j + 1) * slot_size
                integral_grid = np.searchsorted(x_grid, [x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound])
                tmp_res[i][j],_ = dblquad(third_heatmap_function, x_lower_bound,x_upper_bound,y_lower_bound,y_upper_bound, args = (mc.destination, mc.location, mc.velocity, mc.chargingRange))



        third_heatmap_encode += tmp_res

# df = pd.DataFrame(third_heatmap_encode)
# sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
# plt.show()

input_heatmap = np.zeros((size,size), dtype = np.float64)
a_norm = np.interp(first_heatmap_encode, (first_heatmap_encode.min(), first_heatmap_encode.max()), (0, 1))
b_norm = np.interp(second_heatmap_encode, (second_heatmap_encode.min(), second_heatmap_encode.max()), (0, 1))
c_norm = np.interp(third_heatmap_encode, (third_heatmap_encode.min(), third_heatmap_encode.max()), (0, 1))
input_heatmap = a_norm + b_norm - c_norm


df = pd.DataFrame(input_heatmap)
sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")

end = time.time()
print(end-start)
plt.show()