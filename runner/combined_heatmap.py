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
    return 1000000 * np.exp(-(((x-a)**2 + (y-b)**2)+0.01)**(1/20))


def second_heatmap_function(x,y, rct, cr, a, b):
        return rct*0.01/((((x-a)**2 + (y-b)**2)**(1/10)))


def third_heatmap_function(x,y,destination, location, velocity, cr):
    esp = 0.1
    y_0,x_0 = destination
    a,b = location
    dis_to_des = ((x_0-a)**2 + (y_0-b)**2)**(1/2) + esp
    dis_from_des_to_point = ((x-x_0)**2 + (y-y_0)**2)**(1/2) + esp
    return velocity/(dis_to_des*dis_from_des_to_point)


netIO = NetworkIO("C:/GIT/GIT_WRSN/multi_agent_rl_wrsn/physical_env/network/network_scenarios/test_50.yaml")
env, net = netIO.makeNetwork()
with open("C:/GIT/multi_agent_rl_wrsn/environments/mc/mc_samples/default.yaml", 'r') as file:
    mc_argc = yaml.safe_load(file)
mcs = [MobileCharger(copy.deepcopy(net.baseStation.location), mc_phy_spe=mc_argc) for _ in range(10)]
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

#Cakculating part
first_heatmap_encode = np.zeros((size,size), dtype = np.float64)
first_heatmap_encode_precalculate = np.zeros((len(net.listNodes),size,size), dtype = np.float64)
amount = np.zeros((size+1,size+1))

for id,node in tqdm(enumerate(net.listNodes)):
    tmp_res = np.zeros((size, size), dtype=np.float32)
    node_location_id_x = np.searchsorted(x_grid, node.location[0], side="right") - 1
    node_location_id_y = np.searchsorted(y_grid, node.location[1], side='right') - 1
    amount[node_location_id_y][node_location_id_x] += 1
    left_id_x, right_id_x = max(0, node_location_id_x - 2), min(size, node_location_id_x + 3)
    left_id_y, right_id_y = max(0, node_location_id_y - 2), min(size, node_location_id_y + 3)
    for i in range(left_id_y, right_id_y):
        for j in range(left_id_x, right_id_x):
            x_lower_bound = i*slot_size
            x_upper_bound = (i+1)*slot_size
            y_lower_bound = j*slot_size
            y_upper_bound = (j+1)*slot_size
            tmp_res[i][j],_ = dblquad(first_heatmap_function, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, args = (node.location[1], node.location[0]))
    first_heatmap_encode_precalculate[id] = tmp_res

#Saving Part
first_heatmap_encode_precalculate = np.array(first_heatmap_encode_precalculate)
np.save("C:/GIT/GIT_WRSN/multi_agent_rl_wrsn/additional_files/first_heatmap.npy", first_heatmap_encode_precalculate)

#Loading part
first_heatmap_encode_precalculate = np.load("C:/GIT/GIT_WRSN/multi_agent_rl_wrsn/additional_files/first_heatmap.npy")
first_heatmap_encode_precalculate = np.transpose(first_heatmap_encode_precalculate, (1, 2, 0))
first_heatmap_encode = np.zeros((size,size), dtype = np.float64)
sensors_energy = np.zeros((len(net.listNodes)))

#Getting sensors' energy
for id, sensor in enumerate(net.listNodes):
    sensors_energy[id] = net.listNodes[id].energy


for i in range(size):
    for j in range(size):
        first_heatmap_encode[i][j] = np.sum(np.divide(first_heatmap_encode_precalculate[i][j], sensors_energy))




df = pd.DataFrame(first_heatmap_encode)
sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
plt.show()

df = pd.DataFrame(amount)
sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
plt.show()







#2nd heatmap
mcs[0].location = [100,900]
mcs[1].location = [100,700]
mcs[2].location = [100,500]
second_heatmap_encode = np.zeros((size, size))
for id, mc in enumerate(mcs):
    print("AT M", id, "======================================")
    if mc.chargingTime != 0:
        mc_location_id_x = np.searchsorted(x_grid, mc.location[0], side="right") - 1
        mc_location_id_y = np.searchsorted(y_grid, mc.location[1], side='right') - 1

        left_id_x, right_id_x = max(0, mc_location_id_x - 2), min(size, mc_location_id_x + 3)
        left_id_y, right_id_y = max(0, mc_location_id_y - 2), min(size, mc_location_id_y + 3)
        print("CHARGING")
        print(mc_location_id_x,mc_location_id_y)

        # print(id, mc.chargingTime)
        for i in range(left_id_y,right_id_y):
            # print("DONE AT I =", i)
            for j in range(left_id_x,right_id_x):
                # print ("DONE AT I, J", i, j)
                x_upper_bound = i* slot_size
                x_lower_bound = (i+1) * slot_size
                y_upper_bound = j * slot_size
                y_lower_bound = (j+1) * slot_size

                value, _ = dblquad(second_heatmap_function, x_lower_bound, x_upper_bound, y_lower_bound, y_upper_bound, args =(mc.chargingTime, mc.chargingRange, mc.location[0], mc.location[1]))
                second_heatmap_encode[i][j] += value

df = pd.DataFrame(second_heatmap_encode)
sns_plot = sns.heatmap(df, annot=False, fmt="d", cmap="YlGnBu")
plt.show()

end = time.time()


# Third heatmap

# Moving several MC away
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