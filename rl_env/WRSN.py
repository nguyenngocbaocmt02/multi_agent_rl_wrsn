import yaml
import copy
import gym
from gym import spaces
import numpy as np
import sys
import os
from scipy.spatial.distance import euclidean
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from physical_env.network.NetworkIO import NetworkIO
from physical_env.mc.MobileCharger import MobileCharger
from scipy.integrate import quad, dblquad

def dblfunc(x, y, hX, hY):
    sqrz = x ** 2 + y ** 2
    sqrhZ = hX ** 2 + hY ** 2 
    km = sqrz / (-2 * sqrhZ) 
    return np.exp(km)
def func(x, hX): 
    km = x ** 2 / (-2 * hX ** 2) 
    return np.exp(km)

    
class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent, map_size=100):
        self.scenario_io = NetworkIO(scenario_path)
        with open(agent_type_path, "r") as file:
            self.mc_phy_para = yaml.safe_load(file)
        self.num_agent = num_agent
        self.map_size = map_size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 100, 100,), dtype=np.float64)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64)
        self.state = None
        self.mcs_process = [None for _ in range(num_agent)]
        self.mcs_action = [None for _ in range(num_agent)]
        self.mcs_prev_state = [None for _ in range(num_agent)]
        self.mcs_prev_fitness = [None for _ in range(num_agent)]
        self.reset()

    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate())
        self.mcs = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.mc_phy_para) for _ in range(self.num_agent)]
        for id, mc in enumerate(self.mcs):
            mc.env = self.env
            mc.net = self.net
            mc.id = id
            mc.cur_phy_action = [self.net.baseStation.location[0], self.net.baseStation.location[1], 0]
        self.env.run(until=100)
        tmp_state = self.get_state()
        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True
        tmp_fitness = self.get_network_fitness()  
        for id, mc in enumerate(self.mcs):
            self.mcs_prev_state[id] = tmp_state
            self.mcs_action[id] = np.reshape(np.append(self.down_mapping(copy.deepcopy(self.net.baseStation.location)), 0), (3,))
            self.mcs_process[id] = self.env.process(self.mcs[id].operate_step(copy.deepcopy(mc.cur_phy_action)))
            self.mcs_prev_fitness[id] = tmp_fitness      

        for id, mc in enumerate(self.mcs):
            if euclidean(mc.location, mc.cur_phy_action[0:2]) < mc.epsilon and mc.cur_phy_action[2] == 0:
                return {"agent_id":id, "prev_state": self.mcs_prev_state[id],
                        "prev_fitness": self.mcs_prev_fitness[id], "action":self.mcs_action[id],
                        "state": tmp_state, "fitness":tmp_fitness,
                        "network": self.net, "mcs":self.mcs, "terminal":tmp_terminal}
        return {"agent_id":None, "prev_state": None,
        "prev_fitness": None, "action": None,
        "state": tmp_state, "fitness":tmp_fitness,
        "network": self.net, "mcs":self.mcs, "terminal":tmp_terminal}
    
            
    def down_mapping(self, location):
        return np.array([(location[0] - self.net.frame[0]) / (self.net.frame[1] - self.net.frame[0])
                        ,(location[1] - self.net.frame[2]) / (self.net.frame[3] - self.net.frame[2])])

    
    def up_mapping(self, down_map):
        return np.array([down_map[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0]
                         , down_map[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2]])
    
    def translate(self, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                (self.net.listNodes[0].capacity - self.net.listNodes[0].threshold) /                 
                (self.mc_phy_para["alpha"] / (0+ self.mc_phy_para["beta"]) ** 2) * action[2]])
    
    def get_state(self):
        map_1 = np.zeros((self.map_size, self.map_size), np.float64)
        for node in self.net.listNodes:
            coor = self.down_mapping(node.location)
            id_x_min = max(0 ,int((coor[0] - self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])) / (1.0 / self.map_size)))
            id_x_max = min(self.map_size - 1, int((coor[0] + self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])) / (1.0 / self.map_size)))
            id_y_min = max(0 ,int((coor[1] - self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])) / (1.0 / self.map_size)))
            id_y_max = min(self.map_size - 1, int((coor[1] + self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])) / (1.0 / self.map_size)))
            for i in range(id_x_min, id_x_max + 1):
                for j in range(id_y_min, id_y_max + 1):
                    x_lower = i * (1.0 / self.map_size) - coor[0]
                    x_upper = (i + 1) * (1.0 / self.map_size) - coor[0]
                    y_lower = j * (1.0 / self.map_size) - coor[1]
                    y_upper = (j + 1) * (1.0 / self.map_size) - coor[1]
                    #m = dblquad(dblfunc, x_lower, x_upper, y_lower, y_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0]),
                    #                                                                self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])))[0]
                    #map_1[i][j] += m * (node.energyCS / node.energy)
                    m1 = quad(func, x_lower, x_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])))[0]
                    m2 = quad(func, y_lower, y_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])))[0]
                    map_1[i][j] += m1 * m2 * (node.energyCS / node.energy)
        tmp_max = np.max(map_1)
        tmp_min = np.min(map_1)
        if tmp_max == tmp_min:
            map_1 = np.zeros_like(map_1)
        else:
            map_1 = (map_1 - tmp_min) / (tmp_max - tmp_min)


        map_2 = np.zeros_like(map_1)
        for mc in self.mcs:
            if mc.cur_action_type == "moving":
                continue
            coor = self.down_mapping(mc.location)
            id_x_min = max(0 ,int((coor[0] - self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])) / (1.0 / self.map_size)))
            id_x_max = min(self.map_size - 1, int((coor[0] + self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])) / (1.0 / self.map_size)))
            id_y_min = max(0 ,int((coor[1] - self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])) / (1.0 / self.map_size)))
            id_y_max = min(self.map_size - 1, int((coor[1] + self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])) / (1.0 / self.map_size)))
            for i in range(id_x_min, id_x_max + 1):
                for j in range(id_y_min, id_y_max + 1):
                    x_lower = i * (1.0 / self.map_size) - coor[0]
                    x_upper = (i + 1) * (1.0 / self.map_size) - coor[0]
                    y_lower = j * (1.0 / self.map_size) - coor[1]
                    y_upper = (j + 1) * (1.0 / self.map_size) - coor[1]
                    #m = dblquad(dblfunc, x_lower, x_upper, y_lower, y_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0]),
                    #                                                                self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])))[0]
                    #map_2[i][j] += m * mc.cur_phy_action[2] 
                    m1 = quad(func, x_lower, x_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])))[0]
                    m2 = quad(func, y_lower, y_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])))[0]
                    map_2[i][j] += m1 * m2 * mc.cur_phy_action[2]       
        tmp_max = np.max(map_2)
        tmp_min = np.min(map_2)
        if tmp_max == tmp_min:
            map_2 = np.zeros_like(map_2)
        else:
            map_2 = (map_2 - tmp_min) / (tmp_max - tmp_min)

        map_3 = np.zeros_like(map_1)
        for mc in self.mcs:
            if mc.cur_action_type == "charging":
                continue
            coor = self.down_mapping(mc.location)
            id_x_min = max(0 ,int((coor[0] - self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])) / (1.0 / self.map_size)))
            id_x_max = min(self.map_size - 1, int((coor[0] + self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])) / (1.0 / self.map_size)))
            id_y_min = max(0 ,int((coor[1] - self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])) / (1.0 / self.map_size)))
            id_y_max = min(self.map_size - 1, int((coor[1] + self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])) / (1.0 / self.map_size)))
            for i in range(id_x_min, id_x_max + 1):
                for j in range(id_y_min, id_y_max + 1):
                    x_lower = i * (1.0 / self.map_size) - coor[0]
                    x_upper = (i + 1) * (1.0 / self.map_size) - coor[0]
                    y_lower = j * (1.0 / self.map_size) - coor[1]
                    y_upper = (j + 1) * (1.0 / self.map_size) - coor[1]
                    #m = dblquad(dblfunc, x_lower, x_upper, y_lower, y_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0]),
                    #                                                                self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])))[0]
                    #map_3[i][j] += m / (1 + euclidean(mc.location, np.array([mc.cur_phy_action[0], mc.cur_phy_action[1]])) / (mc.velocity)) 
                    m1 = quad(func, x_lower, x_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[1] - self.net.frame[0])))[0]
                    m2 = quad(func, y_lower, y_upper, args=(self.mc_phy_para["charging_range"] / (self.net.frame[3] - self.net.frame[2])))[0]
                    map_3[i][j] += m1 * m2 / (1 + euclidean(mc.location, np.array([mc.cur_phy_action[0], mc.cur_phy_action[1]])) / (mc.velocity)) 
        tmp_max = np.max(map_3)
        tmp_min = np.min(map_3)
        if tmp_max == tmp_min:
            map_3 = np.zeros_like(map_3)
        else:
            map_3 = (map_3 - tmp_min) / (tmp_max - tmp_min)
        self.state = np.stack((map_1, map_2, map_3))
        return self.state
    
    def get_network_fitness(self):
        node_t = [-1 for node in self.net.listNodes]
        tmp1 = []
        tmp2 = []
        for node in self.net.baseStation.direct_nodes:
            if node.status == 1:
                tmp1.append(node)
                if node.energyCS == 0:
                    node_t[node.id] = float("inf")
                else:
                    node_t[node.id] = (node.energy - node.threshold) / (node.energyCS)
        while True:
            if len(tmp1) == 0:
                break
            # For each node, we set value of target covered by this node as 1
            # For each node, if we have not yet reached its neighbor, then level of neighbors equal this node + 1
            for node in tmp1:
                for neighbor in node.neighbors:
                    if neighbor.status != 1:
                        continue
                    if neighbor.energyCS == 0:
                        neighborLT = float("inf")
                    else:
                        neighborLT = (neighbor.energy - neighbor.threshold) / (neighbor.energyCS)
                    if  node_t[neighbor.id] == -1 or (node_t[node.id] > node_t[neighbor.id] and neighborLT > node_t[neighbor.id]):
                        tmp2.append(neighbor)
                        node_t[neighbor.id] = min(neighborLT, node_t[node.id])

            # Once all nodes at current level have been expanded, move to the new list of next level
            tmp1 = tmp2[:]
            tmp2.clear()
        target_t = [0 for target in self.net.listTargets]
        for node in self.net.listNodes:
            for target in node.listTargets:
                target_t[target.id] = max(target_t[target.id], node_t[node.id])
        return min(target_t)
    
    def step(self, agent_id, input_action):
        if agent_id is not None:
            action = copy.deepcopy(input_action)
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.mcs_process[agent_id] = self.env.process(self.mcs[agent_id].operate_step(self.translate(action)))
            self.mcs_action[agent_id] = action
            self.mcs_prev_state[agent_id] = self.get_state()
            self.mcs_prev_fitness[agent_id] = self.get_network_fitness()

        general_process = self.net_process
        for id, mc in enumerate(self.mcs):
            general_process = general_process | self.mcs_process[id]
        self.env.run(until=general_process)
        if self.net.alive == 0:
            return {"agent_id":None, "prev_state": None,
                        "prev_fitness":None, "action":None,
                        "state": self.get_state(), "fitness":self.get_network_fitness(),
                        "network": self.net, "mcs":self.mcs, "terminal":True}
        for id, mc in enumerate(self.mcs):
            if euclidean(mc.location, mc.cur_phy_action[0:2]) < mc.epsilon and mc.cur_phy_action[2] == 0:
                return {"agent_id":id, "prev_state": self.mcs_prev_state[id],
                        "prev_fitness":self.mcs_prev_fitness[id], "action":self.mcs_action[id],
                        "state": self.get_state(), "fitness":self.get_network_fitness(),
                        "network": self.net, "mcs":self.mcs, "terminal":False}