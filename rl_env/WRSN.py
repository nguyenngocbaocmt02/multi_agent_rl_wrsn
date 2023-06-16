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
import matplotlib.pyplot as plt

def func(x, hX): 
    km = x ** 2 / (-2 * hX ** 2) 
    return np.exp(km)

    
class WRSN(gym.Env):
    def __init__(self, scenario_path, agent_type_path, num_agent, map_size=100):
        self.scenario_io = NetworkIO(scenario_path)
        with open(agent_type_path, "r") as file:
            self.agent_phy_para = yaml.safe_load(file)
        self.num_agent = num_agent
        self.map_size = map_size
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(3, 100, 100,), dtype=np.float64)
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(3,), dtype=np.float64)
        self.agents_process = [None for _ in range(num_agent)]
        self.agents_action = [None for _ in range(num_agent)]
        self.agents_prev_state = [None for _ in range(num_agent)]
        self.agents_prev_fitness = [None for _ in range(num_agent)]
        self.agents_exclusive_reward = [0 for _ in range(num_agent)]
        self.reset()

    def reset(self):
        self.env, self.net = self.scenario_io.makeNetwork()
        self.net_process = self.env.process(self.net.operate())
        self.reward_update_process = self.env.process(self.update_reward())
        self.agents = [MobileCharger(copy.deepcopy(self.net.baseStation.location), self.agent_phy_para) for _ in range(self.num_agent)]
        for id, agent in enumerate(self.agents):
            agent.env = self.env
            agent.net = self.net
            agent.id = id
            agent.cur_phy_action = [self.net.baseStation.location[0], self.net.baseStation.location[1], 0]
        self.env.run(until=100)
        if self.net.alive == 1:
            tmp_terminal = False
        else:
            tmp_terminal = True
        tmp_fitness = self.get_network_fitness()  
        for id, agent in enumerate(self.agents):
            self.agents_prev_state[id] = self.get_state(agent.id)
            self.agents_action[id] = np.reshape(np.append(self.down_mapping(copy.deepcopy(self.net.baseStation.location)), 0), (3,))
            self.agents_process[id] = self.env.process(self.agents[id].operate_step(copy.deepcopy(agent.cur_phy_action)))
            self.agents_prev_fitness[id] = tmp_fitness   
            self.agents_exclusive_reward[id] = 0.0  

        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < agent.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id":id, 
                        "prev_state": self.agents_prev_state[id],
                        "action":self.agents_action[id], 
                        "reward": 0.0,
                        "state": self.agents_prev_state[id],
                        "terminal":tmp_terminal,
                        "info": [self.net, self.agents]}
        return {"agent_id":None, 
                "prev_state": None,
                "action": None,
                "reward": None,
                "state": None,
                "terminal":tmp_terminal,
                "info": [self.net, self.agents]}
    
            
    def down_mapping(self, location):
        return np.array([(location[0] - self.net.frame[0]) / (self.net.frame[1] - self.net.frame[0])
                        ,(location[1] - self.net.frame[2]) / (self.net.frame[3] - self.net.frame[2])])

    
    def up_mapping(self, down_map):
        return np.array([down_map[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0]
                         , down_map[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2]])
    
    def translate(self, agent_id, action):
        return np.array([action[0] * (self.net.frame[1] - self.net.frame[0]) + self.net.frame[0],
                action[1] * (self.net.frame[3] - self.net.frame[2]) + self.net.frame[2],
                (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe["threshold"]) /                 
                (self.agents[agent_id].alpha / (self.agents[agent_id].beta) ** 2) * action[2]])
    
    def update_reward(self):
        while True:
            for agent in self.agents:
                if agent.status == 0:
                    continue
                upper_time = (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe["threshold"]) \
                 / (agent.alpha / (agent.beta ** 2))
                if agent.cur_action_type == "charging":
                    incentive = 0
                    for node in agent.connected_nodes:
                        if node.status == 1:
                            cs_term = node.energyCS / (agent.alpha / (agent.beta ** 2))
                            charge_term = (agent.beta ** 2) / ((euclidean(node.location, agent.location) + agent.beta) ** 2) / (upper_time) 
                            incentive +=  cs_term * charge_term
                    self.agents_exclusive_reward[agent.id] += incentive
            yield self.env.timeout(1.0)
            
    
    def get_state(self, agent_id):
        agent = self.agents[agent_id]
        x = np.arange(0.0, 1.0,  1 / self.map_size)
        y = np.arange(0.0, 1.0,  1 / self.map_size)
        xx, yy = np.meshgrid(x, y)
        
        map_1 = np.zeros_like(xx)
        for node in self.net.listNodes:
            if node.status == 0: 
                continue
            coor = self.down_mapping(node.location)
            xx_coor = xx - coor[0]
            yy_coor = yy - coor[1]
            hX = agent.chargingRange / (self.net.frame[1] - self.net.frame[0])
            hY = agent.chargingRange / (self.net.frame[3] - self.net.frame[2])
            pdf = ((node.energyCS / (agent.alpha / (agent.beta ** 2))) / ((node.energy - node.threshold) / (node.capacity - node.threshold))) * func(xx_coor, hX) * func(yy_coor, hY)
            map_1 += pdf

        map_2 = np.zeros_like(xx)
        coor = self.down_mapping(agent.location)
        xx_coor = xx - coor[0]
        yy_coor = yy - coor[1]
        tmp = min((self.net.frame[3] - self.net.frame[2]), (self.net.frame[1] - self.net.frame[0]))
        hX =  0.5 * tmp / (self.net.frame[1] - self.net.frame[0])
        hY =  0.5 * tmp / (self.net.frame[3] - self.net.frame[2])
        pdf = (agent.energy / agent.capacity) * func(xx_coor, hX) * func(yy_coor, hY)
        map_2 += pdf

        map_3 = np.zeros_like(xx)
        for another in self.agents:
            if another.id == agent.id:
                continue
            if another.cur_action_type == "moving":
                continue
            coor = self.down_mapping([another.cur_phy_action[0], another.cur_phy_action[1]])
            xx_coor = xx - coor[0]
            yy_coor = yy - coor[1]
            hX = another.chargingRange / (self.net.frame[1] - self.net.frame[0])
            hY = another.chargingRange / (self.net.frame[3] - self.net.frame[2])
            upper_time = (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe["threshold"]) \
            / (another.alpha / (another.beta ** 2))
            pdf = (another.cur_phy_action[2] / upper_time) * func(xx_coor, hX) * func(yy_coor, hY)
            map_3 += pdf

        map_4 = np.zeros_like(xx)
        for another in self.agents:
            if another.id == agent.id:
                continue
            if another.cur_action_type == "charging":
                continue
            coor = self.down_mapping([another.cur_phy_action[0], another.cur_phy_action[1]])
            xx_coor = xx - coor[0]
            yy_coor = yy - coor[1]
            hX = another.chargingRange / (self.net.frame[1] - self.net.frame[0])
            hY = another.chargingRange / (self.net.frame[3] - self.net.frame[2])
            upper_time = (self.scenario_io.node_phy_spe["capacity"] - self.scenario_io.node_phy_spe["threshold"]) \
            / (another.alpha / (another.beta ** 2))
            pdf = func(xx_coor, hX) * func(yy_coor, hY) * (euclidean(another.location, np.array([another.cur_phy_action[0], agent.cur_phy_action[1]])) / another.velocity) / upper_time
            map_4 += pdf
        return np.stack((map_1, map_2, map_3, map_4))
    
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

            tmp1 = tmp2[:]
            tmp2.clear()
        target_t = [0 for target in self.net.listTargets]
        for node in self.net.listNodes:
            for target in node.listTargets:
                target_t[target.id] = max(target_t[target.id], node_t[node.id])
        return min(target_t)
    
    def get_reward(self, agent_id):
        return self.agents_exclusive_reward[agent_id] + (self.get_network_fitness() / self.agents_prev_fitness[agent_id] - 1)
    
    def step(self, agent_id, input_action):
        
        if agent_id is not None:
            action = np.array(input_action).copy()
            action = np.clip(action, self.action_space.low, self.action_space.high)
            self.agents_process[agent_id] = self.env.process(self.agents[agent_id].operate_step(self.translate(agent_id, action)))
            self.agents_action[agent_id] = action
            self.agents_prev_state[agent_id] = self.get_state(agent_id)
            self.agents_prev_fitness[agent_id] = self.get_network_fitness()
            self.agents_exclusive_reward[agent_id] = 0

        general_process = self.net_process
        for id, agent in enumerate(self.agents):
            if agent.status != 0:
                general_process = general_process | self.agents_process[id]
        self.env.run(until=general_process)
        if self.net.alive == 0:
            return {"agent_id":None, 
                    "prev_state": None,
                    "action":None, 
                    "reward": None,
                    "state": None,
                    "terminal":True,
                    "info": [self.net, self.agents]}
        for id, agent in enumerate(self.agents):
            if euclidean(agent.location, agent.cur_phy_action[0:2]) < agent.epsilon and agent.cur_phy_action[2] == 0:
                return {"agent_id": id, 
                        "prev_state": self.agents_prev_state[id],
                        "action":self.agents_action[id], 
                        "reward": self.get_reward(id),
                        "state": self.get_state(id), 
                        "terminal": False,
                        "info": [self.net, self.agents]}
