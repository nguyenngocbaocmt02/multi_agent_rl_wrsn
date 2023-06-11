import os
import numpy as np
import torch
import torch.nn as nn
import time
import random
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from controller.ppo.actor.CNNActor import CNNActor
from controller.ppo.critic.CNNCritic import CNNCritic
import copy
import shutil
import csv

class PPO:
    def __init__(self, args, model_path=None):
        self.model_path = model_path
        self.gamma = args["gamma"]
        self.clip = args["clip"]
        self.batch_size = args["batch_size"]
        self.n_updates_per_iteration = args["n_updates_per_iteration"]
        self.save_freq = args["save_freq"]
        self.actor = CNNActor()
        self.critic = CNNCritic()
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=args["lr_actor"])
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=args["lr_critic"])
        self.logger = {
            'i_so_far': 0,          # iterations so far
			't_so_far': 0,          # timesteps so far
			'ep_lens': [],       # episodic lengths in batch
			'ep_lifetime': [],       # episodic returns in batch
			'actor_losses': [],     # losses of actor network in current iteration
            'critic_losses': [],
            'rewards': [],
            'delta_t': time.time_ns()
		}
        if model_path is None:
            self.actor.apply(lambda x: nn.init.xavier_uniform_(x.weight) if type(x) == nn.Conv2d or type(x) == nn.Linear else None)
            self.critic.apply(lambda x: nn.init.xavier_uniform_(x.weight) if type(x) == nn.Conv2d or type(x) == nn.Linear else None)
            self.log_file = None
        else:
            self.critic.load_state_dict(torch.load(os.path.join(model_path, "critic.pth")))
            self.actor.load_state_dict(torch.load(os.path.join(model_path, "actor.pth")))
            self.log_file = os.path.join(model_path, "log.csv")
            with open(self.log_file, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                last_row = rows[-1] if rows else []
                self.logger['t_so_far'] = int(last_row[1])
                self.logger['i_so_far'] = int(last_row[0])

    def get_action(self, state):
        state = torch.FloatTensor(state)
        if state.ndim == 3:
            state = torch.unsqueeze(state, dim=0)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action = dist.sample()
        action_log_prob = dist.log_prob(action)
        action = torch.squeeze(action)
        return action.detach().numpy(), action_log_prob.detach().numpy()
    
    def evaluate(self, batch_states, batch_actions):
        mean, log_std = self.actor(batch_states)
        std = log_std.exp()
        dist = MultivariateNormal(mean, torch.diag_embed(std))
        action_log_prob = dist.log_prob(batch_actions)
        return action_log_prob

    def get_value(self, state):
        state = torch.FloatTensor(state)
        value = self.critic(state)
        return value
    
    def roll_out(self, env, max_time):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_next_states = []
        batch_rewards = []
        
        t = 0
        while t < self.batch_size:
            request = env.reset()
            cnt = 0
            log_probs = [None for _ in range(env.num_agent)]
            while env.env.now < max_time:
                action, log_prob = self.get_action(request["state"])
                log_probs[request["agent_id"]] = log_prob
                request = env.step(request["agent_id"], action)
                if request["terminal"]:
                    break
                if log_probs[request["agent_id"]] is None:
                    continue
                t += 1
                cnt += 1
                batch_states.append(request["prev_state"])
                batch_actions.append(request["action"])
                batch_next_states.append(request["state"])
                batch_rewards.append((request["fitness"] - request["prev_fitness"]) / request["prev_fitness"])
                batch_log_probs.append(log_probs[request["agent_id"]])
                
            self.logger["ep_lens"].append(cnt)
            self.logger["ep_lifetime"].append(env.env.now)
            self.logger["rewards"].append(copy.deepcopy(batch_rewards))
            
        random_indices = np.random.choice(len(batch_rewards), size=self.batch_size, replace=False)
        batch_rewards = torch.FloatTensor(np.array(batch_rewards)[random_indices])
        batch_states = torch.FloatTensor(np.array(batch_states)[random_indices])
        batch_actions = torch.FloatTensor(np.array(batch_actions)[random_indices])
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)[random_indices])
        batch_log_probs = torch.FloatTensor(np.array(batch_log_probs)[random_indices])

        return batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states
    
    def train(self, env, max_time, trained_iterations, save_folder):
        t_so_far = 0 
        i_so_far = 0 
        logs = []
        while i_so_far <= trained_iterations:                                           
            batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states = self.roll_out(env, max_time)
            t_so_far += self.batch_size
            i_so_far += 1
            self.logger['t_so_far'] += self.batch_size
            self.logger['i_so_far'] += 1
            
            for _ in range(self.n_updates_per_iteration):
                V_est2 = torch.squeeze(self.get_value(batch_states).detach())
                V_est1 = batch_rewards + self.gamma * torch.squeeze(self.get_value(batch_next_states).detach())
                
                A_k =  V_est1 - V_est2       
                A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)
                curr_log_probs = self.evaluate(batch_states, batch_actions)

                ratios = torch.exp(curr_log_probs - batch_log_probs)
                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k
                actor_loss = (-torch.min(surr1, surr2)).mean()
                V_est1.requires_grad_()
                V_est2.requires_grad_()
                critic_loss = nn.MSELoss()(V_est1, V_est2)
                self.optimizer_actor.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.optimizer_actor.step()
                self.optimizer_critic.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.optimizer_critic.step()
                self.logger['actor_losses'].append(actor_loss.detach())
                self.logger['critic_losses'].append(critic_loss.detach())
            # Print a summary of our training so far
            logs.append(self._log_summary())
            # Save our model if it's time
            if self.logger['i_so_far'] % self.save_freq == 0:
                folder = os.path.join(save_folder, str(self.logger['i_so_far']))
                if not os.path.exists(folder):
                    os.makedirs(folder)
                torch.save(self.actor.state_dict(), os.path.join(folder, 'actor.pth'))
                torch.save(self.critic.state_dict(), os.path.join(folder, 'critic.pth'))
                if self.log_file is not None:
                    shutil.copy(self.log_file, folder)
                with open(os.path.join(folder, 'log.csv'), 'a', newline='') as file:
                    writer = csv.writer(file)
                    for row in logs:
                        writer.writerow(row)
                logs = []
                self.log_file = os.path.join(folder, 'log.csv')

                
    def _log_summary(self):
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['ep_lens'])
        avg_ep_lifetime = np.mean([np.sum(ep_rews) for ep_rews in self.logger['ep_lifetime']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])
        avg_critic_loss = np.mean([losses.float().mean() for losses in self.logger['critic_losses']])

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9

		# Round decimal places for more aesthetic logging messages
        delta_t = str(round(delta_t, 2))
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_lifetime = str(round(avg_ep_lifetime, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))
        avg_critic_loss = str(round(avg_critic_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Lifetime: {avg_ep_lifetime}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['ep_lens'] = []
        self.logger['ep_lifetime'] = []  
        self.logger['actor_losses'] = []
        self.logger['critic_losses'] = []
        return [i_so_far, t_so_far, avg_ep_lens, avg_ep_lifetime, avg_actor_loss, avg_critic_loss, delta_t]
