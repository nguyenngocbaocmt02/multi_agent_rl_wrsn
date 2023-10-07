import os
import numpy as np
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from controller.ppo.actor.UnetActor import UNet
from controller.ppo.critic.CNNCritic import CNNCritic
from torch.utils.tensorboard import SummaryWriter
import copy
import shutil
import csv
from torchviz import make_dot

class PPO:
    def __init__(self, args, device, model_path=None):
        print(device)
        self.model_path = model_path
        self.device = device
        self.gamma = args["gamma"]
        self.clip = args["clip"]
        self.batch_size = args["batch_size"]
        self.minibatch_size = args["minibatch_size"]
        self.n_updates_per_iteration = args["n_updates_per_iteration"]
        self.save_freq = args["save_freq"]
        self.gae = args["gae"]
        self.clip_vloss = args["clip_vloss"]
        self.ent_coef = args["ent_coef"]
        self.vf_coef = args["vf_coef"]
        self.gae_lambda = args["gae_lambda"]
        self.norm_adv = args["norm_adv"]
        self.max_grad_norm = args["max_grad_norm"]
        self.actor = UNet().to(self.device)
        self.critic = CNNCritic().to(self.device)
        

        self.logger = {
            'i_so_far': 0,          # iterations so far
			't_so_far': 0,          # timesteps so far
			'ep_lens': [],       # episodic lengths in batch
			'ep_lifetime': [],       # episodic returns in batch
			'losses': [],     # losses of actor network in current iteration
            'rewards': [],
            'delta_t': time.time_ns()
		}
        if model_path is None:
            self.log_file = None
        else:
            self.critic.load_state_dict(torch.load(os.path.join(model_path, "critic.pth")))
            self.actor.load_state_dict(torch.load(os.path.join(model_path, "actor.pth")))
            self.critic.to(self.device)
            self.actor.to(self.device)
            self.log_file = os.path.join(model_path, "log.csv")
            with open(self.log_file, 'r') as file:
                reader = csv.reader(file)
                rows = list(reader)
                last_row = rows[-1] if rows else []
                self.logger['t_so_far'] = int(last_row[1])
                self.logger['i_so_far'] = int(last_row[0])

        parameters = list(self.actor.parameters()) + list(self.critic.parameters())
        # Create optimizer for combined parameters
        self.optimizer = optim.Adam(parameters, lr=args["lr"])

    def cal_rt_adv(self, states, rewards, next_states, terminals):
        with torch.no_grad():
            values = self.get_value((states))
            next_values = self.get_value(next_states)

            if self.gae:
                advantages = torch.zeros_like(rewards)
                lastgaelam = 0
                for t in reversed(range(len(rewards))): 
                    delta = rewards[t] + self.gamma * next_values[t] * int(terminals[t]) - values[t]
                    lastgaelam = delta + self.gamma * self.gae_lambda * int(terminals[t]) * lastgaelam
                    advantages[t] = lastgaelam
                returns = advantages + values
            else:
                returns = torch.zeros_like(rewards)
                for t in reversed(range(len(rewards))):
                    if t == len(rewards):
                        next_return = next_values[t]
                    else:
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + self.gamma * terminals[t] * next_return
                advantages = returns - values
            return returns, advantages, values
    
    def get_action(self, state_in):
        state = torch.FloatTensor(state_in).to(self.device)
        if state.ndim == 3:
            state = torch.unsqueeze(state, dim=0)
        mean, log_std = self.actor(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = dist.sample()
        action_log_prob = dist.log_prob(action)

        return action.detach().cpu().numpy(), action_log_prob.sum().detach().cpu().numpy()
    
    def evaluate(self, batch_states, batch_actions):
        mean, log_std = self.actor(batch_states)
        std = torch.exp(log_std)
        
        dist = Normal(mean, std)
        action_log_prob = dist.log_prob(batch_actions)
        return action_log_prob.sum((1, 2)), dist.entropy().sum((1,2))

    def get_value(self, state):
        value = self.critic(state)
        return value.sum(1)
    
    def roll_out(self, env):
        batch_states = []
        batch_actions = []
        batch_log_probs = []
        batch_next_states = []
        batch_rewards = []
        batch_returns = []
        batch_advantages = []
        batch_values = []
        
        t = 0
        while t < self.batch_size:
            states = [[] for _ in range(env.num_agent)]
            actions = [[] for _ in range(env.num_agent)]
            log_probs = [[] for _ in range(env.num_agent)]
            next_states = [[] for _ in range(env.num_agent)]
            rewards = [[] for _ in range(env.num_agent)]
            terminals = [[] for _ in range(env.num_agent)]
            request = env.reset()
            cnt = [0 for _ in range(env.num_agent)]
            log_probs_pre = [None for _ in range(env.num_agent)]
            while True:
                action, log_prob = self.get_action(request["state"])
                log_probs_pre[request["agent_id"]] = log_prob
                request = env.step(request["agent_id"], action)
                if request["terminal"]:
                    break
                if log_probs_pre[request["agent_id"]] is None:
                    continue
                t += 1
                cnt[request["agent_id"]] += 1
                states[request["agent_id"]].append(request["prev_state"])
                actions[request["agent_id"]].append(request["input_action"])
                next_states[request["agent_id"]].append(request["state"])
                rewards[request["agent_id"]].append(request["reward"])
                log_probs[request["agent_id"]].append(log_probs_pre[request["agent_id"]])
                terminals[request["agent_id"]].append(request["terminal"])

                self.writer_log_all.writerow([
                    request["agent_id"],
                    round(request["action"][0], 4),
                    round(request["action"][1], 4),
                    round(request["action"][2], 4),
                    round(request["detailed_rewards"][0], 4),
                    round(request["detailed_rewards"][1], 4),
                    round(request["detailed_rewards"][2], 4)
                ])
                self.file_log_all.flush()

            for id in range(env.num_agent):
                if len(states[id]) == 0:
                    continue
                returns, advantages, values = self.cal_rt_adv(torch.Tensor(np.array(states[id])).to(self.device), torch.Tensor(np.array(rewards[id])).to(self.device), torch.Tensor(np.array(next_states[id])).to(self.device), torch.Tensor(np.array(terminals[id])).to(self.device))
                batch_states.extend(states[id])
                batch_actions.extend(actions[id])
                batch_log_probs.extend(log_probs[id])
                batch_next_states.extend(next_states[id])
                batch_rewards.extend(rewards[id])
                batch_advantages.extend(advantages)
                batch_returns.extend(returns)
                batch_values.extend(values)
                self.logger["rewards"].append(np.array(rewards[id]))

            self.logger["ep_lens"].append(cnt)
            self.logger["ep_lifetime"].append(env.env.now)
            
        
        mean = np.mean(batch_rewards)
        # Calculate the absolute differences between elements and the 50th percentile
        abs_diff = np.abs(batch_rewards - mean)
        indices = np.argsort(abs_diff)
        selected_num = int(self.batch_size / 2.0)
        random_num = self.batch_size - selected_num
        indices = np.concatenate((indices[-selected_num:], np.random.choice(len(batch_rewards) - selected_num, size=random_num, replace=False)))

        batch_rewards = torch.FloatTensor(np.array(batch_rewards)[indices])
        batch_states = torch.FloatTensor(np.array(batch_states)[indices])
        batch_actions = torch.FloatTensor(np.array(batch_actions)[indices])
        batch_next_states = torch.FloatTensor(np.array(batch_next_states)[indices])
        batch_log_probs = torch.FloatTensor(np.array(batch_log_probs)[indices])
        batch_returns = torch.stack([batch_returns[_] for _ in indices])
        batch_advantages = torch.stack([batch_advantages[_] for _ in indices])
        batch_values = torch.stack([batch_values[_] for _ in indices])
        return batch_states.to(self.device), batch_actions.to(self.device), batch_log_probs.to(self.device), batch_rewards.to(self.device), batch_next_states.to(self.device), batch_advantages.to(self.device), batch_returns.to(self.device), batch_values.to(self.device)
    
    def train(self, env, trained_iterations, save_folder):
        self.file_log_all = open('log_cur.csv', 'w', newline='')
        self.writer_log_all = csv.writer(self.file_log_all)
        writer = SummaryWriter("runs/ppo")
        start_time = time.time()
        t_so_far = 0 
        i_so_far = 0 
        logs = []
        while i_so_far <= trained_iterations:                                           
            batch_states, batch_actions, batch_log_probs, batch_rewards, batch_next_states, batch_advantages, batch_returns, batch_values = self.roll_out(env)
            t_so_far += self.batch_size
            i_so_far += 1
            self.logger['t_so_far'] += self.batch_size
            self.logger['i_so_far'] += 1
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for _ in range(self.n_updates_per_iteration):
                np.random.shuffle(b_inds)
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]
                    newlogprob, entropy = self.evaluate(batch_states[mb_inds], batch_actions[mb_inds])
                    newvalue = torch.squeeze(self.get_value(batch_states[mb_inds]))
                    logratio = newlogprob - batch_log_probs[mb_inds]
                    ratio = logratio.exp()
                    with torch.no_grad():
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip).float().mean().item()]
                    mb_advantages = batch_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip, 1 + self.clip)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - batch_returns[mb_inds]) ** 2
                        v_clipped = batch_values[mb_inds] + torch.clamp(
                            newvalue - batch_values[mb_inds],
                            -self.clip,
                            self.clip,
                        )
                        v_loss_clipped = (v_clipped - batch_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - batch_returns[mb_inds]) ** 2).mean()
                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                    self.optimizer.zero_grad()
                    loss.backward()
                    
                    nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                

                    self.logger['losses'].append(torch.clone(loss).detach().cpu().numpy())
            y_pred, y_true = batch_values.detach().cpu().numpy(), batch_returns.detach().cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # TRY NOT TO MODIFY: record rewards for plotting purposes
            writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], i_so_far)
            writer.add_scalar("losses/value_loss", v_loss.item(), i_so_far)
            writer.add_scalar("losses/policy_loss", pg_loss.item(), i_so_far)
            writer.add_scalar("losses/entropy", entropy_loss.item(), i_so_far)
            writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), i_so_far)
            writer.add_scalar("losses/approx_kl", approx_kl.item(), i_so_far)
            writer.add_scalar("losses/clipfrac", np.mean(clipfracs), i_so_far)
            writer.add_scalar("losses/explained_variance", explained_var, i_so_far)
            print("SPS:", int(i_so_far) / (time.time() - start_time))
            writer.add_scalar("charts/SPS", int(i_so_far) / (time.time() - start_time), i_so_far)
                
            # Print a summary of our training so far
            logs.append(self._log_summary())
            # Open the file in append mode, creating it if it doesn't exist

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
                    writer_log = csv.writer(file)
                    for row in logs:
                        writer_log.writerow(row)
                logs = []
                self.log_file = os.path.join(folder, 'log.csv')
        self.file_log_all.close()

                
    def _log_summary(self):
        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['ep_lens'])
        avg_ep_lifetime = np.mean([np.sum(ep_rews) for ep_rews in self.logger['ep_lifetime']])
        avg_loss = np.mean([losses.mean() for losses in self.logger['losses']])
        avg_rew = np.mean([rewards.mean() for rewards in self.logger['rewards']])

        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9

		# Round decimal places for more aesthetic logging messages
        delta_t = str(round(delta_t, 2))
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_lifetime = str(round(avg_ep_lifetime, 2))
        avg_loss = str(round(avg_loss, 5))
        avg_rew = str(round(avg_rew, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Lifetime: {avg_ep_lifetime}", flush=True)
        print(f"Average Loss: {avg_loss}", flush=True)
        print(f"Average Reward: {avg_rew}", flush=True)
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

		# Reset batch-specific logging data
        self.logger['ep_lens'] = []
        self.logger['ep_lifetime'] = []  
        self.logger['losses'] = []
        return [i_so_far, t_so_far, avg_ep_lens, avg_ep_lifetime, avg_loss, avg_rew, delta_t]
