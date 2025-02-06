import random
import time

import torch
import torch.nn.functional as F


class Estimators:
    def __init__(self, alpha, epsilon, gamma, tau, device, no_of_actions, n_steps=2, v_min=-900, v_max=900):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_atoms = no_of_actions
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, self.num_atoms).to(device)
        self.device = device
        self.tau = tau
        self.n_steps = n_steps
        self.counter = 0

    def Q_value(self, actor, critic, target, s, a, r, s_next, a_next, done, task_indicator, ad_reward):
        #actor_actions = actor(s, task_indicator)
        a_p = []
        for i in range(0, 3):
            a_p.append(a[i])
        Q_current = critic(s, a_p, task_indicator)
        #print(target.__class__.__name__)
        with torch.no_grad():
            a_n =[]
            for i in range(0, 3):
                a_n.append(a_next[i].detach())
            Q_next = target(s_next.detach(),a_n, task_indicator.detach())
        Q_new = r.unsqueeze(-1) + ad_reward.unsqueeze(-1)
        z_t = torch.zeros_like(Q_next[0]).to(self.device)
        #print(z_t.shape)
        Q_target = [z_t,z_t,z_t]
        if self.counter > 3e4:
            id =-1
        else:
            id = random.randint(0, 2)
        self.counter += 1
        importance = 0.
        for i in range(0,3):
            if id != i:
                Q_target_updated = Q_next[i].clone()
                current_action_idx = torch.argmax(a[i], dim=-1)
                next_action_idx = torch.argmax(a_next[i], dim=-1)
                if next_action_idx.dim() <2:
                    next_action_idx = next_action_idx.unsqueeze(1)
                    current_action_idx = current_action_idx.unsqueeze(1)
                q_next = Q_target_updated.gather(1, next_action_idx)
                Q_new += self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + (1-done)*self.gamma * q_next) + done*(r.unsqueeze(-1) + ad_reward.unsqueeze(-1))
                Q_target_updated.scatter_(dim=1, index=current_action_idx, src=Q_new)
                Q_target[i] = Q_target_updated
            else:
                Q_target_updated = Q_next[i].clone()
                current_action_idx = torch.argmax(a[i], dim=-1)
                next_action_idx = torch.argmax(a_next[i], dim=-1)
                if next_action_idx.dim() < 2:
                    next_action_idx = next_action_idx.unsqueeze(1)
                    current_action_idx = current_action_idx.unsqueeze(1)
                q_next = Q_target_updated.gather(1, next_action_idx)
                Q_new += self.alpha * (
                            (r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + (1 - done) * self.gamma * q_next) + done * (
                                     r.unsqueeze(-1) + ad_reward.unsqueeze(-1))
                Q_target_updated.scatter_(dim=1, index=current_action_idx, src=Q_new)
                Q_target[i] = Q_target_updated*importance
                Q_current[i] = Q_current[i]*importance
        return Q_target, Q_current

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
