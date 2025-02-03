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

    def Q_value(self, actor, critic, target, s, a, r, s_next, a_next, game_over, task_indicator, ad_reward):
        Q_current = actor(s, task_indicator)
        critic_eval = critic(s, Q_current, task_indicator)
        with torch.no_grad():
            Q_next = target(s_next, task_indicator).detach()
        Q_target_updated = Q_next.clone()
        Q_new = r.unsqueeze(-1) + ad_reward.unsqueeze(-1)
        current_action_idx = torch.argmax(a, dim=-1)
        next_action_idx = torch.argmax(a_next, dim=-1)
        q_next = Q_next.gather(1, next_action_idx)
        Q_new += self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + self.gamma * q_next)
        Q_new = 0.5 * Q_new + 0.5 * critic_eval
        Q_target_updated.scatter_(dim=1, index=current_action_idx, src=Q_new)
        return Q_target_updated, Q_current

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
