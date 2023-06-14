import time

import torch
import torch.nn.functional as F


class ValueFunction:
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

    def Q_value(self, net, net2, s, a, r, s_next, a_next, game_over, task_indicator, ad_reward):
        Q = net(s, task_indicator)
        # Calculate Q-values using the main network
        # Q_main = net2(s_next,a,task_indicator)
        Q_main = net(s_next, task_indicator)
        Q_target = Q.clone()

        for idx in range(len(game_over)):
            Q_new = r[idx] + ad_reward[idx]
            current_action_idx = torch.argmax(a[idx]).item()
            if not game_over[idx]:
                # Indexing Q-main using max value position from next action give us SARS(A)
                next_action_idx = torch.argmax(a_next[idx]).item()
                Q_new += self.alpha * (
                        (r[idx] + ad_reward[idx]) + self.gamma * Q_main[idx][next_action_idx])
                # Multi-step return -> looking ahead of choosing path by n steps
                for step in range(1, self.n_steps):
                    if idx + step < len(game_over):
                        Q_new += (self.gamma ** step) * (r[idx + step] + ad_reward[idx + step])

            Q_target[idx][current_action_idx] = Q_new

        return Q_target, Q, Q_main

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
