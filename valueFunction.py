import time

import torch


class ValueFunction:
    def __init__(self, alpha, epsilon, gamma, tau, device,no_of_actions,n_steps=2,v_min=-900,v_max=900):
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

    def Q_value(self, net, net2, s, a, r, s_next, a_next, game_over):
        Q = net(s)
        # Calculate Q-values using the main network
        Q_main = net(s_next).detach()
        Q_target = Q.clone()

        for idx in range(len(game_over)):
            Q_new = r[idx]

            if not game_over[idx]:
                # Indexing Q-main using max value position from next action give us SARS(A)
                next_action_idx = torch.argmax(a_next[idx]).item()
                Q_new += self.alpha * (r[idx] + self.gamma * Q_main[idx][next_action_idx])

                # Multi-step return -> looking ahead of choosing path by n steps
                for step in range(1, self.n_steps):
                    if idx + step < len(game_over):
                        Q_new += (self.gamma ** step) * r[idx + step]

            current_action_idx = torch.argmax(a[idx]).item()
            Q_target[idx][current_action_idx] = Q_new

        return Q_target, Q

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

    def distributional_projection(self, rewards, Q_target, Q):
        # Project distrubution of rewards -> minimizing Kullback–Leibler (relative entropy) divergence between Q_target and Q
        batch_size = rewards.size(0)
        target_atoms = rewards.unsqueeze(1) + self.gamma * self.support.unsqueeze(0)
        target_atoms = torch.clamp(target_atoms, self.v_min, self.v_max)
        b = (target_atoms - self.v_min) / self.delta_z
        lower_bounds = b.floor().long()
        upper_bounds = b.ceil().long()
        # Error check
        #assert torch.all(lower_bounds >= 0) and torch.all(upper_bounds < Q_target.size(1)), "Index out of bounds error"
        lower_mask = torch.zeros_like(Q_target).scatter_(1, lower_bounds, 1)
        upper_mask = torch.zeros_like(Q_target).scatter_(1, upper_bounds, 1)
        Q_target = (lower_mask + upper_mask) * 0.5
        Q_target /= Q_target.sum(dim=1, keepdim=True)
        loss = -torch.sum(Q_target * torch.log(Q + 1e-8)) / batch_size
        return loss