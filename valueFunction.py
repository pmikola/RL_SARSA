import torch


class ValueFunction:
    def __init__(self, alpha, epsilon, gamma, tau, device):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device
        self.tau = tau

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
                Q_new +=  self.alpha*(r[idx] + self.gamma * Q_main[idx][next_action_idx] )

            current_action_idx = torch.argmax(a[idx]).item()
            Q_target[idx][current_action_idx] = Q_new

        return Q_target, Q

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
