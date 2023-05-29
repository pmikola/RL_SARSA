import torch


class ValueFunction:
    def __init__(self, alpha, epsilon, gamma,tau, device):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau
        self.device = device

    def Q_value(self, net, target_net, s, a, r, s_next, a_next, game_over):
        prediction = net(s)
        target = prediction.clone()

        # Q_main = net(s_next).detach()
        Q_target = target.clone()

        for idx in range(len(game_over)):
            Q_new = r[idx]

            if not game_over[idx]:
                next_action_idx = torch.argmax(a_next[idx]).item()
                Q_next = target_net(s_next).detach()
                Q_new += self.gamma * Q_next[idx][next_action_idx]

            current_action_idx = torch.argmax(a[idx]).item()
            Q_target[idx][current_action_idx] = Q_new

        return Q_target, prediction

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))