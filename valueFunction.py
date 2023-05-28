import torch


class ValueFunction:
    def __init__(self, alpha, epsilon, gamma, device):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

    # def Q_value(self, net, s, a, r, s_next, game_over):
    #     prediction = net(s)
    #     target = prediction.clone()
    #
    #     # Calculate Q-values using the main network
    #     Q_main = net(s_next).detach()
    #     Q_target = target.clone()
    #
    #     for idx in range(len(game_over)):
    #         Q_new = r[idx]
    #         if not game_over[idx]:
    #             Q_new += self.gamma * Q_main[idx][torch.argmax(a_next[idx]).item()]
    #
    #         action_idx = torch.argmax(a[idx]).item()
    #         Q_target[idx][action_idx] = Q_new
    #
    #     return Q_target, prediction

    def Q_value(self, net, s, a, r, s_next, game_over):
        prediction = net(s)
        target = prediction.clone()

        # Calculate Q-values using the main network
        Q_main = net(s_next).detach()
        Q_target = target.clone()

        for idx in range(len(game_over)):
            Q_new = r[idx]
            if not game_over[idx]:
                Q_new = r[idx] + self.gamma * torch.max(Q_main[idx])
            action_idx = torch.argmax(a[idx]).item()
            Q_target[idx][action_idx] = Q_new

        return Q_target, prediction