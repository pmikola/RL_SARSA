class ValueFunction:
    def __init__(self, alpha, epsilon, gamma,device):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.device = device

    def Q_value(self, Q, s, a, r, s_next, a_next, game):
        aa = game.agent.soft_argmax1d(a)
        an = game.agent.soft_argmax1d(a_next)
        qval = Q[s.int(), aa.int()].clone().to(self.device)
        target = r + self.gamma * Q[s_next.int(), an.int()].clone().to(self.device)
        td_error = target - qval
        update = Q[s.int(), aa.int()].clone().detach().to(self.device) + self.alpha * td_error
        Q[s.int(), aa.int()].data = update
        return Q, qval, target