class ValueFunction:
    def __init__(self, alpha, epsilon, gamma):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def Q_value(self, Q, s, a, r, s_next, a_next, game):
        aa = game.agent.soft_argmax1d(a)
        an = game.agent.soft_argmax1d(a_next)
        preds = Q[s.int(), aa.int()].clone()
        target = r + self.gamma * Q[s_next.int(), an.int()].clone()
        update = Q[s.int(), aa.int()].clone().detach() + self.alpha * (target - preds)
        Q[s.int(), aa.int()].data = update
        return Q, preds, target