
class ValueFunction:
    def __init__(self, alpha, epsilon, gamma):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma

    def Q_value(self, Q, s, a, r, s_next, a_next, game):
        # self.Q[s, a] = self.Q[s, a] + self.alpha * (self.reward + self.gamma * self.Q[s_next, a_next] - self.Q[s, a])
        # print(s.astype(int))

        aa = game.agent.soft_argmax1d(a)
        an = game.agent.soft_argmax1d(a_next)
        # tmp = self.Q.clone()
        preds = Q[s.int(), aa.int()].clone()
        # print(preds)
        target = r + self.gamma * Q[s_next.int(), an.int()].clone()
        update = Q[s.int(), aa.int()] + self.alpha * (target - preds)
        Q[s.int(), aa.int()] = update
        # print(Q.is_leaf,Q.is_leaf,Q.is_leaf)
        return Q, preds, target
