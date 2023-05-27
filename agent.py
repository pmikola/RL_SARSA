import numpy as np
import torch
from torch import nn


class Agent:
    def __init__(self, neuralNetwork, valueFunction):
        self.net = neuralNetwork
        print(self.net)
        self.vF = valueFunction

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.1, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=False)
        self.Q_MAX = 0.
        self.loss = nn.MSELoss(reduction='sum')

    def train(self, Qval, s, a, r, sn, an, game):
        self.game = game

        Qval = Qval.clone().detach().requires_grad_(True).to(torch.float32)
        with torch.no_grad():
            Qval, preds, target = self.vF.Q_value(Qval, s, a, r, sn, an, game)
        preds = preds.clone().detach().requires_grad_(True)
        target = target.clone().detach().requires_grad_(True)

        self.l = self.loss(preds, target)

        state_a = self.net.state_dict().__str__()

        self.optimizer.zero_grad()
        self.l.backward(retain_graph=True)
        self.optimizer.step()

        state_b = self.net.state_dict().__str__()

        if state_a == state_b:
            print("LAYERS WEIGHT SUM:", self.net.layers[0].weight.sum())

        if Qval.mean() > self.Q_MAX:
            self.Q_MAX = Qval.mean()
        return Qval

    def soft_argmax1d(self, input, beta=100):
        *_, n = input.shape
        input = torch.softmax(beta * input, dim=-1)
        indices = torch.linspace(0, 1, n)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    def soft_argmin1d(self, input, beta=100):
        *_, n = input.shape
        softmin = nn.Softmin(dim=-1)
        input = softmin(beta * input)
        indices = torch.linspace(0, 1, n)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    def chooseAction(self, state, dataset):
        is_random = 0
        if np.random.uniform(0, 1) < self.vF.epsilon:  # random action
            self.actions, _, _ = dataset.create_target(50)  # For kj_total_var single regression output
            is_random = 1
        else:
            self.actions = self.net(state.clone())
        return self.actions.clone(), is_random

    def action2value(self, game, action, num_of_actions, lower_limit, higher_limit):
        action_space = torch.arange(lower_limit, higher_limit, (higher_limit - lower_limit) / num_of_actions)
        max_value_ind = game.agent.soft_argmax1d(action).int()
        return action_space[max_value_ind]

    def value2action(self, game, action_value, num_of_actions, lower_limit, higher_limit):
        action_space = torch.arange(lower_limit, higher_limit, (higher_limit - lower_limit) / num_of_actions)
        idx = game.agent.soft_argmin1d(torch.abs(action_space - action_value)).int()
        action_space.fill_(0.)
        action_space[idx] = 1.
        return action_space

    def checkReward(self, reward, body_part, action, dataset, step_counter, std):
        ref_value = torch.sum(dataset.kj_total[step_counter][body_part]) / 2
        ref_value_min = ref_value - ref_value * std / 100
        ref_value_max = ref_value + ref_value * std / 100
        if ref_value_min < action < ref_value_max:
            reward += 1
        else:
            pass
        return reward

    def int2binary(self, step_counter):
        bin = list(map(float, f'{step_counter:04b}'))
        binT = torch.FloatTensor(bin)
        binT.requires_grad = True
        return binT