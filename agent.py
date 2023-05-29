import random
from collections import deque

import numpy as np
import torch
from torch import nn

import torch.nn.functional as F
from torchviz import make_dot

MAX_MEMORY = 100_000
BATCH_SIZE = 200


class Agent:
    def __init__(self, neuralNetwork,neuralNetwork2, valueFunction, device):
        self.net = neuralNetwork
        self.net2 = neuralNetwork2
        print(self.net)
        self.vF = valueFunction
        self.memory = deque(maxlen=MAX_MEMORY)  # popleft()
        self.device = device

        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=False)
        self.optimizer2 = torch.optim.Adam(self.net.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=False)

        self.Q_MAX = 0.
        # self.loss = nn.CrossEntropyLoss(reduction='mean').to(self.device)
        self.loss = nn.MSELoss(reduction='sum').to(self.device)
        self.loss2 = nn.MSELoss(reduction='sum').to(self.device)

    def remember(self, state, action, reward, next_state, a_next, done):
        self.memory.append((state, action, reward, next_state, a_next, done))

    def train_short_memory(self, state, action, reward, next_state, a_next, done):
        self.train_step(state, action, reward, next_state, a_next, done)

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory, BATCH_SIZE)  # list of tuples
        else:
            mini_sample = self.memory
        states, actions, rewards, next_states, next_actions, dones = zip(*mini_sample)
        self.train_step(states, actions, rewards, next_states, next_actions, dones)

    def train_step(self, s, a, r, s_next, a_next, game_over):
        if len(torch.stack(list(s), dim=0).shape) == 1:
            s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0).to(self.device)
            s_next = torch.unsqueeze(torch.tensor(s_next, dtype=torch.float), 0).to(self.device)
            a = torch.unsqueeze(torch.tensor(a, dtype=torch.float), 0).to(self.device)
            a_next = torch.unsqueeze(torch.tensor(a_next, dtype=torch.float), 0).to(self.device)
            r = torch.unsqueeze(torch.tensor(r, dtype=torch.float), 0).to(self.device)
            game_over = (game_over,)  # tuple with only one value
        else:
            s = torch.tensor(torch.stack(list(s), dim=0), dtype=torch.float).to(self.device)
            a = torch.tensor(torch.stack(list(a), dim=0), dtype=torch.float).to(self.device)
            r = torch.tensor(torch.stack(list(torch.tensor(r, dtype=torch.float)), dim=0), dtype=torch.float).to(
                self.device)
            s_next = torch.tensor(torch.stack(list(s_next), dim=0), dtype=torch.float).to(self.device)
            a_next = torch.tensor(torch.stack(list(a_next), dim=0), dtype=torch.float).to(self.device)

        target, prediction = self.vF.Q_value(self.net,self.net2, s, a, r, s_next, a_next, game_over)
        # state_a = self.net.state_dict().__str__()

        self.optimizer.zero_grad()
        loss = self.loss(target, prediction)
        loss.backward(retain_graph=True)
        self.optimizer.step()
        # state_b = self.net.state_dict().__str__()
        self.optimizer2.zero_grad()
        loss2 = self.loss(target, self.net2(s_next).detach())
        loss2.backward(retain_graph=True)
        self.optimizer2.step()

        self.vF.soft_update(self.net, self.net2)

        # if state_a != state_b:
        #    print("LAYERS WEIGHT SUM:", self.net.layers[0].weight.sum())

    def take_action(self, s, step_counter, total_counter, dataset, game):
        _, _, body_part = dataset.decode_input(s)
        actions, is_random_next = self.chooseAction(s, dataset, total_counter)
        if is_random_next == 1:
            action = self.actions[step_counter][body_part]
            a = game.agent.value2action(action, self.net.no_of_heads,
                                        game.lower_limit, game.upper_limit)
        else:
            a = self.actions
        a_value = game.agent.action2value(a, self.net.no_of_heads, game.lower_limit, game.upper_limit)
        return a, a_value, body_part

    def get_state(self, step_counter, dataset):
        turn = self.int2binary(step_counter).to(self.device)
        patient = dataset.create_input_set().to(self.device)
        if step_counter >= 9:
            done = torch.tensor([1.]).to(self.device)
            game_over = True
        else:
            done = torch.tensor([0.]).to(self.device)
            game_over = False

        s = torch.cat((patient, turn, done), axis=0)
        # print(s)
        return s, done, game_over

    def soft_argmax1d(self, input, beta=100):
        *_, n = input.shape
        softmax = nn.Softmax(dim=-1)
        input = softmax(beta * input)
        indices = torch.linspace(0, 1, n, device=self.device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    def soft_argmin1d(self, input, beta=100):
        *_, n = input.shape
        softmin = nn.Softmin(dim=-1)
        input = softmin(beta * input)
        indices = torch.linspace(0, 1, n, device=self.device)
        result = torch.sum((n - 1) * input * indices, dim=-1)
        return result

    def chooseAction(self, state, dataset, total_counter):
        is_random = 0
        hair_type, skin_type, _ = dataset.decode_input(state)
        if np.random.uniform(0, 1) < self.vF.epsilon * total_counter:  # random action
            self.actions, _, _ = dataset.create_target(100)  # For kj_total_var single regression output
            is_random = 1
        else:
            state = state.to(self.device)
            # dot = make_dot(self.actions, params=dict(self.net.named_parameters()))
            # dot.render(directory='doctest-output', view=True)
            self.actions = self.net(state.clone())
        return self.actions.clone(), is_random

    def action2value(self, action, num_of_actions, lower_limit, higher_limit):
        action_space = torch.arange(lower_limit, higher_limit, (higher_limit - lower_limit) / num_of_actions)
        max_value_ind = torch.argmax(action).int()
        return action_space[max_value_ind]

    def value2action(self, action_value, num_of_actions, lower_limit, higher_limit):
        action_space = torch.arange(lower_limit, higher_limit, (higher_limit - lower_limit) / num_of_actions).to(
            self.device)
        idx = torch.argmin(torch.abs(action_space - action_value)).int()
        action_space = torch.zeros_like(action_space).to(self.device)
        action_space[idx] = 1.
        return action_space

    def checkReward(self, reward, body_part, action, dataset, step_counter, std):
        ref_value = torch.sum(dataset.kj_total[step_counter][body_part]) / 2
        ref_value_min = ref_value - ref_value * std / 100
        ref_value_max = ref_value + ref_value * std / 100
        if ref_value_min < action < ref_value_max:
            reward += 1
        else:
            reward -= 1
        return reward

    def int2binary(self, step_counter):
        bin = list(map(float, f'{step_counter:04b}'))
        binT = torch.FloatTensor(bin)
        binT.requires_grad = True
        return binT
