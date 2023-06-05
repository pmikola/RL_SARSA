import random
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchviz import make_dot

from binary_converter import float2bit


class Agent:
    def __init__(self, neuralNetwork, neuralNetwork2, valueFunction, num_e_bits, num_m_bits, device):
        self.net = neuralNetwork
        self.net2 = neuralNetwork2
        self.num_e_bits = num_e_bits
        self.num_m_bits = num_m_bits
        self.n_e_bits = torch.tensor([1.] * self.num_e_bits).to(device)
        self.n_m_bits = torch.tensor([1.] * self.num_m_bits).to(device)
        self.sign = torch.tensor([1.]).to(device)
        self.no_of_guesses = 0.
        print(self.net)
        print(self.net2)
        self.BATCH_SIZE = 200
        self.MAX_MEMORY = 100_000
        self.MAX_PRIORITY = torch.tensor(1.).to(device)
        self.vF = valueFunction
        self.memory = deque(maxlen=self.MAX_MEMORY)  # popleft()
        self.device = device
        self.priority = torch.tensor(self.MAX_PRIORITY, dtype=torch.float).to(
            self.device)  # Set initial priority to maximum

        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        # self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
        # weight_decay=0, amsgrad=True)

        self.optimizer = torch.optim.RAdam(self.net.parameters(), lr=0.00005, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0)
        # self.optimizer2 = torch.optim.Adam(self.net2.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08,
        #                                   weight_decay=5e-3, amsgrad=True)

        self.Q_MAX = 0.
        self.lossMSE = nn.MSELoss().to(self.device)
        self.lossL1 = nn.L1Loss().to(self.device)
        self.lossHubert = nn.HuberLoss().to(self.device)
        self.lossKLD = nn.KLDivLoss(reduction="batchmean", log_target=True).to(device)
        # self.loss2 = nn.MSELoss(reduction='sum').to(self.device)

    def remember(self, state, action, reward, next_state, a_next, done):
        self.memory.append((state, action, reward, next_state, a_next, done, self.MAX_PRIORITY))

    def train_short_memory(self, state, action, reward, next_state, a_next, done):
        if len(self.memory) > 0:
            self.train_step(state, action, reward, next_state, a_next, done)

    def prioritized_replay(self, mini_sample, no_top_k):
        states, actions, rewards, next_states, next_actions, dones, priorities = zip(*mini_sample)
        prio = torch.stack(list(priorities), dim=0)
        values, ind = torch.topk(prio, no_top_k)
        indexes = ind.type(torch.int64).tolist()
        # print(len(indexes))
        s = []
        a = []
        r = []
        s_next = []
        a_next = []
        d = []
        for i in indexes:
            s.append(states[i])
            a.append(actions[i])
            r.append(rewards[i])
            s_next.append(next_states[i])
            a_next.append(next_actions[i])
            d.append(dones[i])
        return s, a, r, s_next, a_next, d

    def train_long_memory(self, total_counter):
        k = 0
        if total_counter % 3 and len(self.memory) > self.BATCH_SIZE:
            # RANDOM REPLAY
            if k == 1:
                print("RANDOM")
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
            # mini_sample = self.memory
            # time.sleep(1)
            s, a, r, s_next, a_next, d, p = zip(*mini_sample)
            self.train_step(s, a, r, s_next, a_next, d)
        elif total_counter % 2 and len(self.memory) > self.BATCH_SIZE:
            # PRIORITY HIGH LOSS EXPERIENCE REPLAY
            if k == 1:
                print("PRIORITY")
            mini_sample = self.memory
            s, a, r, s_next, a_next, d = self.prioritized_replay(mini_sample, self.BATCH_SIZE)
            self.train_step(s, a, r, s_next, a_next, d)
        else:
            # RANDOM REPLAY
            if len(self.memory) > self.BATCH_SIZE:
                if k == 1:
                    print("RANDOM -> BATCH SIZE")
                mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
                s, a, r, s_next, a_next, d, p = zip(*mini_sample)
                self.train_step(s, a, r, s_next, a_next, d)
            else:
                if k == 1:
                    print("MEMORY SIZE")
                mini_sample = self.memory
                s, a, r, s_next, a_next, d, p = zip(*mini_sample)
                # time.sleep(1)
                self.train_step(s, a, r, s_next, a_next, d)

    def loss2state(self, i, p, updated_experience):
        self.loss_bits = \
            float2bit(torch.tensor([p, p]), num_e_bits=self.num_e_bits, num_m_bits=self.num_m_bits, bias=127.)[
                0].to(
                self.device)
        self.memory[i] = list(updated_experience)
        for j in range(0, len(self.loss_bits)):
            self.memory[i][0][j] = self.loss_bits[j]

    def train_step(self, s, a, r, s_next, a_next, game_over):
        if len(torch.stack(list(s), dim=0).shape) == 1:
            s = torch.unsqueeze(s.clone().detach(), 0).to(self.device)
            s_next = torch.unsqueeze(s_next.clone().detach(), 0).to(self.device)
            a = torch.unsqueeze(a.clone().detach(), 0).to(self.device)
            a_next = torch.unsqueeze(a_next.clone().detach(), 0).to(self.device)
            r = torch.unsqueeze(torch.tensor(r, dtype=torch.float), 0).to(self.device)
            game_over = (game_over,)  # tuple with only one value
        else:
            s = torch.stack(list(s), dim=0).clone().detach().to(self.device)
            a = torch.stack(list(a), dim=0).clone().detach().to(self.device)
            r = torch.stack(list(torch.tensor(r, dtype=torch.float))).clone().detach().to(self.device)
            s_next = torch.stack(list(s_next), dim=0).clone().detach().to(self.device)
            a_next = torch.stack(list(a_next), dim=0).clone().detach().to(self.device)
            # hn = torch.stack(list(hn), dim=0).clone().detach().to(self.device)
        # print(s)
        target, prediction = self.vF.Q_value(self.net, self.net2, s, a, r, s_next, a_next, game_over)
        # state_a = self.net.state_dict().__str__()

        self.optimizer.zero_grad()
        # lossHubert = self.lossHubert(target, prediction)
        # lossL1 = self.lossL1(target, prediction)
        # lossKLD = self.lossKLD(target, prediction)
        lossMSE = self.lossMSE(target, prediction)

        # LDP is better for long term learning but MSE gives faste
        # ldp = self.vF.distributional_projection(r, target, prediction)
        l = lossMSE  # + 0.5 * ldp
        # print(lossMain.sum().item(),ldp.sum().item())
        l.backward()
        self.optimizer.step()
        # Update priorities
        abs_td_errors = torch.abs(target - prediction).detach()  # Magnitude of our TD error
        priorities = abs_td_errors + 1e-8  # Add small constant to avoid zero priority

        for i, priority in enumerate(priorities):
            experience = self.memory[i]
            p = torch.max(priority)
            updated_experience = (*experience[:-1], p)
            self.memory[i] = updated_experience
            # LOSS AS BIT ARRAY STATE INPUT
            # self.loss2state( i, p, updated_experience)

        # state_b = self.net.state_dict().__str__()
        # self.optimizer2.zero_grad()
        # l2 = self.loss2(target, self.net2(s_next))
        # l2.backward(retain_graph=True)
        # self.optimizer2.step()

        # self.vF.soft_update(self.net, self.net2)

        # if state_a != state_b:
        #    print("LAYERS WEIGHT SUM:", self.net.layers[0].weight.sum())

        ###### COMPUTATIONAL GRAPH GENERATION ######
        # dot = make_dot(prediction, params=dict(self.net.named_parameters()))
        # dot.render(directory='doctest-output', view=True)

    def take_action(self, s, step_counter, dataset, game):
        _, _, body_part = dataset.decode_input(s)
        actions, is_random_next = self.chooseAction(s, dataset, game)
        if is_random_next == 1:
            self.no_of_guesses += 1
            # action = self.actions[step_counter][body_part]
            # print(action)
            a = game.agent.value2action(self.actions, self.net.no_of_heads,
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
        # s = torch.cat((patient, turn, done, self.n_e_bits, self.n_m_bits), axis=0)

        s = torch.cat((patient, turn, done), dim=0)
        s = torch.cat((s, s), dim=0)
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

    def chooseAction(self, state, dataset, game):
        is_random = 0
        hair_type, skin_type, _ = dataset.decode_input(state)
        explore_coef = self.vF.epsilon * (
                game.game_cycles * game.number_of_treatments * game.games / 1.5) / (
                               game.total_counter + 1)
        # if game.cycle >= game.game_cycles-2:
        #     explore_coef = self.vF.epsilon / 2
        if game.cycle >= game.game_cycles - 1:
            explore_coef = -.1
        else:
            pass
        if np.random.uniform(0., 1.) < explore_coef:
            # self.actions, _, _ = dataset.create_target(200)  # For kj_total_var std
            self.actions = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            is_random = 1
        else:
            state = state.to(self.device)
            self.actions = self.net(state.clone())
        return self.actions.clone(), is_random

    def action2value(self, action, num_of_actions, lower_limit, upper_limit):
        action_space = torch.arange(lower_limit, upper_limit, (upper_limit - lower_limit) / num_of_actions)
        max_value_ind = torch.argmax(action).int()
        return action_space[max_value_ind]

    def value2action(self, action_value, num_of_actions, lower_limit, upper_limit):
        action_space = torch.arange(lower_limit, upper_limit, (upper_limit - lower_limit) / num_of_actions).to(
            self.device)
        idx = torch.argmin(torch.abs(action_space - action_value)).int()
        action_space = torch.zeros_like(action_space).to(self.device)
        action_space[idx] = 1.
        return action_space

    def checkReward(self, reward, action, state, dataset, step_counter,game, lower_limit, upper_limit, std):
        hair_type, skin_type, body_part = dataset.decode_input(state)
        ref_value = torch.sum(dataset.kj_total[step_counter][body_part]) / 2
        ref_value_min = ref_value - ref_value * std / 100
        ref_value_max = ref_value + ref_value * std / 100
        reward_factor = torch.abs(ref_value - action).item()
        reward_factor_0 = torch.abs(0. - action).item()
        rf = -(reward_factor + 1e-8) / abs(upper_limit - lower_limit)*0.5
        rf_0 = -(reward_factor_0 + 1e-8) / abs(upper_limit - lower_limit)*0.5
        if game.cycle > 10:
            # MAIN TASK ->LEARNING
            if skin_type > 2 and action >= 1. or hair_type > 1 and action >= 1.:
                reward -= 0.5 + rf_0
            else:
                reward += 0.5 + rf_0
                if ref_value_min < action < ref_value_max:
                    reward += 0.5 + rf
                else:
                    reward -= 0.5 + rf
        else:
            # PRE-TRAINING WITH SIMPLER TASK
            if ref_value_min < action < ref_value_max:
                reward += 0.5 + rf
            else:
                reward -= 0.5 + rf
        return reward

    def int2binary(self, step_counter):
        bin = list(map(float, f'{step_counter:04b}'))
        binT = torch.FloatTensor(bin)
        binT.requires_grad = True
        return binT
