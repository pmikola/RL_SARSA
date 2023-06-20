import random
import time
from collections import deque

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchviz import make_dot
from scipy.stats import invgauss, wald

from binary_converter import float2bit


class Agent:
    def __init__(self, Q_Network, Policy_Network, valueFunction, no_of_states, num_e_bits, num_m_bits, device):
        self.last_pain = None
        self.counter = 0
        self.counter_coef = 0
        self.exp_over = 10
        self.net = Q_Network
        self.net2 = Policy_Network
        self.num_e_bits = num_e_bits
        self.num_m_bits = num_m_bits
        self.c = 0
        self.no_of_states = no_of_states
        self.in_a_row_reward = 0.
        self.n_e_bits = torch.tensor([1.] * self.num_e_bits).to(device)
        self.n_m_bits = torch.tensor([1.] * self.num_m_bits).to(device)
        self.sign = torch.tensor([1.]).to(device)
        self.no_of_guesses = 0.
        self.task_indicator = torch.tensor([1., 0., 0., 0.]).to(device)
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
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
                                          weight_decay=0, amsgrad=True)

        # self.optimizer = torch.optim.RAdam(self.net.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
        #                                    weight_decay=0)
        self.optimizer2 = torch.optim.Adam(self.net2.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-08,
                                           weight_decay=0, amsgrad=True)

        self.Q_MAX = 0.
        self.lossMSE = nn.MSELoss().to(self.device)
        self.lossL1 = nn.L1Loss().to(self.device)
        self.lossHubert = nn.HuberLoss().to(self.device)
        self.lossKLD = nn.KLDivLoss(reduction="batchmean", log_target=True).to(device)
        # self.loss2 = nn.MSELoss(reduction='sum').to(self.device)

    def remember(self, state, action, reward, next_state, a_next, done, task_indicator, ad_reward):

        self.memory.append(
            (state, action, reward, next_state, a_next, done, task_indicator, ad_reward, self.MAX_PRIORITY))

    def train_short_memory(self, state, action, reward, next_state, a_next, done, tid, ad_reward):
        if len(self.memory) > 0:
            l = self.train_step(state, action, reward, next_state, a_next, done, tid, ad_reward)
            return l

    def prioritized_replay(self, mini_sample, no_top_k):
        states, actions, rewards, next_states, next_actions, dones, tid, ad_reward, priorities = zip(*mini_sample)
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
        t_id = []
        adr = []
        for i in indexes:
            s.append(states[i])
            a.append(actions[i])
            r.append(rewards[i])
            s_next.append(next_states[i])
            a_next.append(next_actions[i])
            d.append(dones[i])
            t_id.append(tid[i])
            adr.append(ad_reward[i])
        return s, a, r, s_next, a_next, d, t_id, ad_reward

    def train_long_memory(self, total_counter):
        k = 0
        if total_counter % 3 == 0 and len(self.memory) > self.BATCH_SIZE:
            # RANDOM REPLAY
            if k == 1:
                print("RANDOM")
            mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
            # mini_sample = self.memory
            # time.sleep(1)
            s, a, r, s_next, a_next, d, tid, p, ad_reward = zip(*mini_sample)
            l = self.train_step(s, a, r, s_next, a_next, d, tid, ad_reward)
        if total_counter % 2 == 0 and len(self.memory) > self.BATCH_SIZE:
            # PRIORITY HIGH LOSS EXPERIENCE REPLAY
            if k == 1:
                print("PRIORITY")
            mini_sample = self.memory
            s, a, r, s_next, a_next, d, tid, ad_reward = self.prioritized_replay(mini_sample, self.BATCH_SIZE)
            l = self.train_step(s, a, r, s_next, a_next, d, tid, ad_reward)
        else:
            # RANDOM REPLAY
            if len(self.memory) > self.BATCH_SIZE:
                if k == 1:
                    print("RANDOM -> BATCH SIZE")
                mini_sample = random.sample(self.memory, self.BATCH_SIZE)  # list of tuples
                s, a, r, s_next, a_next, d, tid, p, ad_reward = zip(*mini_sample)
                l = self.train_step(s, a, r, s_next, a_next, d, tid, ad_reward)
            else:
                if k == 1:
                    print("MEMORY SIZE")
                mini_sample = self.memory
                s, a, r, s_next, a_next, d, tid, p, ad_reward = zip(*mini_sample)
                # time.sleep(1)
                l = self.train_step(s, a, r, s_next, a_next, d, tid, ad_reward)

        return l

    def loss2state(self, i, p, updated_experience):
        self.loss_bits = \
            float2bit(torch.tensor([p, p]), num_e_bits=self.num_e_bits, num_m_bits=self.num_m_bits, bias=127.)[
                0].to(
                self.device)
        self.memory[i] = list(updated_experience)
        for j in range(0, len(self.loss_bits)):
            self.memory[i][0][j] = self.loss_bits[j]

    def train_step(self, s, a, r, s_next, a_next, game_over, tid, ad_reward):
        if len(torch.stack(list(s), dim=0).shape) == 1:
            s = torch.unsqueeze(s.clone().detach(), 0).to(self.device)
            s_next = torch.unsqueeze(s_next.clone().detach(), 0).to(self.device)
            a = torch.unsqueeze(a.clone().detach(), 0).to(self.device)
            a_next = torch.unsqueeze(a_next.clone().detach(), 0).to(self.device)
            r = torch.unsqueeze(torch.tensor(r, dtype=torch.float), 0).to(self.device)
            ad_reward = torch.unsqueeze(torch.tensor(ad_reward, dtype=torch.float), 0).to(self.device)
            tid = torch.unsqueeze(tid.clone().detach(), 0).to(self.device)

            game_over = (game_over,)  # tuple with only one value
        else:
            s = torch.stack(list(s), dim=0).clone().detach().to(self.device)
            a = torch.stack(list(a), dim=0).clone().detach().to(self.device)
            r = torch.stack(list(torch.tensor(r, dtype=torch.float))).clone().detach().to(self.device)
            ad_reward = torch.stack(list(torch.tensor(ad_reward, dtype=torch.float))).clone().detach().to(self.device)
            s_next = torch.stack(list(s_next), dim=0).clone().detach().to(self.device)
            a_next = torch.stack(list(a_next), dim=0).clone().detach().to(self.device)
            # tid = torch.stack([torch.tensor(t, device=self.device) for t in tid], dim=0).clone().detach().to(
            # self.device)
            tid = torch.stack(list(tid), dim=0).clone().detach().to(self.device)

            # tid = torch.stack(list(tid), dim=0).clone().detach().to(self.device)

            # hn = torch.stack(list(hn), dim=0).clone().detach().to(self.device)
        # print(s)
        target, prediction, prediction2 = self.vF.Q_value(self.net, self.net2, s, a, r, s_next, a_next, game_over,
                                                          tid, ad_reward)
        # state_a = self.net.state_dict().__str__()

        self.optimizer.zero_grad()
        lossHubert = self.lossHubert(target, prediction)  # BEST LOSS
        # lossHubert_sum = self.lossHubert(target.sum(), prediction.sum())
        # lossL1 = self.lossL1(target, prediction)
        # lossKLD = self.lossKLD(target, prediction)  # slower training but better with multi-task
        # lossMSE = self.lossMSE(target, prediction)

        # LDP is better for long term learning but MSE gives faste
        # ldp = self.vF.distributional_projection(r, target, prediction)
        l = lossHubert  # lossKLD + lossMSE  # + 0.5 * ldp
        # print(lossMain.sum().item(),ldp.sum().item())
        l.backward(retain_graph=True)
        self.optimizer.step()
        # lossHubert_sum.backward(retain_graph=True)
        # self.optimizer.step()
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
        target, prediction, prediction2 = self.vF.policy(self.net, self.net2, s, a, r, s_next, a_next, game_over,
                                                         tid, ad_reward)
        self.optimizer2.zero_grad()
        lossHubert2 = self.lossHubert(target, prediction2)
        l2 = lossHubert2

        l2.backward(retain_graph=True)
        self.optimizer2.step()

        # self.vF.soft_update(self.net, self.net2)

        # if state_a != state_b:
        #    print("LAYERS WEIGHT SUM:", self.net.layers[0].weight.sum())

        ###### COMPUTATIONAL GRAPH GENERATION ######
        # dot = make_dot(prediction, params=dict(self.net.named_parameters()))
        # dot.render(directory='doctest-output', view=True)

        return l.item()

    def take_action(self, s, step_counter, dataset, game):
        _, _, body_part = dataset.decode_input(s)

        actions, is_random_next = self.chooseAction(s, dataset, game)

        if is_random_next == 1:
            self.no_of_guesses += 1
            # action = self.actions[step_counter][body_part]
            # print(action)

            a = game.agent.value2action(self.actions, self.net.no_of_heads,
                                        game.lower_limit, game.upper_limit)
            a = torch.unsqueeze(a, dim=0)


        else:
            a = self.actions
        a_value = game.agent.action2value(a, self.net.no_of_heads, game.lower_limit, game.upper_limit)
        return a, a_value, body_part

    def take_next_action(self, s_next, action, step_counter, dataset, game):
        _, _, body_part = dataset.decode_input(s_next)

        actions, is_random_next = self.chooseNextAction(s_next, action, dataset, game)

        if is_random_next == 1:
            self.no_of_guesses += 1
            # action = self.actions[step_counter][body_part]
            # print(action)

            a = game.agent.value2action(self.actions, self.net.no_of_heads,
                                        game.lower_limit, game.upper_limit)
            a = torch.unsqueeze(a, dim=0)
        else:
            a = self.actions
        a_value = game.agent.action2value(a, self.net.no_of_heads, game.lower_limit, game.upper_limit)
        return a, a_value, body_part

    def get_state(self, pain, step_counter, dataset):
        turn = self.int2binary(step_counter).to(self.device)
        patient = dataset.create_input_set().to(self.device)
        if step_counter >= 9:
            done = torch.tensor([1.]).to(self.device)
            game_over = True
        else:
            done = torch.tensor([0.]).to(self.device)
            game_over = False
        # s = torch.cat((patient, turn, done, self.n_e_bits, self.n_m_bits), axis=0)

        s = torch.cat((patient, turn, done, pain), dim=0)
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
        explore_coef = self.vF.epsilon
        hair_type, skin_type, _ = dataset.decode_input(state)
        # c = self.vF.epsilon *  (self.counter_coef + 1)
        # print(1 / np.power(self.counter_coef, 2))
        if game.cycle >= game.game_cycles - 2:
            explore_coef = self.counter_coef + 1 * 1000.
        else:
            pass
        if np.random.uniform(-self.vF.epsilon, 1 / np.power(self.counter_coef + 1, 2)) > explore_coef:
            # self.actions, _, _ = dataset.create_target(200)  # For kj_total_var std
            self.actions = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            is_random = 1
        else:
            state = state.to(self.device)
            self.actions = self.net(state.clone(), self.task_indicator)
        return self.actions.clone(), is_random

    def chooseNextAction(self, state_next, action, dataset, game):
        is_random = 0
        explore_coef = self.vF.epsilon
        hair_type, skin_type, _ = dataset.decode_input(state_next)
        # c = self.vF.epsilon *  (self.counter_coef + 1)
        # print(1 / np.power(self.counter_coef, 2))
        if game.cycle >= game.game_cycles - 2:
            explore_coef = self.counter_coef + 1 * 1000.
        else:
            pass
        if np.random.uniform(-self.vF.epsilon, 1 / np.power(self.counter_coef + 1, 2)) > explore_coef:
            # self.actions, _, _ = dataset.create_target(200)  # For kj_total_var std
            self.actions = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit)).to(self.device)
            is_random = 1
        else:
            state_next = state_next.to(self.device)
            action = action.to(self.device)
            self.actions = self.net2(state_next.clone(), action.clone(), self.task_indicator)
            # self.actions = self.net(state_next.clone(), self.task_indicator)

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

    def checkRewardAndModBehavior(self, reward, action, state, dataset, step_counter, game, lower_limit, upper_limit,
                                  std, pain, pain_p):
        hair_type, skin_type, body_part = dataset.decode_input(state)
        self.task_indicator[0] = torch.tensor([1.]).to(self.device)
        r = 0.1
        additional_reward = 0.
        if game.task_id == 0.:
            r_v = dataset.kj_total[step_counter][body_part]
            ref_value = torch.sum(r_v) / 2
        elif game.task_id == 1.:
            r_v = dataset.hz[step_counter][body_part]
            ref_value = torch.sum(r_v) / 2
        else:
            r_v = dataset.j_cm2[step_counter][body_part]
            ref_value = torch.sum(r_v) / 2
        rf_vl = ref_value * std / 100
        ref_value_min = ref_value - rf_vl
        ref_value_max = ref_value + rf_vl
        # reward_factor = torch.abs(ref_value - action).item()
        # reward_factor_0 = torch.abs(0. - action).item()
        # print(action)
        # rf = -(reward_factor + 1e-8) / abs(upper_limit - lower_limit)
        # rf_0 = -(reward_factor_0 + 1e-8) / abs(upper_limit - lower_limit)

        if self.last_pain is None:
            self.last_pain = pain
        else:
            if self.last_pain[step_counter - 1] >= pain[step_counter] and game.task_id != 0:
                additional_reward += 0.5
            if (self.last_pain[step_counter - 1] + 1e-8) / 2 >= pain[step_counter] and game.task_id != 0:
                additional_reward += 2
            if pain[step_counter] <= 0.05 and game.task_id != 0:
                additional_reward += 3

        if step_counter == 0:
            self.c = not self.c
            # print(self.counter)
        if self.c == 1:
            r_0 = 10.
            r_a = 10.
        else:
            r_0 = 10.
            r_a = 10.
        if skin_type > 2 or hair_type > 1:
            if action >= 1.:
                reward -= 0.
            else:
                reward += r * r_0
        else:
            if pain[step_counter] >= self.last_pain[step_counter - 1] and game.task_id != 0 and step_counter >= 1:
                if ref_value_min - rf_vl < action < ref_value_max - rf_vl:
                    reward += r * r_a  # + rf
                    pain_p = 1. - pain[step_counter].cpu()
                    print("last",pain_p)
                else:
                    # ommiting negarive rewards better regarding Q value without mean/sum estimate??
                    reward -= 0.  # r * r_a  # + rf
            elif pain[step_counter] >= 0.25 and game.task_id != 0:
                if ref_value_min - rf_vl < action < ref_value_max - rf_vl:
                    reward += r * r_a  # + rf
                    pain_p = 1. - pain[step_counter].cpu()
                    print("first",pain_p)
                else:
                    # ommiting negarive rewards better regarding Q value without mean/sum estimate??
                    reward -= 0.  # r * r_a  # + rf
            else:
                if ref_value_min < action < ref_value_max:
                    reward += r * r_a  # + rf
                else:
                    # ommiting negarive rewards better regarding Q value without mean/sum estimate??
                    reward -= 0.  # r * r_a  # + rf

        if step_counter == 8:
            del self.last_pain
            self.last_pain = None
        else:
            self.last_pain[step_counter] = pain[step_counter]
            self.in_a_row_reward = 0.
        if step_counter == 8 and reward >= 5:
            additional_reward += 1.

        if step_counter == 8 and reward >= 6:
            additional_reward += 2.

        if step_counter == 8 and reward >= 7:
            additional_reward += 5.

        if step_counter == 8 and reward >= 8:
            additional_reward += 7.  # 10
        if step_counter == 8 and reward >= 9:
            # self.in_a_row_reward += 1.
            print("  maximum reward!")
            additional_reward += 15.  # + self.in_a_row_reward  # 25

        return reward, additional_reward, pain_p

    def int2binary(self, step_counter):
        bin = list(map(float, f'{step_counter:04b}'))
        binT = torch.FloatTensor(bin)
        binT.requires_grad = True
        return binT
