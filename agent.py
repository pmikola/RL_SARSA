import random
import time
from collections import deque
from statistics import mean

import numpy as np
import torch
from kiwisolver import strength
from numpy import argmin
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from torchviz import make_dot
from scipy.stats import invgauss, wald

from binary_converter import float2bit


class Agent:
    def __init__(self, actor,target_actor, critic_1, critic_2, target_critic_1,target_critic_2, valueFunction, no_of_states, num_e_bits, num_m_bits, device):
        self.total_counter = 0.
        self.eps = (1e-3 + 0.95 * np.exp(-1e-2 * self.total_counter))
        self.counter = 0
        self.counter_coef = 0
        self.exp_over = 20
        self.actor = actor
        self.target_actor = target_actor
        self.target_critic_1 = target_critic_1
        self.target_critic_2 = target_critic_2
        self.critic_1 = critic_1
        self.critic_2 = critic_2
        self.num_e_bits = num_e_bits
        self.num_m_bits = num_m_bits
        self.c = 0
        self.no_of_states = no_of_states
        self.n_e_bits = torch.tensor([1.] * self.num_e_bits).to(device)
        self.n_m_bits = torch.tensor([1.] * self.num_m_bits).to(device)
        self.sign = torch.tensor([1.]).to(device)
        self.no_of_guesses = 0.
        self.BATCH_SIZE = 64
        self.MAX_MEMORY = 20000
        self.MAX_PRIORITY = torch.tensor(-100).to(device)
        self.vF = valueFunction
        self.memory = deque(maxlen=self.MAX_MEMORY)  # popleft()
        self.device = device
        self.priority = torch.tensor(self.MAX_PRIORITY, dtype=torch.float).to(
            self.device)
        self.i_s = random.randint(0, 2)

        # self.optimizer = torch.optim.SGD(self.net.parameters(), lr=0.001)
        self.optimizer_q_value_critic_1 = torch.optim.Adam(self.critic_1.parameters(),
                                                           lr=1e-4,
                                                           betas=(0.9, 0.999),
                                                           eps=1e-08,
                                                           weight_decay=1e-6,
                                                           amsgrad=True
                                                           )

        self.optimizer_q_value_critic_2 = torch.optim.Adam(self.critic_2.parameters(),
                                                           lr=1e-4,
                                                           betas=(0.9, 0.999),
                                                           eps=1e-08,
                                                           weight_decay=1e-6,
                                                           amsgrad=True
                                                           )

        self.optimizer_actor_policy_gradient = torch.optim.Adam(self.actor.parameters(),
                                                                lr=1e-4,
                                                                betas=(0.9, 0.999),
                                                                eps=1e-08,
                                                                weight_decay=1e-6,
                                                                amsgrad=True)

        self.Q_MAX = 0.
        self.lossMSE = nn.MSELoss().to(self.device)
        self.lossL1 = nn.L1Loss().to(self.device)
        self.loss_fn = nn.HuberLoss().to(self.device)
        self.lossKLD = nn.KLDivLoss(reduction="batchmean", log_target=True).to(device)
        # self.loss2 = nn.MSELoss(reduction='sum').to(self.device)

    def remember(self, state, a1,a2,a3, reward, next_state, a_next1,a_next2,a_next3, done, task_id, ad_reward):
        self.memory.append((state, a1,a2,a3, reward, next_state, a_next1,a_next2,a_next3, done, task_id, ad_reward, self.MAX_PRIORITY))

    def train_short_memory(self, state, a1,a2,a3, reward, next_state, a_next1,a_next2,a_next3, done, tid, ad_reward):
        if len(self.memory) > 0:
            l,tid = self.train_step(state, a1,a2,a3, reward, next_state, a_next1,a_next2,a_next3, done, tid, ad_reward)
            return l,tid

    def prioritized_replay(self, mini_sample, no_top_k):
        states, a1,a2,a3, rewards, next_states, a_next1,a_next2,a_next3, dones, tid, ad_reward, priorities = zip(*mini_sample)
        prio = torch.stack(list(priorities), dim=0).squeeze(dim=1)
        values, ind = torch.topk(prio, no_top_k)
        indexes = ind.type(torch.int64).tolist()
        # print(len(indexes))
        s = []
        a_1 = []
        a_2 = []
        a_3 = []
        r = []
        s_next = []
        a_next_1 = []
        a_next_2 = []
        a_next_3 = []
        d = []
        t_id = []
        adr = []
        for i in indexes:
            s.append(states[i])
            a_1.append(a1[i])
            a_2.append(a2[i])
            a_3.append(a3[i])
            r.append(rewards[i])
            s_next.append(next_states[i])
            a_next_1.append(a_next1[i])
            a_next_2.append(a_next2[i])
            a_next_3.append(a_next3[i])
            d.append(dones[i])
            t_id.append(tid[i])
            adr.append(ad_reward[i])
        return s, a_1,a_2,a_3, r, s_next, a_next_1,a_next_2,a_next_3, d, t_id, adr

    def train_long_memory(self, total_counter):
        k = 0
        if total_counter % 2 == 0 and len(self.memory) > self.BATCH_SIZE:
            # RANDOM REPLAY
            if k == 1:
                print("RANDOM")
            mini_sample = random.choices(self.memory, k=self.BATCH_SIZE)  # list of tuples
            s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward, priorities = zip(*mini_sample)
            l,tid = self.train_step(s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward)
        # elif total_counter % 3 == 0 and len(self.memory) > self.BATCH_SIZE:
        #     # PRIORITY HIGH LOSS EXPERIENCE REPLAY
        #     if k == 1:
        #         print("PRIORITY")
        #     mini_sample = random.choices(self.memory, k=4*self.BATCH_SIZE)
        #     s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward = self.prioritized_replay(mini_sample, self.BATCH_SIZE)
        #     l,tid = self.train_step(s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward)
        else:
            # RANDOM REPLAY
            if len(self.memory) > self.BATCH_SIZE:
                if k == 1:
                    print("RANDOM -> BATCH SIZE")
                mini_sample = random.sample(self.memory, self.BATCH_SIZE)
                s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward, priorities = zip(*mini_sample)
                l = self.train_step(s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward)
            else:
                if k == 1:
                    print("MEMORY SIZE")
                mini_sample = random.choices(self.memory, k=self.BATCH_SIZE)
                s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward, priorities = zip(*mini_sample)
                l,tid = self.train_step(s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, d, tid, ad_reward)
        return l,tid

    def train_step(self, s, a1,a2,a3, r, s_next, a_next1,a_next2,a_next3, done, tid, ad_reward):
        if len(torch.stack(list(s), dim=0).shape) == 1:
            s = torch.unsqueeze(s.clone().detach(), 0).to(self.device)
            s_next = torch.unsqueeze(s_next.clone().detach(), 0).to(self.device)
            a1 = torch.unsqueeze(a1.clone().detach(), 0).to(self.device)
            a2 = torch.unsqueeze(a2.clone().detach(), 0).to(self.device)
            a3 = torch.unsqueeze(a3.clone().detach(), 0).to(self.device)
            a = [a1,a2,a3]
            a_next1 = torch.unsqueeze(a_next1.clone().detach(), 0).to(self.device)
            a_next2 = torch.unsqueeze(a_next2.clone().detach(), 0).to(self.device)
            a_next3 = torch.unsqueeze(a_next3.clone().detach(), 0).to(self.device)
            a_next = [a_next1,a_next2,a_next3]
            r = torch.unsqueeze(torch.tensor(r, dtype=torch.float), 0).to(self.device)
            ad_reward = torch.unsqueeze(torch.tensor(ad_reward, dtype=torch.float), 0).to(self.device)
            tid = torch.unsqueeze(torch.tensor([tid]).clone().detach(), 0).to(self.device)
            done = torch.unsqueeze(done.clone().detach(), 0).to(self.device)
        else:
            s = torch.stack(list(s), dim=0).clone().detach().to(self.device)
            r = torch.stack(list(torch.tensor(r, dtype=torch.float))).clone().detach().to(self.device)
            ad_reward = torch.stack(list(torch.tensor(ad_reward, dtype=torch.float))).clone().detach().to(self.device)
            s_next = torch.stack(list(s_next), dim=0).clone().detach().to(self.device)
            a1 = torch.stack(list(a1), dim=0).clone().detach().to(self.device)
            a2 = torch.stack(list(a2), dim=0).clone().detach().to(self.device)
            a3 = torch.stack(list(a3), dim=0).clone().detach().to(self.device)
            a = [a1.squeeze(1),a2.squeeze(1),a3.squeeze(1)]
            a_next1 = torch.stack(list(a_next1), dim=0).clone().detach().to(self.device)
            a_next2 = torch.stack(list(a_next2), dim=0).clone().detach().to(self.device)
            a_next3 = torch.stack(list(a_next3), dim=0).clone().detach().to(self.device)
            a_next = [a_next1.squeeze(1),a_next2.squeeze(1),a_next3.squeeze(1)]
            tid = torch.stack(list(torch.tensor([tid])), dim=0).clone().detach().to(self.device)
            done = torch.stack(list(done), dim=0).clone().detach().to(self.device)

        # Note: Q learning part
        Q_target = self.vF.Q_value(self.eps,self.actor,self.target_actor,self.critic_1, self.critic_2, self.target_critic_1, self.target_critic_2, s, a, r, s_next, a_next, done, tid, ad_reward)
        # state_a = self.net.state_dict().__str__()
        #single_head_til = 5
        Q_current_1 = self.critic_1(s.detach(), a, tid.detach())
        Q_current_2 = self.critic_2(s.detach(), a, tid.detach())
        self.optimizer_q_value_critic_1.zero_grad()
        l_1 = 0
        # if self.total_counter % single_head_til == 0:
        #     chosen_action_Q = Q_current_1[self.i_s].gather(1, torch.argmax(a[self.i_s],dim=-1).unsqueeze(1))
        #     l_1 += self.loss_fn( chosen_action_Q,Q_target[self.i_s])
        # else:
        for i in range(0,3):
            self.i_s = i
            chosen_action_Q = Q_current_1[self.i_s].gather(1, torch.argmax(a[self.i_s], dim=-1).unsqueeze(1))
            l_1 += self.loss_fn( chosen_action_Q,Q_target[self.i_s])
        l_1.backward()
        self.optimizer_q_value_critic_1.step()

        self.optimizer_q_value_critic_2.zero_grad()
        l_2 = 0.
        # if self.total_counter % single_head_til == 0:
        #     chosen_action_Q = Q_current_2[self.i_s].gather(1, torch.argmax(a[self.i_s], dim=-1).unsqueeze(1))
        #     l_2 += self.loss_fn(chosen_action_Q,Q_target[self.i_s])
        # else:
        for i in range(0, 3):
            self.i_s = i
            chosen_action_Q = Q_current_2[self.i_s].gather(1, torch.argmax(a[self.i_s], dim=-1).unsqueeze(1))
            l_2 += self.loss_fn(chosen_action_Q,Q_target[self.i_s])
        l_2.backward()
        self.optimizer_q_value_critic_2.step()

        v_policy = self.actor(s.detach(), tid.detach())
        Qvalue = self.critic_1(s.detach(), v_policy, tid)
        self.optimizer_actor_policy_gradient.zero_grad()
        l_a = 0.
        # if self.total_counter % single_head_til == 0:
        #     chosen_action_log_prob = torch.log(v_policy[self.i_s].gather(1, torch.argmax(a[self.i_s],dim=-1).unsqueeze(1)) + 1e-8)
        #     chosen_action_Q = Qvalue[self.i_s].gather(1, torch.argmax(a[self.i_s],dim=-1).unsqueeze(1))
        #     l_a += -torch.mean(chosen_action_log_prob*chosen_action_Q) # Note: Maximising Q_value of the policy
        # else:
        for i in range(0, 3):
            self.i_s = i
            chosen_action_log_prob = torch.log(v_policy[self.i_s].gather(1, torch.argmax(a[self.i_s], dim=-1).unsqueeze(1)) + 1e-8)
            chosen_action_Q = Qvalue[self.i_s].gather(1, torch.argmax(a[self.i_s], dim=-1).unsqueeze(1))
            l_a += -torch.mean(chosen_action_log_prob*chosen_action_Q) # Note: Maximising Q_value of the policy
        l_a.backward()
        self.optimizer_actor_policy_gradient.step()

        td_errors = sum(torch.abs(Q_target[i] - Q_current_1[i]).detach() + torch.abs(Q_target[i] - Q_current_2[i]).detach()+ (1e-8 if i == 2 else 0) for i in range(3))
        if td_errors.shape[0] == 1:
            td_errors = torch.mean(td_errors, dim=-1)
            experience = self.memory[-1]
            updated_experience = (*experience[:-1], td_errors)
            self.memory[-1] = updated_experience
        self.vF.soft_update(self.critic_1, self.target_critic_1)
        self.vF.soft_update(self.critic_2, self.target_critic_2)
        self.vF.soft_update(self.actor, self.target_actor)
        ###### COMPUTATIONAL GRAPH GENERATION ######
        # dot = make_dot(prediction, params=dict(self.net.named_parameters()))
        # dot.render(directory='doctest-output', view=True)
        return l_1.item() +l_2.item()+ l_a.item(),tid

    def take_action(self, s,task_id, step_counter, dataset, game):
        _, _, body_part = dataset.decode_input(s)
        actions, is_random_next = self.chooseAction(s,task_id, dataset, game)
        if is_random_next == 1:
            self.no_of_guesses += 1
            a1 = game.agent.value2action(actions[0], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
            a2 = game.agent.value2action(actions[1], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
            a3 = game.agent.value2action(actions[2], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
            a1 = torch.unsqueeze(a1, dim=0)
            a2 = torch.unsqueeze(a2, dim=0)
            a3 = torch.unsqueeze(a3, dim=0)
            a = [a1, a2, a3]
        else:
            a = actions
        a_value1 = game.agent.action2value(a[0], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
        a_value2 = game.agent.action2value(a[1], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
        a_value3 = game.agent.action2value(a[2], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
        a_value = [a_value1, a_value2, a_value3]
        return a, a_value, body_part

    def take_next_action(self, s_next,task_id, action, step_counter, dataset, game):
        _, _, body_part = dataset.decode_input(s_next)
        actions, is_random_next = self.chooseNextAction(s_next,task_id, dataset, game)
        if is_random_next == 1:
            self.no_of_guesses += 1
            a1 = game.agent.value2action(actions[0], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
            a2 = game.agent.value2action(actions[1], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
            a3 = game.agent.value2action(actions[2], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
            a1 = torch.unsqueeze(a1, dim=0)
            a2 = torch.unsqueeze(a2, dim=0)
            a3 = torch.unsqueeze(a3, dim=0)
            a = [a1,a2,a3]
        else:
            a = actions
        a_value1 = game.agent.action2value(a[0], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
        a_value2 = game.agent.action2value(a[1], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
        a_value3 = game.agent.action2value(a[2], self.actor.no_of_actions, game.lower_limit, game.upper_limit)
        a_value = [a_value1, a_value2, a_value3]
        return a, a_value, body_part

    def get_state(self, step_counter, dataset):
        turn = self.int2binary(step_counter).to(self.device)
        patient = dataset.create_input_set().to(self.device)
        if step_counter >= 8:
            done = torch.tensor([1.]).to(self.device)
            game_over = True
        else:
            done = torch.tensor([0.]).to(self.device)
            game_over = False

        s = torch.cat((patient, turn, done), dim=0)
        input_state = torch.cat((s, s), dim=0)
        input_state +=input_state*torch.rand_like(input_state)*(1e-2/self.vF.epsilon)#*self.eps
        # print(s)
        return input_state, done, game_over

    def chooseAction(self, state,task_id, dataset, game):
        is_random = 0
        explore_coef = self.vF.epsilon
        hair_type, skin_type, _ = dataset.decode_input(state)
        if self.total_counter %9*25 == 0:
            self.eps = (1e-6 + 0.999 * np.exp(-1.3e-2 * self.total_counter))
        if game.cycle > game.game_cycles * 0.95:
            self.eps = 0.
        if  np.random.uniform(0,self.eps) > explore_coef:
            a1 = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            a2 = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            a3 = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            actions = [a1, a2, a3]
            is_random = 1
        else:
            state = state.to(self.device)
            actions = self.actor(state, task_id)
        return actions, is_random

    def chooseNextAction(self, state_next,task_id, dataset, game):
        is_random = 0
        explore_coef = self.vF.epsilon
        hair_type, skin_type, _ = dataset.decode_input(state_next)
        if self.total_counter % 9*25 == 0:
            self.eps = (1e-6 + 0.999 * np.exp(-1.3e-2 * self.total_counter))
        if game.cycle > game.game_cycles * 0.95:
            self.eps = 0.
        if  np.random.uniform(0,self.eps) > explore_coef:
            a1 = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            a2 = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            a3 = torch.tensor(np.random.uniform(game.lower_limit, game.upper_limit))
            actions = [a1, a2, a3]
            is_random = 1
        else:
            state_next = state_next.to(self.device)
            actions = self.actor(state_next, task_id)
        return actions, is_random

    def action2value(self, action, num_of_actions, lower_limit, upper_limit):
        action_space = torch.arange(lower_limit, upper_limit, (upper_limit - lower_limit) / num_of_actions)
        max_value_ind = torch.argmax(action).int()
        return action_space[max_value_ind]

    def value2action(self, action_value, num_of_actions, lower_limit, upper_limit):
        action_space = torch.arange(lower_limit, upper_limit, (upper_limit - lower_limit) / num_of_actions).to(self.device)
        idx = torch.argmin(torch.abs(action_space - action_value)).int()
        action_space = torch.zeros_like(action_space).to(self.device)
        action_space[idx] = 1.
        return action_space

    def checkReward(self, head_rewards,reward, action_value, state, dataset, step_counter, game, lower_limit, upper_limit, std):
        hair_type, skin_type, body_part = dataset.decode_input(state)
        r_h0,r_h1,r_h2 = head_rewards
        if game.task_id == 0:
            ref_value_s1 = torch.sum(dataset.kj_total[0][step_counter][body_part]) / 2
            ref_value_s2 = torch.sum(dataset.kj_total[1][step_counter][body_part]) / 2
            ref_value_s3 = torch.sum(dataset.kj_total[2][step_counter][body_part]) / 2
        elif game.task_id == 1:
            ref_value_s1 = torch.sum(dataset.hz[0][step_counter][body_part]) / 2
            ref_value_s2 = torch.sum(dataset.hz[1][step_counter][body_part]) / 2
            ref_value_s3 = torch.sum(dataset.hz[2][step_counter][body_part]) / 2
        else:
            ref_value_s1 = torch.sum(dataset.j_cm2[0][step_counter][body_part]) / 2
            ref_value_s2 = torch.sum(dataset.j_cm2[1][step_counter][body_part]) / 2
            ref_value_s3 = torch.sum(dataset.j_cm2[2][step_counter][body_part]) / 2

        ref_value_min_s1 = ref_value_s1 - ref_value_s1 * std / 100
        ref_value_max_s1 = ref_value_s1 + ref_value_s1 * std / 100
        ref_value_min_s2 = ref_value_s2 - ref_value_s2 * std / 100
        ref_value_max_s2 = ref_value_s2 + ref_value_s2 * std / 100
        ref_value_min_s3 = ref_value_s3 - ref_value_s3 * std / 100
        ref_value_max_s3 = ref_value_s3 + ref_value_s3 * std / 100
        no_steps = 1000
        strength_coeff_r = 1.
        strength_coeff_p = 1.
        additional_reward = 0.
        reward_table_0 = torch.linspace(0.1, 1, no_steps//2).to(self.device)
        reward_table_1 = torch.linspace(1, 0.1, no_steps//2).to(self.device)
        #reward_table_0 = (torch.exp(reward_table_0 * strength_coeff_r) - 1) / (torch.exp(torch.tensor(strength_coeff_r).to(self.device)) - 1)
        #reward_table_1 = (torch.exp(reward_table_1 * strength_coeff_r) - 1) / (torch.exp(torch.tensor(strength_coeff_r).to(self.device)) - 1)
        reward_table = torch.cat([reward_table_0, reward_table_1],dim=0)
        punishment_table_0 = torch.linspace(0.3, 0.3, no_steps // 2).to(self.device)
        punishment_table_1 = torch.linspace(0.1, 0.9, no_steps // 2).to(self.device)
        #punishment_table_0 = (torch.exp(punishment_table_0 * strength_coeff_p) - 1) / (torch.exp(torch.tensor(strength_coeff_p).to(self.device)) - 1)
        #punishment_table_1 = (torch.exp(punishment_table_1 * strength_coeff_p) - 1) / (torch.exp(torch.tensor(strength_coeff_p).to(self.device)) - 1)
        punishment_table = torch.cat([punishment_table_0, punishment_table_1], dim=0)
        if ref_value_min_s1 < action_value[0].item() < ref_value_max_s1:
            distance_vals = torch.linspace(ref_value_min_s1,ref_value_max_s1, no_steps)
            distance_vals = torch.abs(distance_vals/action_value[0] - 1)
            r_idx = torch.argmin(distance_vals)
            reward += reward_table[r_idx].item()
            r_h0 += reward_table[r_idx].item()
        else:
            distance_vals_min = torch.linspace(game.lower_limit, ref_value_min_s1, no_steps // 2)
            distance_vals_max = torch.linspace(ref_value_max_s1, game.upper_limit, no_steps // 2)
            distance_vals = torch.cat([distance_vals_min, distance_vals_max])
            distance_vals = torch.abs(distance_vals / action_value[0] - 1)
            p_idx = torch.argmin(distance_vals)
            additional_reward -= punishment_table[p_idx].item()
        if ref_value_min_s2 < action_value[1].item() < ref_value_max_s2:
            distance_vals = torch.linspace(ref_value_min_s2, ref_value_max_s2, no_steps)
            distance_vals = torch.abs(distance_vals / action_value[1] - 1)
            r_idx = torch.argmin(distance_vals)
            reward += reward_table[r_idx].item()
            r_h1 += reward_table[r_idx].item()
        else:
            distance_vals_min = torch.linspace(game.lower_limit, ref_value_min_s2, no_steps // 2)
            distance_vals_max = torch.linspace(ref_value_max_s2, game.upper_limit, no_steps // 2)
            distance_vals = torch.cat([distance_vals_min, distance_vals_max])
            distance_vals = torch.abs(distance_vals / action_value[1] - 1)
            p_idx = torch.argmin(distance_vals)
            additional_reward -= punishment_table[p_idx].item()
        if ref_value_min_s3 < action_value[2].item() < ref_value_max_s3:
            distance_vals = torch.linspace(ref_value_min_s3, ref_value_max_s3, no_steps)
            distance_vals = torch.abs(distance_vals / action_value[2] - 1)
            r_idx = torch.argmin(distance_vals)
            reward += reward_table[r_idx].item()
            r_h2 += reward_table[r_idx].item()
        else:
            distance_vals_min = torch.linspace(game.lower_limit, ref_value_min_s3, no_steps // 2)
            distance_vals_max = torch.linspace(ref_value_max_s3, game.upper_limit, no_steps // 2)
            distance_vals = torch.cat([distance_vals_min, distance_vals_max])
            distance_vals = torch.abs(distance_vals / action_value[2] - 1)
            p_idx = torch.argmin(distance_vals)
            additional_reward -= punishment_table[p_idx].item()

        if step_counter == 8 and reward*0.34 >= 5:
            additional_reward = 1.
        if step_counter == 8 and reward*0.34 >= 6:
            additional_reward = 2.5
        if step_counter == 8 and reward*0.34 >= 7:
            additional_reward = 5.
        if step_counter == 8 and reward*0.34 >= 8:
            additional_reward = 10.
        if step_counter == 8 and reward*0.34 >= 9:
            additional_reward = 25.
        self.i_s = random.randint(0, 2)
        idx = int(argmin([r_h0,r_h1,r_h2]))
        self.i_s = idx
        return reward, additional_reward,[r_h0,r_h1,r_h2]

    def int2binary(self, step_counter):
        bin = list(map(float, f'{step_counter:04b}'))
        binT = torch.FloatTensor(bin)
        binT.requires_grad = True
        return binT
