import time

import numpy
import numpy as np
import torch
from torch import nn


class Game:
    def __init__(self, valueFunction, agent, device, no_of_rounds):

        self.pain = None
        self.pain_p = None
        self.n_samples = None
        self.ad_reward = None
        self.reward = 0.
        self.laser_params = None
        self.patient = None
        self.game = None
        self.valueFunction = valueFunction
        self.agent = agent
        self.device = device
        self.dataset = None
        self.std = 15
        self.lower_limit_kj = 0.
        self.upper_limit_kj = 30.
        self.lower_limit_hz = 0.
        self.upper_limit_hz = 15.
        self.lower_limit_j = 0.
        self.upper_limit_j = 28.
        self.lower_limit = 0.
        self.upper_limit = 0.
        self.game_over = False
        self.task_id = 0.
        self.total_counter = 0.
        self.number_of_treatments = no_of_rounds
        self.game_cycles = None
        self.games = None
        self.cycle = None

    def reset(self, dataset):
        pass

    def playntrain(self, game, dataset, games=100):
        self.game = game
        self.game.agent.no_of_guesses = 0.
        self.agent.net.train()
        self.games = games

        # self.agent.net2.train()
        self.dataset = dataset
        rewards = []
        a_val = []
        losses = []
        # self.total_counter = 0.
        self.task_id = game.task_id
        if self.task_id == 0.:
            self.lower_limit = self.lower_limit_kj
            self.upper_limit = self.upper_limit_kj

            game.agent.task_indicator[1:4] = torch.tensor([1., 0., 0.]).to(self.device)
            print("  SETTING UP KJ", game.agent.task_indicator)
        elif self.task_id == 1.:
            self.lower_limit = self.lower_limit_hz
            self.upper_limit = self.upper_limit_hz
            game.agent.task_indicator[1:4] = torch.tensor([0., 1., 0.]).to(self.device)
            print("  SETTING UP Hz", game.agent.task_indicator)

        else:
            self.lower_limit = self.lower_limit_j
            self.upper_limit = self.upper_limit_j
            game.agent.task_indicator[1:4] = torch.tensor([0., 0., 1.]).to(self.device)
            print("  SETTING UP J", game.agent.task_indicator)
        step_counter = 0

        for k in range(games):
            self.pain_p = np.random.random_sample()
            self.n_samples = 1000
            self.pain = torch.tensor([0., 0., 0., 0., 0., 0., 0., 0., 0.]).to(self.device)
            while True:
                self.s, self.done, self.game_over = self.agent.get_state(self.pain, step_counter, dataset)

                self.a, self.a_value, _ = self.agent.take_action(self.s, step_counter, dataset,
                                                                 game)

                ###### EVALUATE PAIN ########
                pain_level = sum(np.random.binomial(1, self.pain_p, self.n_samples) == 0) / float(self.n_samples)

                self.pain[step_counter] = torch.Tensor(np.array(pain_level)).to(self.device)
                ###### EVALUATE PAIN ########

                self.reward, self.ad_reward,self.pain_p = self.agent.checkRewardAndModBehavior(self.reward, self.a_value, self.s, self.dataset,
                                                                     step_counter,
                                                                     game, self.lower_limit, self.upper_limit,
                                                                     self.std, self.pain,self.pain_p)

                # print("p_lvl : ",pain_level,"p :",self.pain_p,step_counter)
                self.s_next, self.done, self.game_over = self.agent.get_state(self.pain, step_counter, dataset)
                self.a_next, a_val_next, _ = self.agent.take_next_action(self.s_next, self.a, step_counter, dataset,
                                                                         game)

                l = self.agent.train_short_memory(self.s, self.a, self.reward, self.s_next,
                                                  self.a_next, self.game_over,
                                                  self.game.agent.task_indicator, self.ad_reward)
                # remember
                self.game.agent.remember(self.s, self.a, self.reward, self.s_next, self.a_next, self.game_over,
                                         self.game.agent.task_indicator, self.ad_reward)

                step_counter += 1
                self.total_counter += 1
                # self.reward += 1
                # steps_total += 1
                if step_counter >= self.number_of_treatments:
                    step_counter = 0
                    l = self.agent.train_long_memory(self.total_counter)
                    losses.append(l)
                    rewards.append(self.reward)
                    a_val.append(self.a_value)
                    self.reward = 0.
                    break
        return rewards, a_val, losses
