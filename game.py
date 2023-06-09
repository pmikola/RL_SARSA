import time
import numpy as np
import torch
from torch import nn


class Game:
    def __init__(self, valueFunction, agent, device, no_of_rounds):
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
        # self.total_counter = 0.
        self.task_id = game.task_id
        if self.task_id == 0.:
            self.lower_limit = self.lower_limit_kj
            self.upper_limit = self.upper_limit_kj

            game.agent.task_indicator[1:4] = torch.tensor([1., 0., 0.]).to(self.device)
            print("SETTING UP KJ", game.agent.task_indicator)
        elif self.task_id == 1.:
            self.lower_limit = self.lower_limit_hz
            self.upper_limit = self.upper_limit_hz
            game.agent.task_indicator[1:4] = torch.tensor([0., 1., 0.]).to(self.device)
            print("SETTING UP Hz",game.agent.task_indicator)

        else:
            self.lower_limit = self.lower_limit_j
            self.upper_limit = self.upper_limit_j
            game.agent.task_indicator[1:4] = torch.tensor([0., 0., 1.]).to(self.device)
            print("SETTING UP J",game.agent.task_indicator)
        step_counter = 0
        for k in range(games):
            while True:
                self.s, self.done, self.game_over = self.agent.get_state(step_counter, dataset)

                self.a, self.a_value, _ = self.agent.take_action(self.s, step_counter, dataset,
                                                                 game)

                self.reward = self.agent.checkReward(self.reward, self.a_value, self.s, self.dataset, step_counter,
                                                     game, self.lower_limit, self.upper_limit,
                                                     self.std)

                self.s_next, self.done, self.game_over = self.agent.get_state(step_counter, dataset)
                self.a_next, a_val_next, _ = self.agent.take_action(self.s_next, step_counter, dataset,
                                                                    game)

                # print(self.a_value,a_val_next)
                # time.sleep(1)
                # train short memory
                self.agent.train_short_memory(self.s, self.a, self.reward, self.s_next, self.a_next, self.game_over)

                # remember
                self.agent.remember(self.s, self.a, self.reward, self.s_next, self.a_next, self.game_over)

                step_counter += 1
                self.total_counter += 1
                # self.reward += 1
                # steps_total += 1
                if step_counter >= self.number_of_treatments:
                    step_counter = 0
                    self.agent.train_long_memory(self.total_counter)
                    rewards.append(self.reward)
                    a_val.append(self.a_value)
                    self.reward = 0.
                    break
        return rewards, a_val
