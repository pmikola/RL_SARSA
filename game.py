import time
import numpy as np
import torch
from torch import nn


class Game:
    def __init__(self, valueFunction, agent, device):
        self.reward = 0.
        self.laser_params = None
        self.patient = None
        self.game = None
        self.valueFunction = valueFunction
        self.agent = agent
        self.device = device
        self.dataset = None
        self.std = 15
        self.lower_limit = 0.
        self.upper_limit = 30.
        self.game_over = False
        self.total_counter = 0.
        self.number_of_treatments = 9
        self.game_cycles = None

    def reset(self, dataset):
        pass

    def playntrain(self, game, dataset, rounds=100):
        self.game = game
        self.game.agent.no_of_guesses = 0.
        self.agent.net.train()
        self.rounds = rounds
        #self.agent.net2.train()
        self.dataset = dataset
        rewards = []
        #self.total_counter = 0.
        step_counter = 0
        for k in range(rounds):
            while True:
                self.s, self.done, self.game_over = self.agent.get_state(step_counter, dataset)

                self.a, self.a_value, body_part = self.agent.take_action(self.s, step_counter, dataset,
                                                                         game)
                self.reward = self.agent.checkReward(self.reward, body_part, self.a_value, self.dataset, step_counter,
                                                     self.std)

                self.s_next, self.done, self.game_over = self.agent.get_state(step_counter, dataset)
                self.a_next, _, _ = self.agent.take_action(self.s_next, step_counter, dataset,
                                                           game)

                # train short memory
                self.agent.train_short_memory(self.s, self.a, self.reward, self.s_next, self.a_next, self.game_over)

                # remember
                self.agent.remember(self.s, self.a, self.reward, self.s_next, self.a_next, self.game_over)

                step_counter += 1
                self.total_counter += 1
                # self.reward += 1
                # steps_total += 1
                if step_counter >= 9:
                    step_counter = 0
                    self.agent.train_long_memory()
                    rewards.append(self.reward)
                    self.reward = 0.
                    break

        return rewards
