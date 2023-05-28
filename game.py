import time
import numpy as np
import torch
from torch import nn


class Game:
    def __init__(self, valueFunction, agent,device):
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

    def init(self, no_of_heads, no_of_states, all_moves, dataset):
        # torch.manual_seed(42)
        self.action = torch.zeros((all_moves, no_of_heads), requires_grad=True).to(self.device)
        self.a = torch.zeros((all_moves, no_of_heads), requires_grad=True).to(self.device)
        self.state = torch.zeros((all_moves, no_of_states), requires_grad=True).to(self.device)
        self.Q = torch.zeros((all_moves, no_of_states, no_of_heads), requires_grad=True).to(self.device)

        step_counter = 0
        for i in range(all_moves):
            turn = self.agent.int2binary(step_counter)
            self.state[i][-1].data.copy_(turn[3].detach())
            self.state[i][-1].requires_grad_(True)
            self.state[i][-2].data.copy_(turn[2].detach())
            self.state[i][-2].requires_grad_(True)
            self.state[i][-3].data.copy_(turn[1].detach())
            self.state[i][-3].requires_grad_(True)
            self.state[i][-4].data.copy_(turn[0].detach())
            self.state[i][-4].requires_grad_(True)

            self.patient = dataset.create_input_set()
            self.state[i][:-4].data.copy_(self.patient.detach())
            self.state[i][:-4].requires_grad_(True)

            step_counter += 1
            if step_counter >= 9:
                step_counter = 0

    def reset(self, dataset):
        pass

    def playntrain(self, game, dataset, rounds=200, num_of_treatments=9):
        self.game = game
        self.agent.net.train()
        self.dataset = dataset
        self.game.init(game.agent.net.no_of_heads, game.agent.net.no_of_states, rounds * num_of_treatments, dataset)
        rewards = []
        steps_total = 0
        for k in range(rounds):
            n = steps_total
            step_counter = 0
            self.actions, is_random = self.agent.chooseAction(self.state[n], self.dataset)
            _, _, body_part = self.dataset.decode_input(self.state[n])
            if is_random == 1:
                self.action = self.actions[step_counter][body_part]
                a = game.agent.value2action(self.game, self.action, self.agent.net.no_of_heads,
                                                    self.lower_limit, self.upper_limit)
                self.a[n].data.copy_(a.detach())
                self.a[n].requires_grad_(True)
            else:
                self.a[n].data.copy_(self.actions.detach())
                self.a[n].requires_grad_(True)
                # self.a[n] = self.actions

            while True:
                self.a_value = self.game.agent.action2value(self.game, self.a[n], self.agent.net.no_of_heads,
                                                            self.lower_limit, self.upper_limit)
                self.reward = self.agent.checkReward(self.reward, body_part, self.a_value, self.dataset, step_counter,
                                                     self.std)

                self.actions, is_random_next = self.agent.chooseAction(self.game.state[n + 1], self.dataset)
                if is_random_next == 1:
                    self.action = self.actions[step_counter][body_part]
                    a = game.agent.value2action(self.game, self.action, self.agent.net.no_of_heads,
                                                            self.lower_limit, self.upper_limit)
                    self.a[n + 1].data.copy_(a.detach())
                    self.a[n + 1].requires_grad_(True)
                else:
                    self.a[n + 1].data.copy_(self.actions.detach())
                    self.a[n + 1].requires_grad_(True)
                    # self.a[n + 1] = self.actions


                Q_next = game.agent.train(self.Q[n], self.state[n], self.a[n], self.reward,
                                          self.state[n + 1], self.a[n + 1], self.game)

                self.Q[n + 1].data.copy_(Q_next.detach())
                self.Q[n + 1].requires_grad_(True)
                #self.Q[n + 1] = Q_next.clone().detach()
                step_counter += 1
                self.reward += 1
                steps_total += 1
                if step_counter >= 9:
                    rewards.append(self.reward)
                    self.reward = 0.
                    break
        return rewards