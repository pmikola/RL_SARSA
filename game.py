import time

import numpy as np
import torch
from torch import nn


class Game:
    def __init__(self, valueFunction, agent):
        self.reward = 0.
        self.laser_params = None
        self.patient = None
        self.game = None
        self.valueFunction = valueFunction
        self.agent = agent
        self.dataset = None
        self.std = 15
        self.lower_limit = 0.
        self.upper_limit = 30.

    def init(self, no_of_heads, no_of_states, all_moves, dataset):
        torch.manual_seed(42)
        self.action = torch.zeros((all_moves, no_of_heads))
        self.a = torch.zeros((all_moves, no_of_heads))
        self.state = torch.zeros((all_moves, no_of_states))
        self.Q = torch.zeros((all_moves, no_of_states, no_of_heads))
        # self.Q.requires_grad = False


        step_counter = 0
        for i in range(all_moves):

            # turn.requires_grad = True
            # print(self.game.state[0])
            turn = self.agent.int2binary(step_counter)
            # print(turn)
            self.state[i][-1] = turn[3]
            self.state[i][-2] = turn[2]
            self.state[i][-3] = turn[1]
            self.state[i][-4] = turn[0]
            # print(self.game.state[step_counter], turn)
            self.patient = dataset.create_input_set()

            self.game.state[i][:-4] = self.patient
            step_counter += 1
            if step_counter >= 9:
                step_counter = 0
        #self.state.requires_grad = True



    def reset(self, dataset):
        pass

    def playntrain(self, game, dataset, rounds=200,num_of_treatments=9):
        self.game = game
        self.dataset = dataset
        self.game.init(game.agent.net.no_of_heads, game.agent.net.no_of_states, rounds * num_of_treatments, dataset)
        rewards = []
        steps_total = 0
        for k in range(0, rounds):
            n = steps_total
            step_counter = 0
            # self.game.reset(self.dataset)
            # print(self.game.state[n])
            self.actions, is_random = self.agent.chooseAction((self.game.state[n]), self.dataset)
            _, _, body_part = self.dataset.decode_input(self.game.state[n])
            if is_random == 1:
                self.action = self.actions[step_counter][body_part]
                self.a[n] = game.agent.value2action(self.game, self.action, self.agent.net.no_of_heads,
                                                    self.lower_limit,
                                                    self.upper_limit)
            else:
                self.a[n] = self.actions

            while True:

                # self.actions_in_turn[step_counter] = np.array(self.a)

                # print(self.game.state)
                self.a_value = self.game.agent.action2value(self.game, self.a[n], self.agent.net.no_of_heads,
                                                            self.lower_limit,
                                                            self.upper_limit)
                self.reward = self.agent.checkReward(self.reward, body_part, self.a_value, self.dataset, step_counter,
                                                     self.std)

                self.actions, is_random_next = self.agent.chooseAction(self.game.state[n + 1], self.dataset)
                if is_random_next == 1:
                    self.action = self.actions[step_counter][body_part]
                    self.a[n + 1] = game.agent.value2action(self.game, self.action, self.agent.net.no_of_heads,
                                                            self.lower_limit,
                                                            self.upper_limit)
                    # print(self.a)
                else:
                    # self.a_next_clone = self.actions_next.clone()
                    self.a[n + 1] = self.actions

                # print(self.action)
                # print(self.action_next)
                # print(self.Q[n+1])

                Q_next = game.agent.train(self.Q[n], self.game.state[n], self.a[n], self.reward,
                                                 self.game.state[n + 1],
                                                 self.a[n + 1], self.game)
                self.Q[n + 1] = Q_next.clone()
                # self.reward += self.reward
                step_counter += 1
                self.reward = self.reward + 1
                steps_total+=1
                if step_counter >= 9:
                    rewards.append(self.reward)
                    self.reward = 0.
                    break
        return rewards, self.agent.net
