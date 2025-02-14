import time
import numpy as np
import torch
from torch import nn


class Game:
    def __init__(self, valueFunction, agent, device, no_of_rounds):
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
        self.upper_limit = 30.
        self.game_over = False
        self.task_id = 0
        self.wavelength = 0.
        self.total_counter = 0.
        self.number_of_treatments = no_of_rounds
        self.game_cycles = None
        self.games = None
        self.cycle = None

    def reset(self, dataset):
        pass

    def playntrain(self, game, dataset, games=10):
        self.game = game
        self.game.agent.no_of_guesses = 0.
        self.agent.actor.train()
        self.games = games
        # self.agent.net2.train()
        self.dataset = dataset
        rewards = []
        a_val = []
        losses = []
        tids = []
        # self.total_counter = 0.
        self.task_id = game.task_id
        if self.task_id == 0:
            self.lower_limit = self.lower_limit_kj
            self.upper_limit = self.upper_limit_kj
            print("  SETTING UP KJ | Task id:", self.task_id)
        elif self.task_id == 1:
            self.lower_limit = self.lower_limit_hz
            self.upper_limit = self.upper_limit_hz
            print("  SETTING UP Hz | Task id:", self.task_id)
        else:
            self.lower_limit = self.lower_limit_j
            self.upper_limit = self.upper_limit_j
            print("  SETTING UP FLUENCE | Task id: ", self.task_id)
        step_counter = 0
        head_rewards = [0.,0.,0.]
        for k in range(games):
            while True:
                self.agent.total_counter = self.total_counter
                self.s, self.done, self.game_over = self.agent.get_state(step_counter, self.dataset)
                self.a, self.a_value, _ = self.agent.take_action(self.s,self.task_id, step_counter, self.dataset,game)
                self.reward,self.ad_reward ,head_rewards= self.agent.checkReward(head_rewards,self.reward, self.a_value, self.s, self.dataset, step_counter,game, self.lower_limit, self.upper_limit,self.std)
                self.s_next, self.done, self.game_over = self.agent.get_state(step_counter, self.dataset)
                self.a_next, a_val_next, _ = self.agent.take_next_action(self.s_next,self.task_id, self.a, step_counter, self.dataset,game)
                self.game.agent.remember(self.s, self.a[0],self.a[1],self.a[2], self.reward, self.s_next, self.a_next[0],self.a_next[1],self.a_next[2], self.done, self.task_id,self.ad_reward)
                l,tid = self.agent.train_short_memory(self.s, self.a[0],self.a[1],self.a[2], self.reward, self.s_next,self.a_next[0],self.a_next[1],self.a_next[2], self.done,self.task_id,self.ad_reward)
                step_counter += 1
                self.total_counter += 1
                losses.append(l)
                tids.append(tid)
                rewards.append(self.reward)
                a_val.append(self.a_value)
                # self.reward += 1
                # steps_total += 1
                if step_counter >= self.number_of_treatments:
                    step_counter = 0
                    _,_ = self.agent.train_long_memory(self.total_counter)
                    self.reward = 0.
                    break
        return rewards, a_val,losses,head_rewards,tids
