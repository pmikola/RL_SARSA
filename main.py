# SARSA(on-policy TD-LAMBDA) FOR AUTOMATIC LASER PARAMETERS SETTING
import random

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import optim

from agent import Agent
from dataset import DataSet
from game import Game
from valueFunction import ValueFunction
from neuralNetwork import NeuralNetwork

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# x = dataSet.decode_input(dataSet.create_input_set())
# y = dataSet.create_target(15)
no_of_actions = 256
no_of_states = 14
alpha = 0.001
epsilon = 0.1
gamma = 0.99
tau = 0.01

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DataSet(device)
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
net = NeuralNetwork(no_of_actions, no_of_states)
net2 = NeuralNetwork(no_of_actions, no_of_states)
net.to(device).requires_grad_(True)
net2.to(device).requires_grad_(True)
valueFunc = ValueFunction(alpha, epsilon, gamma, tau, device)
agent = Agent(net, net2, valueFunc, device)
game = Game(valueFunc, agent, device)
game.game_cycles = 10
cmap = plt.cm.get_cmap('hsv', game.game_cycles + 5)
r_data = []
l_data = []
c_map_data = []

for i in range(0, game.game_cycles):
    rewards = game.playntrain(game, dataset)
    # game.net = net
    print("GAME CYCLE : ", i, "\n", "REWARDS TOTAL : ", sum(rewards), "No. of RANDOM GUESSES: ",
          game.agent.no_of_guesses)
    r_data.append(rewards)
    l_data.append("Epoch: " + str(i))
    c_map_data.append(cmap(i))

plt.hist(r_data, alpha=0.8, stacked=False, histtype='bar', label=l_data, color=c_map_data)

plt.legend(prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=3, fancybox=True, shadow=True)
# plt.legend(prop={'size': 10}, loc='upper center', bbox_to_anchor=(0.5, -0.05),
#           fancybox=True, shadow=True, ncol=5)
plt.xlabel("Reward value per game")
plt.ylabel("No. of rewards")
plt.grid()
plt.show()
