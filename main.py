# SARSA(on-policy TD-LAMBDA) FOR AUTOMATIC LASER PARAMETERS SETTING
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
no_of_actions = 8
no_of_states = 13
alpha = 0.5
epsilon = 0.2
gamma = 0.95

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DataSet(device)

net = NeuralNetwork(no_of_actions, no_of_states)
net.to(device).requires_grad_(True)
valueFunc = ValueFunction(alpha, epsilon, gamma,device)
agent = Agent(net, valueFunc,device)
game = Game(valueFunc, agent,device)
cmap = plt.cm.get_cmap('hsv', 10)
for i in range(0, 5):
    rewards= game.playntrain(game, dataset)
    #game.net = net
    print("GAME CYCLE : ", i, "\n", "REWARDS TOTAL : ", sum(rewards))
    plt.hist(rewards, alpha=0.5, stacked=True, label=str(i), color=cmap(i))

plt.legend(prop={'size': 10})
plt.show()
