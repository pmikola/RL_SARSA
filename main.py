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
no_of_actions = 64
no_of_states = 14
alpha = 0.1
epsilon = 0.0001
gamma = 0.99

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = DataSet(device)

net = NeuralNetwork(no_of_actions, no_of_states)
net.to(device).requires_grad_(True)
valueFunc = ValueFunction(alpha, epsilon, gamma,device)
agent = Agent(net, valueFunc,device)
game = Game(valueFunc, agent,device)
cmap = plt.cm.get_cmap('hsv', 15)
r_data = []
l_data = []
c_map_data = []
for i in range(0, 10):
    rewards= game.playntrain(game, dataset)
    #game.net = net
    print("GAME CYCLE : ", i, "\n", "REWARDS TOTAL : ", sum(rewards))
    r_data.append(rewards)
    l_data.append("Epoch: "+str(i))
    c_map_data.append(cmap(i))

plt.hist(r_data, alpha=0.8, stacked=False,histtype='bar', label=l_data, color=c_map_data)

plt.legend(prop={'size': 10})
plt.xlabel("Reward value per game")
plt.ylabel("No. of rewards")
plt.show()
