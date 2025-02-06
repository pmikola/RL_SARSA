# SARSA(on-policy TD-LAMBDA) FOR AUTOMATIC LASER PARAMETERS SETTING
import random
import time

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from numpy import argmin
from torch import optim

from agent import Agent
from dataset import DataSet
from game import Game
from estimators import Estimators
from neuralNetwork import NeuralNetwork_SA, NeuralNetwork_S

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# x = dataSet.decode_input(dataSet.create_input_set())
# y = dataSet.create_target(15)
no_of_actions = 256
num_e_bits = 5
num_m_bits = 10
no_of_states = 14  # + num_e_bits + num_m_bits
alpha = 1.
epsilon = 1e-2
gamma = 0.99
tau = 0.01
no_of_games = 50
no_of_rounds = 9
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE: ", device)
time.sleep(1)
dataset = DataSet(device)
torch.manual_seed(2023)
random.seed(2023)
np.random.seed(2023)
actor = NeuralNetwork_S(no_of_actions, no_of_states, device)

target_critic_1 = NeuralNetwork_SA(no_of_actions, no_of_states, device)
target_critic_2 = NeuralNetwork_SA(no_of_actions, no_of_states, device)
critic_1 = NeuralNetwork_SA(no_of_actions, no_of_states, device)
critic_2 = NeuralNetwork_SA(no_of_actions, no_of_states, device)
critic_1.task_indicator = actor.task_indicator
critic_2.task_indicator = actor.task_indicator
target_critic_1.task_indicator = actor.task_indicator
target_critic_2.task_indicator = actor.task_indicator

actor.to(device)
target_critic_1.to(device)
target_critic_2.to(device)

critic_1.to(device)
critic_2.to(device)
valueFunc = Estimators(alpha, epsilon, gamma, tau, device, no_of_actions, v_min=-no_of_games * no_of_rounds,
                       v_max=no_of_games * no_of_rounds)
agent = Agent(actor, critic_1, critic_2, target_critic_1,target_critic_2, valueFunc, no_of_states, num_e_bits, num_m_bits, device)
agent.BATCH_SIZE = 128

game = Game(valueFunc, agent, device, no_of_rounds)
game.game_cycles = 100
game.games = no_of_games
cmap = plt.cm.get_cmap('hsv', game.game_cycles + 5)
r_data = []
a1_data = []
a2_data = []
a3_data = []

l_data = []
loss = []
c_map_data = []
total_time = 0.
ax = plt.figure().gca()
r = [0,0,0]
for i in range(1, game.game_cycles + 1):
    game.cycle = i
    start = time.time()
    print("GAME CYCLE : ", i)
    rewards, a_val, losses = game.playntrain(game, dataset, games=no_of_games)
    R = sum(rewards)
    print("  REWARDS TOTAL : ",R, " ||  RANDOM GUESSES: ",
          game.agent.no_of_guesses)
    end = time.time()
    t = end - start
    total_time += t
    print("  Elapsed time : ", t, " [s]")
    print("----------------------------------")
    if game.task_id == 0:
        r[0] = R
    elif game.task_id == 1:
        r[1] = R
    else:
        r[2] = R

    game.total_counter = 0
    game.agent.vF.epsilon+=(1/(game.game_cycles*2))
    game.task_id = random.randint(0,2)
    if game.cycle % 3 == 0:
        game.task_id = int(argmin(r))
    # Note: test  network with learning on
    if game.cycle >= game.game_cycles - 3:
        game.total_counter = 1e10
        game.task_id = 0
    if game.cycle >= game.game_cycles - 2:
        game.total_counter = 1e10
        game.task_id = 1
    if game.cycle >= game.game_cycles - 1:
        game.total_counter = 1e10
        game.task_id = 2

    r_data.append(rewards)
    a1_data.append(a_val[0])
    a2_data.append(a_val[1])
    a3_data.append(a_val[2])

    l_data.append("Epoch: " + str(i))
    c_map_data.append(cmap(i))
    loss.append(losses)

torch.save(agent.actor.state_dict(), 'model.pt')
print("Total time : ", total_time / 60, "[min]")

ax.hist(r_data, alpha=0.65, stacked=False, histtype='bar', label=l_data, color=c_map_data)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
# ax.legend(prop={'size': 6}, loc='upper center', bbox_to_anchor=(0.5, 1.14),
#           ncol=9, fancybox=True, shadow=True)

plt.xlabel("Reward value per game")
plt.ylabel("No. of games")
plt.grid()
plt.show()

ay = plt.figure().gca()

ay.hist(a1_data, density=True, bins=40, alpha=0.4, stacked=True, histtype='stepfilled')#, label=l_data, color=c_map_data)
ay.hist(a2_data, density=True, bins=40, alpha=0.4, stacked=True, histtype='stepfilled')#, label=l_data, color=c_map_data)
ay.hist(a3_data, density=True, bins=40, alpha=0.4, stacked=True, histtype='stepfilled')#, label=l_data, color=c_map_data)

ay.legend(prop={'size': 6}, loc='upper center', bbox_to_anchor=(0.5, 1.14),
          ncol=9, fancybox=True, shadow=True)

plt.xlabel("Values per game")
plt.ylabel("No. of games")
plt.grid()
plt.show()

az = plt.figure().gca()
rewars_total = np.array(sum(r_data, [])) / no_of_rounds
az.plot(rewars_total)

plt.xlabel("No. Games")
plt.ylabel("Rewards")
plt.grid()
plt.show()

b = plt.figure().gca()
loss_total = np.array(sum(loss, [])) / no_of_rounds
b.plot(loss_total)

plt.xlabel("No. Games")
plt.ylabel("loss")
plt.grid()
plt.show()
