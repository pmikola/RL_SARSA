import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# POLICY ARCHITECTURE
class NeuralNetwork(nn.Module):
    def __init__(self, no_of_heads,no_of_states):
        super(NeuralNetwork, self).__init__()
        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.layers = nn.Sequential(
            nn.Linear(self.no_of_states, 512, bias=True),
            nn.ReLU(),
            nn.Linear(512, self.no_of_heads)
        )

    def forward(self, x):
        # x = F.softmax(self.layers(x),dim=0)
        x = self.layers(x)
        return x
