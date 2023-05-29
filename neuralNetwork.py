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
            nn.Linear(self.no_of_states, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, 128, bias=True),
            nn.Tanh(),
            nn.Linear(128, self.no_of_heads, bias=False)
        )
        for param in self.parameters():
            param.requires_grad = True

    def forward(self, x):
        # x = F.softmax(self.layers(x),dim=0)
        x = self.layers(x)
        return x
