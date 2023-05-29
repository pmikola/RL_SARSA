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
            nn.Linear(self.no_of_states, 1024, bias=True),
            nn.Tanh(),
            nn.Linear(1024, 512, bias=True),
            nn.Tanh(),
            nn.Linear(512, self.no_of_heads, bias=True),
            nn.Tanh(),
            nn.Linear(self.no_of_heads, self.no_of_heads, bias=False)
        )

        self.apply(self._init_weights)
        for param in self.parameters():
            param.requires_grad = True

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def forward(self, x):
        # x = F.softmax(self.layers(x),dim=0)
        x = self.layers(x)
        return x
