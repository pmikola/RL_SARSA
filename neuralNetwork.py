import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# POLICY ARCHITECTURE
class NeuralNetwork(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(NeuralNetwork, self).__init__()
        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device
        self.hidden = self.initHidden()




        self.linear1 = nn.Linear(self.no_of_states*2, self.hidden_size*2 , bias=True)

        self.linear2 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        #self.linear2_1 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # self.rnn1 = nn.RNN(self.hidden_size,self.hidden_size, 1)
        self.linear3 = nn.Linear(self.hidden_size, self.no_of_heads, bias=True)
        self.linear4 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=False)

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

    def initHidden(self):
        return torch.zeros(1, self.hidden_size).to(self.device)

    def forward(self, x):
        #x = torch.cat((x,x))
        #x = torch.flatten(x)

        x = torch.tanh(self.linear1(x))
        x = torch.dropout(torch.tanh(self.linear2(x)), 0.0,train=True)
        #x = torch.dropout(torch.tanh(self.linear2_1(x)), 0.0,train=True)

        # x,next_hidden = self.rnn1(x,hidden)
        x = torch.dropout(torch.tanh(self.linear3(x)), 0.0,train=True)
        x = self.linear4(x)
        output = x
        return output
