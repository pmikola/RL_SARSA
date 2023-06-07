import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# POLICY ARCHITECTURE
from torch import Tensor


class NeuralNetwork(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(NeuralNetwork, self).__init__()
        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device
        self.hidden = self.initHidden()

        self.cx1_1 = nn.Linear(self.no_of_states * 4 + 2, self.no_of_states * 4, bias=True)
        self.cx1_2 = nn.Linear(self.no_of_states * 4, 1, bias=False)
        self.cx2_1 = nn.Linear(self.no_of_states * 4 + 2, self.no_of_states * 4, bias=True)
        self.cx2_2 = nn.Linear(self.no_of_states * 4, 1, bias=False)
        self.cx3_1 = nn.Linear(self.no_of_states * 4 + 2, self.no_of_states * 4, bias=True)
        self.cx3_2 = nn.Linear(self.no_of_states * 4, 1, bias=False)

        self.linear1 = nn.Linear(self.no_of_states * 2, self.hidden_size * 2, bias=True)
        self.linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
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

    def kWTA(self, input_vector, no_neurons, k_winers):
        # kWTA = torch.full((input_vector.shape[0], no_neurons),0.01).to(self.device) # Inhibition  <- clipping weights
        # kWTA = torch.randn(input_vector.shape[0], no_neurons).to(self.device) # Inhibition
        kWTA = torch.FloatTensor(input_vector.shape[0], no_neurons).uniform_(0., 0.25).to(self.device)  # Best
        # Inhibition so far <- uniform clipping weights in range
        for i in range(0, input_vector.shape[0]):
            k = k_winers[i] * no_neurons
            top_k_val, top_k_ind = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
            # kWTA[i] = input_vector[i]*0.5 # Inhibition with divisor
            kWTA[i][top_k_ind] = top_k_val  # Unchanged top-k values for gradient calculation (backprop)
        return kWTA

    def forward(self, input, task_indicator):
        # x = torch.cat((x,x))
        # x = torch.flatten(x)

        if input.dim() < 2:
            input = torch.unsqueeze(input, dim=0)
        # print(input_vector.dim())
        ti = torch.unsqueeze(task_indicator, dim=0)
        ti = torch.cat([ti] * input.shape[0])
        context_input = torch.cat((input, ti,input, ti), dim=1)
        cx1 = torch.tanh(self.cx1_1(context_input))
        cx1 = F.sigmoid(self.cx1_2(cx1))
        cx2 = torch.tanh(self.cx2_1(context_input))
        cx2 = F.sigmoid(self.cx2_2(cx2))
        cx3 = torch.tanh(self.cx3_1(context_input))
        cx3 = F.sigmoid(self.cx3_2(cx3))

        x = self.linear1(input)
        # x = torch.tanh(x)
        x = self.kWTA(x, self.hidden_size * 2, cx1)
        x = self.linear2(x)
        # print()
        # x = torch.tanh(x)
        # x = Tensor.topk(x, 100)
        #        print(x.shape[0], x.shape[1])
        x = self.kWTA(x, self.hidden_size, cx2)
        x = torch.dropout(x, 0.0, train=True)

        # x,next_hidden = self.rnn1(x,hidden)
        x = self.linear3(x)
        # x = torch.tanh(x)
        x = self.kWTA(x, self.no_of_heads, cx3)
        x = torch.dropout(x, 0.0, train=True)
        x = self.linear4(x)


        output = x
        return output
