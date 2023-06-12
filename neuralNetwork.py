import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# POLICY ARCHITECTURE
from torch import Tensor


class NeuralNetwork_S(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(NeuralNetwork_S, self).__init__()

        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device

        # self.conv1 = nn.Conv1d(in_channels=self.no_of_states * 2 + 4, out_channels=self.no_of_states * 2 + 4,
        #                        kernel_size=3, padding=1)
        # CONTEXT LAYERS - k-Winner learning (hebbian learning0
        self.input = self.no_of_states * 2 + 4
        self.cx1_1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.cx1_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx2_1 = nn.Linear(self.input, self.hidden_size, bias=True)
        self.cx2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cx3_1 = nn.Linear(self.input, self.no_of_heads, bias=True)
        self.cx3_2 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self.linear1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear3 = nn.Linear(self.hidden_size, self.no_of_heads, bias=True)
        self.linear4 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self._init_weights(nn.Module)
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

    def kWTA(self, input_vector, no_neurons, k_winers):
        # kWTA = torch.full((input_vector.shape[0], no_neurons),0.01).to(self.device) # Inhibition  <- clipping weights
        # kWTA = torch.randn(input_vector.shape[0], no_neurons).to(self.device) # Inhibition
        # kWTA = torch.FloatTensor(input_vector.shape[0], no_neurons).uniform_(-0.01, 0.01).to(self.device)  # ok

        inhibition_coefs = 3.
        kWTA = input_vector / inhibition_coefs  # simple inhibition -> best so far

        # Inhibition so far <- uniform clipping weights in range
        for i in range(0, input_vector.shape[0]):
            # k = k_winers[i] * no_neurons # FOR SINGLE SIGMOID OUTPUT
            k = torch.argmax(k_winers[i])  # FOR SOFTMAX OR DENSE OUTPUT
            # print(k)
            top_k_val_max, top_k_ind_max = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
            # top_k_val_min, top_k_ind_min = torch.topk(input_vector[i], k.int().item(), largest=False, sorted=False)

            # Unchanged top-k values for gradient calculation (backprop)
            kWTA[i][top_k_ind_max] = top_k_val_max

            # zeroing min values and not doing backprop gradient calc of those neuron conections
            # kWTA[i][top_k_ind_min] = 0.
        return kWTA

    def forward(self, state, task_indicator):
        if state.dim() < 2:
            state = torch.unsqueeze(state, dim=0)
        if task_indicator.dim() < 2:
            task_indicator = torch.unsqueeze(task_indicator, dim=0)

        # LIN VS SOFTMAX VS SIGMOID VS RELU ON OUTPUT
        context_input = torch.cat((state, task_indicator), dim=1)

        # print("context state 2", context_input.shape)
        cx1 = torch.tanh(self.cx1_1(context_input))
        cx1 = F.softmax(self.cx1_2(cx1), dim=1)
        cx2 = torch.tanh(self.cx2_1(context_input))
        cx2 = F.softmax(self.cx2_2(cx2), dim=1)
        cx3 = torch.tanh(self.cx3_1(context_input))
        cx3 = F.softmax(self.cx3_2(cx3), dim=1)

        x = self.linear1(context_input)
        # x = torch.tanh(x)
        # x = torch.cat((x, context_input), dim=1)
        x = self.kWTA(x, self.hidden_size * 2, cx1)

        x = self.linear2(x)
        # x = torch.cat((x, context_input), dim=1)
        x = self.kWTA(x, self.hidden_size, cx2)

        #
        x = self.linear3(x)
        x = self.kWTA(x, self.no_of_heads, cx3)

        x = self.linear4(x)
        output = x
        return output


class NeuralNetwork_SA(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(NeuralNetwork_SA, self).__init__()

        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device

        # self.conv1 = nn.Conv1d(in_channels=self.no_of_states * 2 + 4, out_channels=self.no_of_states * 2 + 4,
        #                        kernel_size=3, padding=1)
        # CONTEXT LAYERS - k-Winner learning (hebbian learning0
        self.input = self.no_of_states * 2 + 4 + self.no_of_heads
        self.cx1_1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.cx1_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx2_1 = nn.Linear(self.input, self.hidden_size, bias=True)
        self.cx2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cx3_1 = nn.Linear(self.input, self.no_of_heads, bias=True)
        self.cx3_2 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self.linear1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear3 = nn.Linear(self.hidden_size, self.no_of_heads, bias=True)
        self.linear4 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self._init_weights(nn.Module)
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

    def kWTA(self, input_vector, no_neurons, k_winers):
        # kWTA = torch.full((input_vector.shape[0], no_neurons),0.01).to(self.device) # Inhibition  <- clipping weights
        # kWTA = torch.randn(input_vector.shape[0], no_neurons).to(self.device) # Inhibition
        # kWTA = torch.FloatTensor(input_vector.shape[0], no_neurons).uniform_(-0.01, 0.01).to(self.device)  # ok

        inhibition_coefs = 3.
        kWTA = input_vector / inhibition_coefs  # simple inhibition -> best so far

        # Inhibition so far <- uniform clipping weights in range
        for i in range(0, input_vector.shape[0]):
            # k = k_winers[i] * no_neurons # FOR SINGLE SIGMOID OUTPUT
            k = torch.argmax(k_winers[i])  # FOR SOFTMAX OR DENSE OUTPUT
            # print(k)
            top_k_val_max, top_k_ind_max = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
            # top_k_val_min, top_k_ind_min = torch.topk(input_vector[i], k.int().item(), largest=False, sorted=False)

            # Unchanged top-k values for gradient calculation (backprop)
            kWTA[i][top_k_ind_max] = top_k_val_max

            # zeroing min values and not doing backprop gradient calc of those neuron conections
            # kWTA[i][top_k_ind_min] = 0.
        return kWTA

    def forward(self, state, action, task_indicator):
        action = torch.squeeze(action, dim=0)
        if state.dim() < 2:
            state = torch.unsqueeze(state, dim=0)
        if action.dim() < 2:
            action = torch.unsqueeze(action, dim=0)
        elif action.dim() == 3:
            action = torch.swapaxes(action,0,1)
            action = torch.squeeze(action, dim=0)
        if task_indicator.dim() < 2:
            task_indicator = torch.unsqueeze(task_indicator, dim=0)

        # LIN VS SOFTMAX VS SIGMOID VS RELU ON OUTPUT
        #print(action.shape, state.shape, task_indicator.shape)
        context_input = torch.cat((state, task_indicator), dim=1)
        context_input = torch.cat((context_input,action), dim=1)

        # print("context state 2", context_input.shape)
        cx1 = torch.tanh(self.cx1_1(context_input))
        cx1 = F.softmax(self.cx1_2(cx1), dim=1)
        cx2 = torch.tanh(self.cx2_1(context_input))
        cx2 = F.softmax(self.cx2_2(cx2), dim=1)
        cx3 = torch.tanh(self.cx3_1(context_input))
        cx3 = F.softmax(self.cx3_2(cx3), dim=1)

        x = self.linear1(context_input)
        # x = torch.tanh(x)
        # x = torch.cat((x, context_input), dim=1)
        x = self.kWTA(x, self.hidden_size * 2, cx1)

        x = self.linear2(x)
        # x = torch.cat((x, context_input), dim=1)
        x = self.kWTA(x, self.hidden_size, cx2)

        #
        x = self.linear3(x)
        x = self.kWTA(x, self.no_of_heads, cx3)

        x = self.linear4(x)
        output = x
        return output
