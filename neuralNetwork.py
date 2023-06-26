import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# POLICY ARCHITECTURE
from torch import Tensor


class Q_Network(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(Q_Network, self).__init__()

        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device

        # self.conv1 = nn.Conv1d(in_channels=self.no_of_states * 2 + 4, out_channels=self.no_of_states * 2 + 4,
        #                        kernel_size=3, padding=1)
        # CONTEXT LAYERS - k-Winner learning (hebbian learning0
        self.input = self.no_of_states * 2
        self.hidden_state = self.input + self.hidden_size

        self.cx1_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.cx1_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx2_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.cx2_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx3_1 = nn.Linear(4, self.hidden_size, bias=True)
        self.cx3_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cx4_1 = nn.Linear(4, self.no_of_heads, bias=True)
        self.cx4_2 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self.inhibit1_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit1_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)
        self.inhibit2_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit2_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)
        self.inhibit3_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit3_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)
        self.inhibit4_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit4_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)

        self.linear1 = nn.Linear(self.input, self.hidden_size * 4, bias=True)
        self.linear2 = nn.Linear(self.hidden_size * 4 + self.input, self.hidden_size * 2, bias=True)
        self.linear3 = nn.Linear(self.hidden_size * 2 + self.input, self.hidden_size * 2, bias=True)
        self.linear4 = nn.Linear(self.hidden_size * 2 + self.input, self.hidden_size, bias=True)
        self.linear5 = nn.Linear(self.hidden_size, self.no_of_heads, bias=True)

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

    def kWTA(self, input_vector, k_winers):  # , inhibiton):
        no_neurons = input_vector.shape[1]

        # kWTA = torch.full((input_vector.shape[0], no_neurons),0.01).to(self.device) # Inhibition  <- clipping weights
        # kWTA = torch.randn(input_vector.shape[0], no_neurons).to(self.device) # Inhibition
        # kWTA = torch.FloatTensor(input_vector.shape[0], no_neurons).uniform_(-1 / inhibition_strength,1 / inhibition_strength).to(self.device)  # ok
        kWTA = torch.FloatTensor(input_vector.shape[0], no_neurons).uniform_(-0.01, 0.01).to(
            self.device)  # ok
        # inhibition_coefs = inhibition_strength
        # kWTA = input_vector
        # Inhibition so far <- uniform clipping weights in range
        for i in range(0, input_vector.shape[0]):
            # k = k_winers[i] * no_neurons # FOR SINGLE SIGMOID OUTPUT
            k = torch.argmax(k_winers[i])  # FOR SOFTMAX OR DENSE OUTPUT
            top_k_val_max, top_k_ind_max = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
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
        # context_input = torch.cat((state, task_indicator), dim=1)

        # print("context state 2", context_input.shape)
        cx1 = torch.tanh(self.cx1_1(task_indicator))
        cx1 = self.cx1_2(cx1)
        cx2 = torch.tanh(self.cx2_1(task_indicator))
        cx2 = self.cx2_2(cx2)
        cx3 = torch.tanh(self.cx3_1(task_indicator))
        cx3 = self.cx3_2(cx3)
        cx4 = torch.tanh(self.cx4_1(task_indicator))
        cx4 = self.cx4_2(cx4)

        # inhibition1 = torch.tanh(self.inhibit1_1(task_indicator))
        # inhibition1 = F.softmax(self.inhibit1_2(inhibition1), dim=1)
        # inhibition2 = torch.tanh(self.inhibit2_1(task_indicator))
        # inhibition2 = F.softmax(self.inhibit2_2(inhibition2), dim=1)
        # inhibition3 = torch.tanh(self.inhibit3_1(task_indicator))
        # inhibition3 = F.softmax(self.inhibit3_2(inhibition3), dim=1)
        # inhibition4 = torch.tanh(self.inhibit4_1(task_indicator))
        # inhibition4 = F.softmax(self.inhibit4_2(inhibition4), dim=1)

        x = self.linear1(state)
        x = self.kWTA(x, cx1)

        x = torch.cat((x, state), dim=1)
        x = self.linear2(x)
        x = self.kWTA(x, cx2)

        x = torch.cat((x, state), dim=1)
        x = self.linear3(x)
        x = self.kWTA(x, cx3)

        x = torch.cat((x, state), dim=1)
        x = self.linear4(x)
        x = self.kWTA(x, cx4)
        x = self.linear5(x)
        output = x
        return output


class Policy_Network(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(Policy_Network, self).__init__()

        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device

        # self.conv1 = nn.Conv1d(in_channels=self.no_of_states * 2 + 4, out_channels=self.no_of_states * 2 + 4,
        #                        kernel_size=3, padding=1)
        # CONTEXT LAYERS - k-Winner learning (hebbian learning0
        self.input = self.no_of_states * 2 + 4 + self.no_of_heads
        self.hidden_state_action = self.input + self.hidden_size

        self.cx1_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.cx1_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx2_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.cx2_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx3_1 = nn.Linear(4, self.hidden_size, bias=True)
        self.cx3_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cx4_1 = nn.Linear(4, self.no_of_heads, bias=True)
        self.cx4_2 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self.inhibit1_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit1_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)
        self.inhibit2_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit2_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)
        self.inhibit3_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit3_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)
        self.inhibit4_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.inhibit4_2 = nn.Linear(self.hidden_size * 2, 4, bias=True)

        self.linear1 = nn.Linear(self.input, self.hidden_size * 4, bias=True)
        self.linear2 = nn.Linear(self.hidden_size * 4 + self.input, self.hidden_size * 2, bias=True)
        self.linear3 = nn.Linear(self.hidden_size * 2 + self.input, self.hidden_size * 2, bias=True)
        self.linear4 = nn.Linear(self.hidden_size * 2 + self.input, self.hidden_size, bias=True)
        self.linear5 = nn.Linear(self.hidden_size, self.no_of_heads, bias=True)

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

    def kWTA(self, input_vector, k_winers):
        no_neurons = input_vector.shape[1]
        # inhibition_strength = torch.argmax(inhibiton).int().item() + 1
        # kWTA = torch.full((input_vector.shape[0], no_neurons),0.01).to(self.device) # Inhibition  <- clipping weights
        # kWTA = torch.randn(input_vector.shape[0], no_neurons).to(self.device) # Inhibition
        # kWTA = torch.FloatTensor(input_vector.shape[0], no_neurons).uniform_(-1 / inhibition_strength,1 / inhibition_strength).to(self.device)  # ok

        # inhibition_coefs = inhibition_strength
        kWTA = input_vector
        # Inhibition so far <- uniform clipping weights in range

        for i in range(0, input_vector.shape[0]):
            # k = k_winers[i] * no_neurons # FOR SINGLE SIGMOID OUTPUT
            k = torch.argmax(k_winers[i])  # FOR SOFTMAX OR DENSE OUTPUT
            # print(k)
            top_k_val_max, top_k_ind_max = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
            # top_k_val_min, top_k_ind_min = torch.topk(input_vector[i], k.int().item(), largest=False, sorted=False)
            # inhibition_strength = torch.argmax(inhibiton[i]).int().item() + 1
            # inhibition_strength = inhibition_strength * 0.5
            # kWTA[i] = kWTA[
            #              i] / inhibition_strength  # torch.FloatTensor(input_vector.shape[1]).uniform_(-1 / inhibition_strength,
            # 1 / inhibition_strength).to(
            # self.device)
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
            action = torch.swapaxes(action, 0, 1)
            action = torch.squeeze(action, dim=0)
        if task_indicator.dim() < 2:
            task_indicator = torch.unsqueeze(task_indicator, dim=0)

        # LIN VS SOFTMAX VS SIGMOID VS RELU ON OUTPUT
        # print(action.shape, state.shape, task_indicator.shape)
        context_input = torch.cat((state, task_indicator), dim=1)
        context_input = torch.cat((context_input, action), dim=1)

        # print("context state 2", context_input.shape)
        cx1 = torch.tanh(self.cx1_1(task_indicator))
        cx1 = self.cx1_2(cx1)
        cx2 = torch.tanh(self.cx2_1(task_indicator))
        cx2 = self.cx2_2(cx2)
        cx3 = torch.tanh(self.cx3_1(task_indicator))
        cx3 = self.cx3_2(cx3)
        cx4 = torch.tanh(self.cx4_1(task_indicator))
        cx4 = self.cx4_2(cx4)

        # inhibition1 = torch.tanh(self.inhibit1_1(task_indicator))
        # inhibition1 = F.softmax(self.inhibit1_2(inhibition1), dim=1)
        # inhibition2 = torch.tanh(self.inhibit2_1(task_indicator))
        # inhibition2 = F.softmax(self.inhibit2_2(inhibition2), dim=1)
        # inhibition3 = torch.tanh(self.inhibit3_1(task_indicator))
        # inhibition3 = F.softmax(self.inhibit3_2(inhibition3), dim=1)
        # inhibition4 = torch.tanh(self.inhibit4_1(task_indicator))
        # inhibition4 = F.softmax(self.inhibit4_2(inhibition4), dim=1)

        x = self.linear1(context_input)
        x = self.kWTA(x, cx1)

        x = torch.cat((x, context_input), dim=1)
        x = self.linear2(x)
        x = self.kWTA(x, cx2)

        x = torch.cat((x, context_input), dim=1)
        x = self.linear3(x)
        x = self.kWTA(x, cx3)

        x = torch.cat((x, context_input), dim=1)
        x = self.linear4(x)
        x = self.kWTA(x, cx4)
        x = self.linear5(x)
        output = x
        return output
