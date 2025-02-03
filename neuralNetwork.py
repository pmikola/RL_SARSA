import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# POLICY ARCHITECTURE
from torch import Tensor

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5


class NeuralNetwork_S(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(NeuralNetwork_S, self).__init__()

        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.device = device
        self.modulation_resolution = 4
        self.modulation_scale = 1
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

        self.modulate1_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.modulate1_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)
        self.modulate2_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.modulate2_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)
        self.modulate3_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.modulate3_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)
        self.modulate4_1 = nn.Linear(4, self.hidden_size * 2, bias=True)
        self.modulate4_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)

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

    def kwta(self, input_vector, k_winers, modulation):
        B, N = input_vector.shape
        k_tensor = torch.argmax(k_winers, dim=1)
        modulation_idx = torch.argmax(modulation, dim=1)
        modulation_strength = ((modulation_idx + 1) / self.modulation_resolution) * self.modulation_scale
        modulated = input_vector * modulation_strength.unsqueeze(1)
        sorted_indices = torch.argsort(input_vector, dim=1, descending=True)
        range_tensor = torch.arange(N, device=input_vector.device).unsqueeze(0).expand(B, N)
        ranks = torch.empty_like(sorted_indices)
        ranks.scatter_(1, sorted_indices, range_tensor)
        winners_mask = (ranks < k_tensor.unsqueeze(1))
        kWTA_forward = winners_mask * input_vector + (1 - winners_mask.float()) * modulated
        kWTA = kWTA_forward + (input_vector - input_vector).detach()
        return kWTA

    # def kwta(self, input_vector, k_winers, modulation):
    #     kWTA = input_vector
    #     for i in range(0, input_vector.shape[0]):
    #         k = torch.argmax(k_winers[i])
    #         top_k_val_max, top_k_ind_max = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
    #         modulation_strength = ((torch.argmax(modulation[i]) + 1) / self.modulation_resolution)*self.modulation_scale
    #         kWTA[i] = kWTA[i] * modulation_strength
    #         kWTA[i][top_k_ind_max] = top_k_val_max
    #     return kWTA

    def forward(self, state, task_indicator):
        if state.dim() < 2:
            state = torch.unsqueeze(state, dim=0)
        if task_indicator.dim() < 2:
            task_indicator = torch.unsqueeze(task_indicator, dim=0)

        # LIN VS SOFTMAX VS SIGMOID VS RELU ON OUTPUT
        #context_input = torch.cat((state, task_indicator), dim=1)

        # print("context state 2", context_input.shape)
        cx1 = torch.tanh(self.cx1_1(task_indicator))
        cx1 = self.cx1_2(cx1)
        cx2 = torch.tanh(self.cx2_1(task_indicator))
        cx2 = self.cx2_2(cx2)
        cx3 = torch.tanh(self.cx3_1(task_indicator))
        cx3 = self.cx3_2(cx3)
        cx4 = torch.tanh(self.cx4_1(task_indicator))
        cx4 = self.cx4_2(cx4)

        modulate1 = torch.tanh(self.modulate1_1(task_indicator))
        modulate1 = F.softmax(self.modulate1_2(modulate1), dim=1)
        modulate2 = torch.tanh(self.modulate2_1(task_indicator))
        modulate2 = F.softmax(self.modulate2_2(modulate2), dim=1)
        modulate3 = torch.tanh(self.modulate3_1(task_indicator))
        modulate3 = F.softmax(self.modulate3_2(modulate3), dim=1)
        modulate4 = torch.tanh(self.modulate4_1(task_indicator))
        modulate4 = F.softmax(self.modulate4_2(modulate4), dim=1)

        x = self.linear1(state)
        x = self.kwta(x, cx1,modulate1)

        x = torch.cat((x, state), dim=1)
        x = self.linear2(x)
        x = self.kwta(x, cx2,modulate2)

        x = torch.cat((x, state), dim=1)
        x = self.linear3(x)
        x = self.kwta(x, cx3,modulate3)

        x = torch.cat((x, state), dim=1)
        x = self.linear4(x)
        x = self.kwta(x, cx4,modulate4)
        x = self.linear5(x)
        output = x
        return output


class NeuralNetwork_SA(nn.Module):
    def __init__(self, no_of_heads, no_of_states, device):
        super(NeuralNetwork_SA, self).__init__()

        self.no_of_heads = no_of_heads
        self.no_of_states = no_of_states
        self.hidden_size = 512
        self.modulation_resolution = 4
        self.modulation_scale = 1
        self.device = device
        self.input = self.no_of_states * 2 + 4 + self.no_of_heads
        self.hidden_state_action = self.input + self.hidden_size
        self.alpha = nn.Parameter(torch.full((1,),1.0), requires_grad=True)
        self.gamma = nn.Parameter(torch.full((1,),0.95), requires_grad=True)
        self.tau = nn.Parameter(torch.full((1,),1e-3), requires_grad=True)
        self.beta= nn.Parameter(torch.full((1,),0.5), requires_grad=True)

        self.modulate1_1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.modulate1_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)
        self.modulate2_1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.modulate2_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)
        self.modulate3_1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.modulate3_2 = nn.Linear(self.hidden_size * 2, self.modulation_resolution, bias=True)


        self.cx1_1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.cx1_2 = nn.Linear(self.hidden_size * 2, self.hidden_size * 2, bias=True)
        self.cx2_1 = nn.Linear(self.input, self.hidden_size, bias=True)
        self.cx2_2 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.cx3_1 = nn.Linear(self.input, self.no_of_heads, bias=True)
        self.cx3_2 = nn.Linear(self.no_of_heads, self.no_of_heads, bias=True)

        self.linear1 = nn.Linear(self.input, self.hidden_size * 2, bias=True)
        self.linear2 = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.linear3 = nn.Linear(self.hidden_size, self.no_of_heads, bias=True)
        self.linear4 = nn.Linear(self.no_of_heads,1, bias=True)

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

    def kwta(self, input_vector, k_winers, modulation):
        B, N = input_vector.shape
        k_tensor = torch.argmax(k_winers, dim=1)
        modulation_idx = torch.argmax(modulation, dim=1)
        modulation_strength = ((modulation_idx + 1) / self.modulation_resolution) * self.modulation_scale
        modulated = input_vector * modulation_strength.unsqueeze(1)
        sorted_indices = torch.argsort(input_vector, dim=1, descending=True)
        range_tensor = torch.arange(N, device=input_vector.device).unsqueeze(0).expand(B, N)
        ranks = torch.empty_like(sorted_indices)
        ranks.scatter_(1, sorted_indices, range_tensor)
        winners_mask = (ranks < k_tensor.unsqueeze(1))
        kWTA_forward = winners_mask * input_vector + (1 - winners_mask.float()) * modulated
        kWTA = kWTA_forward + (input_vector - input_vector).detach()
        return kWTA

    # def kwta(self, input_vector, k_winers, modulation):
    #     kWTA = input_vector
    #     for i in range(0, input_vector.shape[0]):
    #         k = torch.argmax(k_winers[i])
    #         top_k_val_max, top_k_ind_max = torch.topk(input_vector[i], k.int().item(), largest=True, sorted=False)
    #         modulation_strength = ((torch.argmax(modulation[i]) + 1) / self.modulation_resolution)*self.modulation_scale
    #         kWTA[i] = kWTA[i] * modulation_strength
    #         kWTA[i][top_k_ind_max] = top_k_val_max
    #     return kWTA

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
        cx1 = torch.tanh(self.cx1_1(context_input))
        cx1 = F.softmax(self.cx1_2(cx1), dim=1)
        cx2 = torch.tanh(self.cx2_1(context_input))
        cx2 = F.softmax(self.cx2_2(cx2), dim=1)
        cx3 = torch.tanh(self.cx3_1(context_input))
        cx3 = F.softmax(self.cx3_2(cx3), dim=1)

        modulate1 = torch.tanh(self.modulate1_1(context_input))
        modulate1 = F.softmax(self.modulate1_2(modulate1), dim=1)
        modulate2 = torch.tanh(self.modulate2_1(context_input))
        modulate2 = F.softmax(self.modulate2_2(modulate2), dim=1)
        modulate3 = torch.tanh(self.modulate3_1(context_input))
        modulate3 = F.softmax(self.modulate3_2(modulate3), dim=1)

        x = self.linear1(context_input)
        x = self.kwta(x, cx1, modulate1)

        # x = torch.cat((x, context_input), dim=1)
        x = self.linear2(x)
        x = self.kwta(x, cx2, modulate2)

        # x = torch.cat((x, context_input), dim=1)
        x = self.linear3(x)
        x = self.kwta(x, cx3, modulate3)

        x = self.linear4(x)
        output = x
        return output
