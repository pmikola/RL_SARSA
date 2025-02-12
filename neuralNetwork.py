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
    def __init__(self, no_of_actions, no_of_states, device):
        super(NeuralNetwork_S, self).__init__()
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states
        self.hidden_size = 128
        self.device = device
        self.modulation_resolution = 10
        self.modulation_scale = 2
        self.input = self.no_of_states * 2
        self.hidden_state = self.input + self.hidden_size
        self.act = nn.LeakyReLU(0.1)

        self.linear1 = nn.Linear(self.input, self.hidden_size, bias=True)
        self.linear2 = nn.Linear(self.hidden_size + self.input, self.hidden_size * 2, bias=True)

        self.res_blocks = nn.Sequential(*[ResidualBlock(self.hidden_size * 2, dropout=0.1) for _ in range(10)])
        self.linear4 = nn.Linear(self.hidden_size * 2 + self.input, self.hidden_size, bias=True)

        self.ls_01 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.ls_02 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.ls_03 = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        self.ls_11 = nn.Linear(self.hidden_size, self.hidden_size//2, bias=True)
        self.ls_12 = nn.Linear(self.hidden_size, self.hidden_size//2, bias=True)
        self.ls_13 = nn.Linear(self.hidden_size, self.hidden_size//2, bias=True)

        self.ls1 = nn.Linear(self.hidden_size//2, self.no_of_actions, bias=True)
        self.ls2 = nn.Linear(self.hidden_size//2, self.no_of_actions, bias=True)
        self.ls3 = nn.Linear(self.hidden_size//2, self.no_of_actions, bias=True)

        self.LNorm1 = nn.LayerNorm(self.hidden_size)
        self.LNorm2 = nn.LayerNorm(self.hidden_size * 2)
        self.LNorm4 = nn.LayerNorm(self.hidden_size)
        self.LNorm_ls01 = nn.LayerNorm(self.hidden_size)
        self.LNorm_ls02 = nn.LayerNorm(self.hidden_size)
        self.LNorm_ls03 = nn.LayerNorm(self.hidden_size)
        self.LNorm_ls11 = nn.LayerNorm(self.hidden_size//2)
        self.LNorm_ls12 = nn.LayerNorm(self.hidden_size//2)
        self.LNorm_ls13 = nn.LayerNorm(self.hidden_size//2)
        self.apply(self._init_weights)
        task_indicator_vectors = self.generate_orthogonal_vectors(3, 32)
        self.task_indicator = nn.Parameter(task_indicator_vectors,requires_grad=True)
        self.limits = nn.Parameter(torch.tensor([0.,30.,0.,15.,0.,28.]),requires_grad=False)

    def forward(self, state, t_id,raw_output=False):
        if isinstance(t_id, int):
            t_id = torch.tensor([t_id]).to(self.device)
        t_id = t_id.long() if isinstance(t_id, torch.Tensor) else torch.tensor(t_id, dtype=torch.long)
        task_id = self.task_indicator[t_id].squeeze(1)
        if task_id.dim() == 3:
            task_id = self.task_indicator[t_id].squeeze(0)
        if state.dim() < 2:
            state = torch.unsqueeze(state, dim=0)

        x = self.act(self.linear1(state))
        x = self.LNorm1(x)

        x = torch.cat((x, state), dim=1)
        x = self.act(self.linear2(x))
        x = self.LNorm2(x)

        x = self.res_blocks(x)

        x = torch.cat((x, state), dim=1)
        x = self.act(self.linear4(x))
        x = self.LNorm4(x)

        x1 = self.act(self.ls_01(x))
        x1 = self.LNorm_ls01(x1)
        x1 = self.act(self.ls_11(x1))

        x2 = self.act(self.ls_02(x))
        x2 = self.LNorm_ls02(x2)
        x2 = self.act(self.ls_12(x2))

        x3 = self.act(self.ls_03(x))
        x3 = self.LNorm_ls03(x3)
        x3 = self.act(self.ls_13(x3))

        ls1 = torch.softmax(self.ls1(x1),dim=-1)
        ls2 = torch.softmax(self.ls2(x2),dim=-1)
        ls3 = torch.softmax(self.ls3(x3),dim=-1)

        if raw_output and not self.training:
            if t_id == 0:
                idx=0
            elif t_id == 1:
                idx=2
            else:
                idx=4
            lower_limit = self.limits[idx]
            upper_limit = self.limits[idx+1]
            action_space = torch.arange(lower_limit.item(), upper_limit.item(), (upper_limit.item() - lower_limit.item()) / ls3.shape[1])
            max_value_ind = torch.argmax(ls1).int()
            ls1 = action_space[max_value_ind].item()
            max_value_ind = torch.argmax(ls2).int()
            ls2 = action_space[max_value_ind].item()
            max_value_ind = torch.argmax(ls3).int()
            ls3 = action_space[max_value_ind].item()
            x = [ls1, ls2, ls3]
        else:
            x = [ls1, ls2, ls3]
        return x

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def modulator(self, input_vector, modulation):
        modulation_idx = torch.argmax(modulation, dim=1)
        modulation_strength = ((modulation_idx + 1) / self.modulation_resolution) * self.modulation_scale
        modulated = input_vector * modulation_strength.unsqueeze(1)
        return modulated

    def generate_orthogonal_vectors(self, n, dim):
        vectors = torch.randn(n, dim)
        orthogonal_vectors = torch.zeros_like(vectors)
        for i in range(n):
            new_vec = vectors[i]
            for j in range(i):
                new_vec -= torch.dot(orthogonal_vectors[j], vectors[i]) * orthogonal_vectors[j]
            orthogonal_vectors[i] = new_vec / new_vec.norm()  # Normalize

        return orthogonal_vectors

class NeuralNetwork_SA(nn.Module):
    def __init__(self, no_of_actions, no_of_states, device):
        super(NeuralNetwork_SA, self).__init__()
        self.task_indicator = None
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states
        self.hidden_size = 256
        self.modulation_resolution = 10
        self.modulation_scale = 2
        self.device = device
        self.input = self.no_of_states * 2 + 32 + self.no_of_actions*3
        self.hidden_state_action = self.input + self.hidden_size
        self.act = nn.LeakyReLU(0.1)#nn.ELU(2)

        self.linear1_a0 = nn.Linear(self.no_of_actions, self.input*2, bias=True)
        self.linear1_a1 = nn.Linear(self.no_of_actions, self.input*2, bias=True)
        self.linear1_a2 = nn.Linear(self.no_of_actions, self.input*2, bias=True)
        self.linear1_c = nn.Linear(self.no_of_states * 2 + 32, self.input*2, bias=True)

        self.linear2 = nn.Linear(self.input*8, self.hidden_size*2, bias=True)
        self.res_blocks = nn.Sequential(*[ResidualBlock(self.hidden_size * 2, dropout=0.1) for _ in range(10)])

        self.linear3 = nn.Linear(self.hidden_size*2, self.hidden_size, bias=True)
        self.linear4_1 = nn.Linear(self.hidden_size, self.no_of_actions, bias=True)

        self.linear4_2 = nn.Linear(self.hidden_size, self.no_of_actions, bias=True)
        self.linear4_3 = nn.Linear(self.hidden_size, self.no_of_actions, bias=True)

        self.linear5_1 = nn.Linear(self.no_of_actions, self.no_of_actions, bias=True)
        self.linear5_2 = nn.Linear(self.no_of_actions, self.no_of_actions, bias=True)
        self.linear5_3 = nn.Linear(self.no_of_actions, self.no_of_actions, bias=True)

        self.LNorm1 = nn.LayerNorm(self.input*8)
        self.LNorm2 = nn.LayerNorm(self.hidden_size*2)
        self.LNorm3 = nn.LayerNorm(self.hidden_size )
        self.LNorm4_1 = nn.LayerNorm(self.no_of_actions)
        self.LNorm4_2 = nn.LayerNorm(self.no_of_actions)
        self.LNorm4_3 = nn.LayerNorm(self.no_of_actions)

        self.apply(self._init_weights)

    def forward(self, state, action, t_id):
        if isinstance(t_id, int):
            t_id = torch.tensor([t_id]).to(self.device)
        t_id = t_id.long() if isinstance(t_id, torch.Tensor) else torch.tensor(t_id, dtype=torch.long)
        task_id = self.task_indicator[t_id].squeeze(1)
        if task_id.dim() == 3:
            task_id = self.task_indicator[t_id].squeeze(0)
        for i in range(0,3):
            action[i] = torch.squeeze(action[i], dim=0)
            if state.dim() < 2:
                state = torch.unsqueeze(state, dim=0)
            if action[i].dim() < 2:
                action[i] = torch.unsqueeze(action[i], dim=0)
            elif action[i].dim() == 3:
                action[i] = torch.swapaxes(action[i], 0, 1)
                action[i] = torch.squeeze(action[i], dim=0)

        context_input = torch.cat((state, task_id), dim=1)

        x0 = self.act(self.linear1_a0(action[0]))
        x1 = self.act(self.linear1_a1(action[1]))
        x2 = self.act(self.linear1_a2(action[2]))
        c = self.act(self.linear1_c(context_input))
        context_input = torch.cat([x0,x1,x2,c], dim=1)
        #x = self.LNorm1(context_input)

        x = self.act(self.linear2(context_input))
        x = self.LNorm2(x)

        x = self.res_blocks(x)

        x = self.act(self.linear3(x))
        x = self.LNorm3(x)

        x1 = self.act(self.linear4_1(x))
        x2 = self.act(self.linear4_2(x))
        x3 = self.act(self.linear4_3(x))

        x1 = self.LNorm4_1(x1)
        x2 = self.LNorm4_2(x2)
        x3 = self.LNorm4_3(x3)

        x1 = self.linear5_1(x1)
        x2 = self.linear5_2(x2)
        x3 = self.linear5_3(x3)
        x = [x1, x2, x3]
        return x


    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def modulator(self, input_vector, modulation):
        modulation_idx = torch.argmax(modulation, dim=1)
        modulation_strength = ((modulation_idx + 1) / self.modulation_resolution) * self.modulation_scale
        modulated = input_vector * modulation_strength.unsqueeze(1)
        return modulated

class ResidualBlock(nn.Module):
    def __init__(self, in_dim, out_dim=None, hidden_dim=None, dropout=0.0):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        if hidden_dim is None:
            hidden_dim = max(in_dim, out_dim)

        self.norm1 = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.norm2 = nn.LayerNorm(out_dim)

        self.act = nn.ELU(2)  # or nn.Mish()
        self.dropout = nn.Dropout(dropout)

        if in_dim == out_dim:
            self.proj = nn.Identity()
        else:
            self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        out = self.norm1(x)
        out = self.act(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        x = self.proj(x)
        return x + out

