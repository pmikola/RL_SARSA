import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# POLICY ARCHITECTURE
from torch import Tensor

LOG_SIG_MAX = 2
LOG_SIG_MIN = -5


class MultiHeadLayer(nn.Module):
    def __init__(self, hidden_size, no_of_actions):
        super(MultiHeadLayer, self).__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(#*[ResidualBlock(hidden_size, dropout=0.1) for _ in range(1)],
            nn.Linear(hidden_size, no_of_actions, bias=True))
            for _ in range(3)
        ])
    def forward(self, x):
        outputs = [head(x) for head in self.heads]
        return outputs

class ValueNetwork(nn.Module):
    def __init__(self, no_of_actions, no_of_states, device):
        super(ValueNetwork, self).__init__()
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states
        self.input = self.no_of_states * 2
        self.hidden_size = 2*self.input+1
        self.device = device
        self.modulation_resolution = 10
        self.modulation_scale = 2
        self.hidden_state = self.input + self.hidden_size
        self.act = nn.LeakyReLU(0.2)

        self.linear1 = nn.Linear(self.input, self.hidden_size, bias=True)
        self.linear2 = nn.Linear(self.hidden_size + self.input+32, self.hidden_size * 2, bias=True)
        self.head_groups = nn.ModuleDict({
            "id0": MultiHeadLayer(self.hidden_size * 2,self.no_of_actions),
            "id1": MultiHeadLayer(self.hidden_size * 2, self.no_of_actions),
            "id2": MultiHeadLayer(self.hidden_size * 2, self.no_of_actions)
        })

        self.LNorm1 = nn.LayerNorm(self.hidden_size)
        self.LNorm2 = nn.LayerNorm(self.hidden_size * 2)

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
        x = torch.cat([x, state,task_id], dim=1)
        x = self.act(self.linear2(x))
        x = self.LNorm2(x)
        group_index = int(torch.mean(t_id.float()).item())
        key = f"id{group_index}"
        selected_heads = self.head_groups[key]
        ls_out = selected_heads(x)
        ls1 = torch.softmax(ls_out[0],dim=-1)
        ls2 = torch.softmax(ls_out[1],dim=-1)
        ls3 = torch.softmax(ls_out[2],dim=-1)

        if raw_output and not self.training:
            if group_index == 0:
                idx=0
            elif group_index == 1:
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

class QNetwork(nn.Module):
    def __init__(self, no_of_actions, no_of_states, device):
        super(QNetwork, self).__init__()
        self.task_indicator = None
        self.no_of_actions = no_of_actions
        self.no_of_states = no_of_states
        self.input = self.no_of_states * 2 + 32 + self.no_of_actions*3
        self.hidden_size = self.input
        self.modulation_resolution = 10
        self.modulation_scale = 2
        self.device = device
        self.hidden_state_action = self.input + self.hidden_size
        self.act = nn.LeakyReLU(0.1)#nn.ELU(2)

        self.linear1_a0 = nn.Linear(self.no_of_actions, self.input*2, bias=True)
        self.linear1_a1 = nn.Linear(self.no_of_actions, self.input*2, bias=True)
        self.linear1_a2 = nn.Linear(self.no_of_actions, self.input*2, bias=True)
        self.linear1_c = nn.Linear(self.no_of_states * 2 + 32, self.input*2, bias=True)
        self.linear2 = nn.Linear(self.input*8, self.hidden_size*2, bias=True)
        self.head_groups = nn.ModuleDict({
            "id0": MultiHeadLayer(self.hidden_size * 2, self.no_of_actions),
            "id1": MultiHeadLayer(self.hidden_size * 2, self.no_of_actions),
            "id2": MultiHeadLayer(self.hidden_size * 2, self.no_of_actions)
        })
        self.LNorm1 = nn.LayerNorm(self.input*8)
        self.LNorm2 = nn.LayerNorm(self.hidden_size*2)
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

        group_index = int(torch.mean(t_id.float()).item())
        key = f"id{group_index}"
        selected_heads = self.head_groups[key]
        ls_out = selected_heads(x)
        ls1 = ls_out[0]
        ls2 = ls_out[1]
        ls3 = ls_out[2]
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

