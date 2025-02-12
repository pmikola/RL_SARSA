import random
import time

import torch
import torch.nn.functional as F


class Estimators:
    def __init__(self, alpha, epsilon, gamma, tau, device, no_of_actions, n_steps=2, v_min=-900, v_max=900):
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_atoms = no_of_actions
        self.v_min = v_min
        self.v_max = v_max
        self.delta_z = (v_max - v_min) / (self.num_atoms - 1)
        self.support = torch.linspace(v_min, v_max, self.num_atoms).to(device)
        self.device = device
        self.tau = tau
        self.n_steps = n_steps
        self.counter = 0

    def Q_value(self,eps,actor,target_actor, critic_1, critic_2, target_critic_1,target_critic_2, s, a, r, s_next, a_next, done, task_indicator, ad_reward):
        #actor_actions = actor(s, task_indicator)
        a_p_1 = []
        a_p_2 = []
        for param in actor.parameters():
            param.requires_grad = True
        for param in critic_1.parameters():
            param.requires_grad = True
        for param in critic_2.parameters():
            param.requires_grad = True

        for i in range(0, 3):
            noise_selector = random.randint(0, 1)
            a_noise = (torch.rand_like(a[i].detach()) * 2 - 1)*(1e-2/self.epsilon)#*eps
            if noise_selector == 0:
                a_p_1.append(a[i].detach()+a[i].detach()*a_noise)
                a_p_2.append(a[i].detach())
            else:
                a_p_1.append(a[i].detach())
                a_p_2.append(a[i].detach() + a[i].detach() * a_noise)

        Q_current_1 = critic_1(s.detach(), a_p_1, task_indicator.detach())
        Q_current_2 = critic_2(s.detach(), a_p_2, task_indicator.detach())
        Q_current_a = actor(s.detach(), task_indicator.detach())
        # print(target.__class__.__name__)
        with torch.no_grad():
            a_n_1 = []
            a_n_2 = []
            for i in range(0, 3):
                noise_selector = random.randint(0, 1)
                a_next_noise = (torch.rand_like(a_next[i].detach()) * 2 - 1)*(1e-2/self.epsilon)#*eps
                if noise_selector == 0:
                    a_n_1.append(a_next[i].detach()+a_next[i].detach()*a_next_noise)
                    a_n_2.append(a_next[i].detach())
                else:
                    a_n_1.append(a_next[i].detach())
                    a_n_2.append(a_next[i].detach() + a_next[i].detach() * a_next_noise)
            Q_next_1 = target_critic_1(s_next.detach(), a_n_1, task_indicator.detach())
            Q_next_2 = target_critic_2(s_next.detach(), a_n_2, task_indicator.detach())
            #Q_next_a = target_actor(s_next.detach(), task_indicator.detach())
        r_t = torch.zeros_like(r.unsqueeze(-1)).to(self.device)
        Q_new = [r_t, r_t, r_t]
        z_t = torch.zeros_like(Q_next_1[0]).to(self.device)
        # print(z_t.shape)
        Q_target = [z_t, z_t, z_t]

        #importance_selector =-1
        self.counter += 1
        for i in range(0, 3):
            q_next_selector = random.randint(0, 1)
            Q_target_updated = [Q_next_1[i].clone(),Q_next_2[i].clone()][q_next_selector]
            current_action_idx = torch.argmax(a[i], dim=-1)
            next_action_idx = torch.argmax(a_next[i], dim=-1)
            if next_action_idx.dim() < 2:
                next_action_idx = next_action_idx.unsqueeze(1)
                current_action_idx = current_action_idx.unsqueeze(1)
            q_next = Q_target_updated.gather(1, next_action_idx)
            Q_new[i] = self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + (1 - done) * self.gamma * q_next) #+ done * (r.unsqueeze(-1) + ad_reward.unsqueeze(-1))
            Q_target_updated.scatter_(dim=1, index=current_action_idx, src=Q_new[i])
            Q_target[i] = Q_target_updated
                # if i == 0:
                #     for param in actor.ls1.parameters():
                #         param.requires_grad = False
                #     for param in actor.ls_11.parameters():
                #         param.requires_grad = False
                #     for param in actor.ls_01.parameters():
                #         param.requires_grad = False
                #     for param in actor.res_blocks_1.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.res_blocks_1.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.linear4_1.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.linear5_1.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.res_blocks_1.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.linear4_1.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.linear5_1.parameters():
                #         param.requires_grad = False
                # elif i == 1:
                #     for param in actor.ls2.parameters():
                #         param.requires_grad = False
                #     for param in actor.ls_12.parameters():
                #         param.requires_grad = False
                #     for param in actor.ls_02.parameters():
                #         param.requires_grad = False
                #     for param in actor.res_blocks_2.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.res_blocks_2.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.linear4_2.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.linear5_2.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.res_blocks_2.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.linear4_2.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.linear5_2.parameters():
                #         param.requires_grad = False
                # else:
                #     for param in actor.ls3.parameters():
                #         param.requires_grad = False
                #     for param in actor.ls_13.parameters():
                #         param.requires_grad = False
                #     for param in actor.ls_03.parameters():
                #         param.requires_grad = False
                #     for param in actor.res_blocks_3.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.res_blocks_3.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.linear4_3.parameters():
                #         param.requires_grad = False
                #     for param in critic_1.linear5_3.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.res_blocks_3.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.linear4_3.parameters():
                #         param.requires_grad = False
                #     for param in critic_2.linear5_3.parameters():
                #         param.requires_grad = False
        return Q_target, Q_current_1,Q_current_2,Q_current_a

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
