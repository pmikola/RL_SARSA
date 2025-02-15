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
        with torch.no_grad():
            Qo_next_a = actor(s_next.detach(), task_indicator.detach())
            Qt_next_a = target_actor(s_next.detach(), task_indicator.detach())
            a_n_1 = []
            a_n_2 = []
            for i in range(0, 3):
                noise_selector = random.randint(0, 1)
                a_next_noise = (torch.rand_like(Qt_next_a[i].detach()) * 2 - 1) * (1e-5 / self.epsilon)#*eps
                if noise_selector == 0:
                    a_n_1.append(Qt_next_a[i].detach() + Qt_next_a[i].detach() * a_next_noise)
                    a_n_2.append(Qt_next_a[i].detach())
                else:
                    a_n_1.append(Qt_next_a[i].detach())
                    a_n_2.append(Qt_next_a[i].detach() + Qt_next_a[i].detach() * a_next_noise)
            #q1n = critic_1(s_next.detach(), a_n_1, task_indicator.detach())
            # q2n = critic_2(s_next.detach(), a_n_2, task_indicator.detach())
            #Q_next_1 = target_critic_1(s_next.detach(), a_n_1, task_indicator.detach())
            #Q_next_2 = target_critic_2(s_next.detach(), a_n_2, task_indicator.detach())
            # ref = torch.zeros_like(Q_next_a[0]).to(self.device)
            ref = torch.zeros_like(r.unsqueeze(-1)).to(self.device)
            Q_target = [ref,ref, ref]
            self.counter += 1
            idx_select = random.randint(0, 1)
            for i in range(3):
                a_star = Qo_next_a[i].argmax(dim=-1, keepdims=True)
               # b_star = Q_next_2[i].argmax(dim=-1,keepdims=True)
                Q_next_1_target = Qt_next_a[i].gather(-1, a_star)
                #Q_next_2_target = q2n[i]#.gather(-1, a_star)
                #Q_t_update = Q_next_1_target.clone()+Q_next_2_target.clone()
                #index_star = [ b_star,a_star][idx_select]
                #print(Q_new[i].shape,r.unsqueeze(-1).shape , ad_reward.unsqueeze(-1).shape , done.shape, Q_next_target_updated.shape)
                #q_target_up_a = self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + ((1 - done) * self.gamma * Q_next_1_target.clone()))
                q_target_up = self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + ((1 - done) * self.gamma *Q_next_1_target))
                #Q_target[i].scatter_(1, index_star, q_target_up)
                Q_target[i] = q_target_up
        return Q_target

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
