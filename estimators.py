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
            Q_next_a = target_actor(s_next.detach(), task_indicator.detach())
            a_n_1 = []
            a_n_2 = []
            for i in range(0, 3):
                noise_selector = random.randint(0, 1)
                a_next_noise = 0.#(torch.rand_like(Q_next_a[i].detach()) * 2 - 1)*(1e-5/self.epsilon)#*eps
                if noise_selector == 0:
                    a_n_1.append(Q_next_a[i].detach()+Q_next_a[i].detach()*a_next_noise)
                    a_n_2.append(Q_next_a[i].detach())
                else:
                    a_n_1.append(Q_next_a[i].detach())
                    a_n_2.append(Q_next_a[i].detach() + Q_next_a[i].detach() * a_next_noise)
            Q_next_1 = target_critic_1(s_next.detach(), a_n_1, task_indicator.detach())
            Q_next_2 = target_critic_2(s_next.detach(), a_n_2, task_indicator.detach())
            z_t = torch.zeros_like(Q_next_1[0]).to(self.device)
            Q_target = [z_t, z_t, z_t]
            self.counter += 1
            q1n = critic_1(s_next.detach(), a_n_1, task_indicator.detach())
            q2n = critic_2(s_next.detach(), a_n_2, task_indicator.detach())
            for i in range(3):
                idx_select = random.randint(0,1)
                a_star = Q_next_1[i].argmax(dim=-1,keepdims=True)
                b_star = Q_next_2[i].argmax(dim=-1,keepdims=True)
                Q_next_1_target = q1n[i].gather(-1, b_star)
                Q_next_2_target = q2n[i].gather(-1, a_star)
                Q_t_update = 0.5* ( Q_next_2_target.clone()+ Q_next_1_target.clone())
                #print(Q_new[i].shape,r.unsqueeze(-1).shape , ad_reward.unsqueeze(-1).shape , done.shape, Q_next_target_updated.shape)
                #q_target_up_a = self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + ((1 - done) * self.gamma * Q_next_1_target.clone()))
                q_target_up_b = self.alpha * ((r.unsqueeze(-1) + ad_reward.unsqueeze(-1)) + ((1 - done) * self.gamma *Q_t_update))
                Q_target[i].scatter_(1, b_star, q_target_up_b)
                #Q_target[i].scatter_(1, b_star, q_target_up_a)
        return Q_target

    def soft_update(self, net, target_net):
        with torch.no_grad():
            for target_param, param in zip(target_net.parameters(), net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))
