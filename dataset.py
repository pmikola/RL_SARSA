# REFERENCE EXPERTS VALUES BODY PART PER SKIN AND HAIR TYPE
import np as np
import torch
from torch.autograd import Variable


class DataSet:
    def __init__(self):
        # SETTINGS FOR IX LASER DEPILATION SERIES
        # REFERENCES FROM THE EXPERTS
        self.kj_total = torch.Tensor(
            [[[14, 14], [12, 12], [13, 13]], [[16, 16], [14, 14], [15, 15]], [[18, 18], [16, 16], [17, 17]],
             [[20, 20], [18, 18], [19, 19]], [[20, 20], [20, 20], [20, 20]], [[20, 20], [20, 20], [20, 20]],
             [[20, 20], [20, 20], [20, 20]],
             [[20, 20], [20, 20], [20, 20]], [[20, 20], [20, 20], [20, 20]]]).requires_grad_(True)

        self.hz = torch.Tensor([[[3, 5], [3, 5], [3, 5]], [[3, 5], [3, 5], [3, 5]], [[5, 5], [3, 10], [5, 5]],
                            [[5, 10], [3, 10], [5, 5]], [[5, 15], [5, 15], [5, 15]], [[5, 15], [5, 15], [5, 15]],
                            [[5, 15], [5, 15], [5, 15]], [[5, 15], [5, 15], [5, 15]], [[5, 15], [5, 15], [5, 15]]]).requires_grad_(True)

        self.j_cm2 = torch.Tensor(
            [[[18, 18], [15, 15], [15, 17]], [[20, 20], [17, 17], [17, 17]], [[23, 23], [19, 19], [20, 20]],
             [[25, 28], [20, 20], [20, 25]], [[25, 28], [20, 25], [20, 25]], [[25, 28], [20, 25], [20, 25]],
             [[25, 28], [20, 25], [20, 25]], [[25, 28], [20, 25], [20, 25]], [[25, 28], [20, 25], [20, 25]]]).requires_grad_(True)

        # FOTOTYPE MAX VALS FOR I-III
        self.hz_black = torch.Tensor([3, 30]).requires_grad_(True)
        self.hz_dark_brown = torch.Tensor([5, 25]).requires_grad_(True)
        self.hz_lighth_brown = torch.Tensor([10, 20]).requires_grad_(True)
        self.hz_blond_red = torch.Tensor([15, 15]).requires_grad_(True)
        self.women_area_per_bodyPart = torch.Tensor([3, 2, 6, 4, 1, 1, 4, 3, 4, 3]).requires_grad_(True)
        self.men_area_per_bodyPart = torch.Tensor([4, 3, 8, 6, 1, 1, 6, 6, 12, 2]).requires_grad_(True)

    def create_target(self, std):
        kj_total_var = torch.zeros((9, 3), requires_grad=True)
        hz_var = torch.zeros((9, 3), requires_grad=True)
        j_cm2_var = torch.zeros((9, 3), requires_grad=True)
        std = std / 100

        for i in range(len(self.kj_total)):
            kj_total_var[i][0].data.copy_(torch.normal(self.kj_total[i][0][0].clone(), self.kj_total[i][0][0].detach()))
            kj_total_var[i][0].requires_grad_(True)
            kj_total_var[i][1].data.copy_(torch.normal(self.kj_total[i][1][0].clone(), self.kj_total[i][1][0].detach()))
            kj_total_var[i][1].requires_grad_(True)
            kj_total_var[i][2].data.copy_(torch.normal(self.kj_total[i][2][0].clone(), self.kj_total[i][2][0].detach()))
            kj_total_var[i][2].requires_grad_(True)
            hz_var[i][0].data.copy_(torch.normal(self.hz[i][0][0].clone(), self.hz[i][0][0].detach()))
            hz_var[i][0].requires_grad_(True)
            hz_var[i][1].data.copy_(torch.normal(self.hz[i][1][0].clone(), self.hz[i][1][0].detach()))
            hz_var[i][1].requires_grad_(True)
            hz_var[i][2].data.copy_(torch.normal(self.hz[i][2][0].clone(), self.hz[i][2][0].detach()))
            hz_var[i][2].requires_grad_(True)

            j_cm2_var[i][0].data.copy_(torch.normal(self.j_cm2[i][0][0].clone(), self.j_cm2[i][0][0].detach()))
            j_cm2_var[i][0].requires_grad_(True)
            j_cm2_var[i][1].data.copy_(torch.normal(self.j_cm2[i][1][0].clone(), self.j_cm2[i][1][0].detach()))
            j_cm2_var[i][1].requires_grad_(True)
            j_cm2_var[i][2].data.copy_(torch.normal(self.j_cm2[i][2][0].clone(), self.j_cm2[i][2][0].detach()))
            j_cm2_var[i][2].requires_grad_(True)

        return kj_total_var, hz_var, j_cm2_var

    def create_input_set(self):

        hair_type = np.random.randint(0, 3)
        skin_type = np.random.randint(0, 5)
        body_type = np.random.randint(0,3)
        #print('hair_type:',hair_type,'skin_type:',skin_type,'body_type:',body_type)
        hair_color = torch.Tensor(list(map(float, f'{hair_type:03b}'))).requires_grad_(True)
        skin_color = torch.Tensor(list(map(float, f'{skin_type:03b}'))).requires_grad_(True)
        body_part = torch.Tensor(list(map(float, f'{body_type:03b}'))).requires_grad_(True)
        rl_input = torch.cat((hair_color,skin_color,body_part)).requires_grad_(True)
        # print(rl_input)
        return rl_input

    def decode_input(self,rl_input):
        hair_type = int(str(int(rl_input[0]))+str(int(rl_input[1]))+str(int(rl_input[2])),2)
        skin_type = int(str(int(rl_input[3]))+str(int(rl_input[4]))+str(int(rl_input[5])),2)
        body_part = int(str(int(rl_input[6])) + str(int(rl_input[7])) + str(int(rl_input[8])),2)
        return hair_type,skin_type,body_part



