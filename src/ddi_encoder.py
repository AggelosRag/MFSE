import torch
from src.se_rgcn import SE_RGCN
from torch.nn import functional as F
from torch.nn import Parameter as Param

class DDI_Encoder(torch.nn.Module):

    def __init__(self, input_dim, num_se, num_base, dest_dict, device,
                 in_dim_drug):
        super(DDI_Encoder, self).__init__()

        self.input_dim = input_dim
        self.hid1 = 64
        self.hid2 = 20
        self.num_se = num_se
        self.num_base = num_base
        self.dest_dict = dest_dict
        self.device = device

        self.drug_w = Param(torch.Tensor(in_dim_drug, input_dim))

        self.se_rgcn = SE_RGCN(self.input_dim, self.hid1, self.num_se,
                                self.num_base, self.dest_dict, self.device,
                                after_relu=False)

        self.se_rgcn2 = SE_RGCN(self.hid1, self.hid2, self.num_se,
                                 self.num_base, self.dest_dict, self.device,
                                 after_relu=True)
        self.reset_parameters()

    def forward(self, batch_copy):
        x_drug_alone = torch.matmul(batch_copy["drug"].x, self.drug_w)
        batch_copy["drug"].x = x_drug_alone

        x_drug = self.se_rgcn(batch_copy.x_dict, batch_copy.edge_index_dict)
        x_drug = F.relu(x_drug, inplace=True)
        batch_copy["drug"].x = x_drug
        x_drug = self.se_rgcn2(batch_copy.x_dict, batch_copy.edge_index_dict)

        return batch_copy, x_drug

    def reset_parameters(self):
        self.drug_w.data.normal_()