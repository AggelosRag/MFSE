import numpy as np
import torch
from torch.nn import Parameter as Param

class MultiInnerProductDecoder2(torch.nn.Module):
    def __init__(self, in_dim, num_et):
        super(MultiInnerProductDecoder2, self).__init__()
        self.num_et = num_et
        self.in_dim = in_dim
        self.weight = Param(torch.Tensor(num_et, in_dim))

        self.reset_parameters()

    def forward(self, z, edge_index, sigmoid=True):

        edge_type_num = 0
        edge_type_values = []
        edge_type_values2 = []
        for edge_type in edge_index.values():
            source_idx = edge_type[0]
            dest_idx = edge_type[1]
            w = self.weight[edge_type_num]
            edge_type_num += 1
            value = (z[source_idx] * z[dest_idx] * w).sum(dim=1)
            edge_type_values.append(
                torch.sigmoid(value) if sigmoid else value)
            edge_type_values2.append(value)

        score = torch.cat(edge_type_values2, dim=0)
        score = torch.sigmoid(score) if sigmoid else score

        return edge_type_values, score

    def reset_parameters(self):
        self.weight.data.normal_(std=1 / np.sqrt(self.in_dim))