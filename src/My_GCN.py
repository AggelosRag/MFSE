import torch
from torch_geometric.nn import MessagePassing


class My_GCN(MessagePassing):

    def __init__(self,
                 in_channels,
                 out_channels,
                 side_effect_num,
                 **kwargs):
        super(My_GCN, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.side_effect_num = side_effect_num

    def forward(self, x, edge_index, weight):
        """"""
        return self.propagate(edge_index, x=x, weight=weight)

    def message(self, x_j, weight):
        tmp = torch.matmul(x_j, weight)
        return tmp

    def __repr__(self):
        return '{}({}, {}, num_relations={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.num_relations)
