import numpy as np
import torch
from torch.nn import Parameter as Param
from torch_geometric.nn import HeteroConv
from src.My_GCN import My_GCN

class M_RGCN(torch.nn.Module):
    r"""
    Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        num_relationse (int): Number of relations.
        num_bases (int): Number of bases used for basis-decomposition.
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 num_bonds,
                 num_bases,
                 device,
                 after_relu,
                 bias=False,
                 **kwargs):
        super(M_RGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_bonds = num_bonds
        self.num_bases = num_bases
        self.device = device
        self.after_relu = after_relu

        self.basis = Param(torch.Tensor(num_bases, in_channels, out_channels))
        self.att = Param(torch.Tensor(num_bonds, num_bases))
        self.root = Param(torch.Tensor(in_channels, out_channels))

        module_dict = {}
        for bond_id in range(num_bonds):
            relationship_name = f'bonds_with_{bond_id}'
            gcn = My_GCN(in_channels, out_channels, bond_id)
            module_dict[('atom', relationship_name, 'atom')] = gcn

        self.conv = HeteroConv(module_dict, aggr='mean')

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        self.att.data.normal_(std=1 / np.sqrt(self.num_bases))

        if self.after_relu:
            self.root.data.normal_(std=2 / self.in_channels)
            self.basis.data.normal_(std=2 / self.in_channels)

        else:
            self.root.data.normal_(std=1 / np.sqrt(self.in_channels))
            self.basis.data.normal_(std=1 / np.sqrt(self.in_channels))

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x_dict, edge_index_dict, dest_dict):
        """"""
        w = torch.matmul(self.att, self.basis.view(self.num_bases, -1))
        w = w.view(self.num_bonds, self.in_channels, self.out_channels)

        weights = {}
        num_rel = 0
        for edge_name in edge_index_dict.keys():
            weights[edge_name] = w[num_rel]
            num_rel += 1

        out_dict = self.conv(x_dict=x_dict, edge_index_dict=edge_index_dict,
                             weight_dict = weights)
        averaged_out = out_dict['atom']

        out = averaged_out + torch.matmul(x_dict['atom'], self.root)

        if self.bias is not None:
            out = out + self.bias
        return out
