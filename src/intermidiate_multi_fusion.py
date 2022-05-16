import numpy as np
import torch
from torch.nn import Parameter as Param
from torch.nn import functional as F

class Multi_Inter_Fusion(torch.nn.Module):
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
                 hid,
                 out_channels,
                 num_relations,
                 num_bases,
                 device,
                 after_relu,
                 bias=False,
                 **kwargs):
        super(Multi_Inter_Fusion, self).__init__()

        self.in_channels = in_channels
        self.hid = hid
        self.out_channels = out_channels
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.device = device
        self.after_relu = after_relu

        self.basis1 = Param(torch.Tensor(num_bases, in_channels, hid))
        self.att1 = Param(torch.Tensor(num_relations, num_bases))
        self.root1 = Param(torch.Tensor(in_channels, hid))
        self.bn1 = torch.nn.BatchNorm1d(hid, track_running_stats=False)

        self.basis2 = Param(torch.Tensor(num_bases, hid, out_channels))
        self.att2 = Param(torch.Tensor(num_relations, num_bases))
        self.root2 = Param(torch.Tensor(hid, out_channels))

        if bias:
            self.bias = Param(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):

        self.att1.data.normal_(std=1 / np.sqrt(self.num_bases))
        self.att2.data.normal_(std=1 / np.sqrt(self.num_bases))

        if self.after_relu:
            self.root1.data.normal_(std=2 / self.in_channels)
            self.basis1.data.normal_(std=2 / self.in_channels)
            self.root2.data.normal_(std=2 / self.hid)
            self.basis2.data.normal_(std=2 / self.hid)
        else:
            self.root1.data.normal_(std=1 / np.sqrt(self.in_channels))
            self.basis1.data.normal_(std=1 / np.sqrt(self.in_channels))
            self.root2.data.normal_(std=1 / np.sqrt(self.hid))
            self.basis2.data.normal_(std=1 / np.sqrt(self.hid))

        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x_input, edge_index_dict):
        """"""
        w1 = torch.matmul(self.att1, self.basis1.view(self.num_bases, -1))
        w1 = w1.view(self.num_relations, self.in_channels, self.hid)

        w2 = torch.matmul(self.att2, self.basis2.view(self.num_bases, -1))
        w2 = w2.view(self.num_relations, self.hid, self.out_channels)

        num_rel = 0
        embeddings = []
        for edge_name in edge_index_dict.keys():
            if edge_name[1] not in ['connects_to', 'interacts_with'] and 'bonds_with' not in edge_name[1]:
                x = torch.matmul(x_input, w1[num_rel])
                out = x + torch.matmul(x_input, self.root1)

                x = F.relu(out, inplace=True)
                x = torch.matmul(x, w2[num_rel])
                x = x + torch.matmul(out, self.root2)

                embeddings.append(x)
                num_rel += 1

        return embeddings
