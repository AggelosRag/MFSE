import torch
from torch.nn import functional as F
from torch import nn
from torch_geometric.data import HeteroData, DataLoader
from torch_geometric.nn import global_mean_pool
from src.m_rgcn import M_RGCN


class M_Encoder(torch.nn.Module):
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
                 num_drugs,
                 num_feat_per_atom,
                 out_channels,
                 device,
                 **kwargs):
        super(M_Encoder, self).__init__()

        self.num_drugs = num_drugs
        self.num_feat_per_atom = num_feat_per_atom
        self.out_channels = out_channels
        self.device = device

        self.rgcn_mol1 = M_RGCN(in_channels=self.num_feat_per_atom,
                                out_channels=self.out_channels,
                                num_bonds=4,
                                num_bases = 32,
                                device = self.device,
                                after_relu = False)
        self.b1 = nn.BatchNorm1d(self.out_channels,
                                 track_running_stats=False)
        self.rgcn_mol2 = M_RGCN(in_channels=self.out_channels,
                                out_channels=self.out_channels,
                                num_bonds=4,
                                num_bases = 32,
                                device = self.device,
                                after_relu = True)

    def forward(self, batch):
        """"""

        drug_emb_dict = {}

        if isinstance(batch['molecules'].datasetb0, list):
            batch['molecules'].datasetb0 = batch['molecules'].datasetb0[0]
        if isinstance(batch['molecules'].datasetb1, list):
            batch['molecules'].datasetb1 = batch['molecules'].datasetb1[0]
        if isinstance(batch['molecules'].datasetb2, list):
            batch['molecules'].datasetb2 = batch['molecules'].datasetb2[0]
        if isinstance(batch['molecules'].datasetb3, list):
            batch['molecules'].datasetb3 = batch['molecules'].datasetb3[0]

        dataloader_b0 = DataLoader(batch['molecules'].datasetb0,
                                shuffle=False,
                                num_workers=0,
                                batch_size=batch['molecules'].datasetb0.len())
        batch_b0  = next(iter(dataloader_b0)).to(self.device)
        dataloader_b1  = DataLoader(batch['molecules'].datasetb1,
                                shuffle=False,
                                num_workers=0,
                                batch_size=batch['molecules'].datasetb1.len())
        batch_b1  = next(iter(dataloader_b1)).to(self.device)
        dataloader_b2  = DataLoader(batch['molecules'].datasetb2,
                                shuffle=False,
                                num_workers=0,
                                batch_size=batch['molecules'].datasetb2.len())
        batch_b2  = next(iter(dataloader_b2)).to(self.device)
        dataloader_b3  = DataLoader(batch['molecules'].datasetb3,
                                shuffle=False,
                                num_workers=0,
                                batch_size=batch['molecules'].datasetb3.len())
        batch_b3  = next(iter(dataloader_b3)).to(self.device)

        data = HeteroData()
        data['atom'].x = batch_b0.x
        data['atom', 'bonds_with_0', 'atom'].edge_index = batch_b0.edge_index.type(torch.LongTensor)
        data['atom', 'bonds_with_1', 'atom'].edge_index = batch_b1.edge_index.type(torch.LongTensor)
        data['atom', 'bonds_with_2', 'atom'].edge_index = batch_b2.edge_index.type(torch.LongTensor)
        data['atom', 'bonds_with_3', 'atom'].edge_index = batch_b3.edge_index.type(torch.LongTensor)
        data.to(self.device)

        drug_emb = self.rgcn_mol1(
                        x_dict = data.x_dict,
                        edge_index_dict = data.edge_index_dict,
                        dest_dict = None
        )
        drug_emb = self.b1(drug_emb)
        drug_emb = F.relu(drug_emb, inplace=True)
        data["atom"].x = drug_emb

        drug_emb = self.rgcn_mol2(
                        x_dict = data.x_dict,
                        edge_index_dict = data.edge_index_dict,
                        dest_dict = None
        )

        x = global_mean_pool(drug_emb, batch_b0.batch)

        return x