import torch
from torch_geometric.nn import HeteroConv, SAGEConv, GCNConv
from torch.nn import functional as F


class PPI_DTI_Encoder(torch.nn.Module):

    def __init__(self,
                 in_dim_prot,
                 in_dim_drug,
                 prot_drug_dim):

        super(PPI_DTI_Encoder, self).__init__()

        #ppi
        self.ppi_d1 = 64
        self.ppi_d2 = 32
        self.ppi_1 = GCNConv(in_dim_prot, self.ppi_d1, cached=True)
        self.ppi_2 = GCNConv(self.ppi_d1, self.ppi_d2, cached=True)

        #dti
        self.dti = HeteroConv({
            ('protein', 'connects_to', 'drug'): SAGEConv(
                (self.ppi_d2, in_dim_drug), prot_drug_dim)
        })

    def forward(self, batch_copy):

        #ppi
        x_prot = batch_copy["protein"].x
        pp_edge_index = batch_copy["protein", "interacts_with", "protein"].edge_index

        x_prot = self.ppi_1(x_prot, pp_edge_index)
        x_prot = F.relu(x_prot, inplace=True)
        x_prot = self.ppi_2(x_prot, pp_edge_index)
        batch_copy["protein"].x = x_prot

        #dti
        prot_drugs = self.dti(batch_copy.x_dict, batch_copy.edge_index_dict)
        x_drug_ppi_dti = prot_drugs['drug']

        return batch_copy, x_drug_ppi_dti