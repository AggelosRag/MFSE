import torch

from src.ddi_encoder import DDI_Encoder
from src.ppi_dti_encoder import PPI_DTI_Encoder
from src.m_encoder import M_Encoder
from src.intermidiate_multi_fusion import Multi_Inter_Fusion


class Encoder(torch.nn.Module):

    def __init__(self, device, in_dim_drug, num_dd_et, in_dim_prot,
                 uni_num_prot, uni_num_drug, dest_dict,
                 num_drug,
                 num_feat_per_atom,
                 prot_drug_dim=64, num_base=32,
                 n_embed=64, n_hid1=32, n_hid2=16, mod='cat'):

        super(Encoder, self).__init__()
        self.num_et = num_dd_et
        self.out_dim = n_hid2
        self.uni_num_drug = uni_num_drug
        self.uni_num_prot = uni_num_prot
        self.device = device

        # protein-target information encoder
        self.ppi_dti = PPI_DTI_Encoder(in_dim_prot, in_dim_drug, prot_drug_dim)

        # drug-drug interaction information encoder
        self.ddi = DDI_Encoder(n_embed, num_dd_et, num_base, dest_dict, device,
                       in_dim_drug)

        # molecular information encoder
        self.m = M_Encoder(num_drugs = num_drug,
                           num_feat_per_atom = num_feat_per_atom,
                           out_channels = 20,
                           device = self.device)

        # side-effect specific meta-fusion
        # self.nn_multi = Multi_Inter_Fusion(
        #                 in_channels = 96, ## ITAN 96!
        #                 hid = 32,
        #                 out_channels = 32,
        #                 num_relations = num_dd_et,
        #                 num_bases = 64,
        #                 device = device,
        #                 after_relu = False,
        # )

    def forward(self, train_batch):

        # protein-target information encoder
        batch_copy = train_batch.clone()
        batch_copy, x_drug_ppi_dti = self.ppi_dti(batch_copy)

        # drug-drug interaction information encoder
        batch_copy, x_drug_ddi = self.ddi(batch_copy)

        # molecular information encoder
        x_drug_m = self.m(batch_copy)

        # side-effect specific meta-fusion
        x_drug = torch.cat((x_drug_ddi, x_drug_ppi_dti), dim=1)
        x_drug = torch.cat((x_drug, x_drug_m), dim=1)
        #x_drug = self.nn_multi(x_drug, batch_copy.edge_index_dict)

        return x_drug
