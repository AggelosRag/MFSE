from pytorch_lightning import LightningDataModule
from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
import numpy as np
import torch

from dataset.utils.load_mfse_interaction_g_data import load_mfse_interaction_g_data
from dataset.utils.load_mfse_m_data import \
    load_mfse_m_data

from dataset.utils.utils import degrees_per_node_train, process_edges2, sparse_id
from dataset.utils.neg_sampling import neg_sampling_train, neg_sampling_test

torch.manual_seed(1111)
np.random.seed(1111)
EPS = 1e-13

# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')


class MFSEDataModule(LightningDataModule):
    def __init__(self, sp_rate, fold, n_splits):
        super().__init__()
        #self.data = None
        self.train_mfse_data = None
        self.test_mfse_data = None
        self.sp_rate = sp_rate    # data split rate
        self.initial_training_data = None
        self.fold = fold
        self.n_splits = n_splits

        self.n_drug_feat = None
        self.n_dd_et = None
        self.n_prot = None
        self.n_drug = None
        self.dest_dict = None
        self.all_edges_undirected = None
        self.train_idx_list = None
        self.degree_per_node_undirected = None

        self.train_mfse_data, self.test_mfse_data = load_mfse_m_data()
        self.num_feat_per_atom = 69

        path = "./dataset/"
        data = load_mfse_interaction_g_data(path)

        self.n_drug = data['drug_num']
        self.n_prot = data['protein_num']
        self.n_dd_et = data['combo_num']
        self.n_drug_feat = data['drug_num']
        self.all_edges_undirected = data['dd_edge_index']

        self.train_mfse_data["protein"].x = sparse_id(data['protein_num'])
        self.test_mfse_data["protein"].x = sparse_id(data['protein_num'])
        self.train_mfse_data["drug"].x = sparse_id(data['drug_num'])
        self.test_mfse_data["drug"].x = sparse_id(data['drug_num'])

        self.train_mfse_data["protein", "connects_to", "drug"].edge_index = \
            data['dp_edge_index']
        self.test_mfse_data["protein", "connects_to", "drug"].edge_index = \
            data['dp_edge_index']

        train_list, test_list, dest_dict, train_idx_list, test_idx_list =\
            process_edges2(data['dd_edge_index'], data['drug_num'], fold, self.n_splits)

        self.train_mfse_data["protein", "interacts_with", "protein"].edge_index = \
            data['pp_edge_index']
        self.test_mfse_data["protein", "interacts_with", "protein"].edge_index = \
            data['pp_edge_index']

        for num in range(data['combo_num']):
            side_effect_name = f'side_effect_{num}'
            self.train_mfse_data['drug', side_effect_name, 'drug'].edge_index = \
                train_list[num]
            self.train_mfse_data['drug', side_effect_name, 'drug'].pos_edge_index = \
                train_list[num]
            self.test_mfse_data['drug', side_effect_name, 'drug'].edge_index = \
                test_list[num]
            self.test_mfse_data['drug', side_effect_name, 'drug'].pos_edge_index = \
                test_list[num]

        self.test_mfse_data = neg_sampling_test(data['dd_edge_index'],
                                                data['drug_num'], data['combo_num'],
                                                round((1 - self.sp_rate), 1), self.test_mfse_data,
                                                test_idx_list
                                                )

        self.train_idx_list = train_idx_list
        self.train_mfse_data = neg_sampling_train(data['dd_edge_index'],
                                                  data['drug_num'], data['combo_num'],
                                                  self.train_mfse_data, self.sp_rate,
                                                  self.test_mfse_data, train_idx_list)

        self.degree_per_node_undirected = degrees_per_node_train(
            self.train_mfse_data, self.n_drug)
        #check_edges(self.train_mfse_data.pos_edge_index_dict, self.test_mfse_data.pos_edge_index_dict)
        #check_edges(self.train_mfse_data.pos_edge_index_dict, self.test_mfse_data.neg_edge_index_dict)

        print(self.train_mfse_data)
        self.dest_dict = dest_dict
        self.initial_training_data = self.train_mfse_data.clone()


    def prepare_data(self):
        # called only on 1 GPU
        pass

    def train_dataloader(self):
        return DataLoader([self.train_mfse_data], shuffle=False,
                          num_workers=0, batch_size=20000)

    def val_dataloader(self):
        return DataLoader([self.test_mfse_data], shuffle=False,
                          num_workers=0, batch_size=20000)