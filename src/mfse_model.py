from collections import defaultdict

import numpy as np
import torch
import pytorch_lightning as pl
from sklearn import metrics

from dataset.data_module_new_full import \
    MFSEDataModule
from src.encoder import Encoder
from src.multi_inner_product_decoder2 import MultiInnerProductDecoder2
from dataset.utils.neg_sampling import neg_sampling_train
from src.utils import auprc_auroc_ap

# torch.manual_seed(1111)
# np.random.seed(1111)
EPS = 1e-13

class Setting(object):
    def __init__(self, sp_rate=0.9, lr=0.01, drug_target_dim=16, n_embed=48,
                 n_hid1=32, n_hid2=16, num_base=32) -> None:
        super().__init__()
        self.sp_rate = sp_rate  # data split rate
        self.lr = lr  # learning rate
        self.drug_target_dim = drug_target_dim  # dim of protein -> drug
        self.n_embed = n_embed  # dim of drug feature embedding
        self.n_hid1 = n_hid1  # dim of layer1's output on d-d graph
        self.n_hid2 = n_hid2  # dim of drug embedding
        self.num_base = num_base  # in decoder


class MFSE(pl.LightningModule):
    def __init__(self, settings: Setting, data_module: MFSEDataModule,
                 last_epoch, device, mod='cat') -> None:

        super().__init__()
        self.mod = mod
        assert mod in {'cat', 'add'}

        self.device_ = device
        self.settings = settings
        self.data_module = data_module
        self.last_epoch = last_epoch
        self.embeddings = None
        self.protein_drug_embs = None
        self.protein_embs = None
        self.drug_embs = None
        self.__prepare_model()
        self.testb = None
        self.record = None
        self.record_per_se_degree = None

    @property
    def device(self):
        return f'cuda:{self.device_}' if self.device_ != 'cpu' else 'cpu'

    def __prepare_model(self):
        # encoder
        self.encoder = Encoder(self.device, self.data_module.n_drug_feat,
                               self.data_module.n_dd_et, self.data_module.n_prot,
                               self.data_module.n_prot, self.data_module.n_drug,
                               self.data_module.dest_dict, self.data_module.n_drug,
                               self.data_module.num_feat_per_atom, self.settings.drug_target_dim,
                               self.settings.num_base, self.settings.n_embed,
                               self.settings.n_hid1, self.settings.n_hid2)

        # decoder
        self.decoder = MultiInnerProductDecoder2(
            # (self.settings.n_hid2),
            60,
            self.data_module.n_dd_et
        )
        # self.decoder = MultiInnerProductDecoder4(
        #     # (self.settings.n_hid2),
        #     32, #itan 64
        #     self.data_module.n_dd_et
        # )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.settings.lr)
        for var_name in optimizer.state_dict():
            print(var_name, "\t", optimizer.state_dict()[var_name])
        return optimizer

    def training_step(self, train_batch, batch_idx):

        # self.embeddings, self.protein_drug_embs, self.protein_embs,\
        #     self.drug_embs =  self.encoder(train_batch)
        # self.embeddings, self.protein_drug_embs, self.protein_embs = self.encoder(train_batch)
        self.embeddings = self.encoder(train_batch)

        #check_edges(train_batch.neg_edge_index_dict, self.testb.neg_edge_index_dict)
        #check_edges(train_batch.neg_edge_index_dict, self.testb.pos_edge_index_dict)

        # train_batch = typed_negative_sampling3(train_batch,
        #                                        self.data_module.n_drug)
        # train_batch = neg_s_2(self.data_module.all_edges_undirected,
        #                               self.data_module.n_drug,
        #                               self.data_module.n_dd_et,
        #                               train_batch, 0.1)
        train_batch = neg_sampling_train(self.data_module.all_edges_undirected,
                                         self.data_module.n_drug,
                                         self.data_module.n_dd_et,
                                         train_batch, 0.9, self.testb,
                                         self.data_module.train_idx_list,
                                         self.device)

        pos_score_list, pos_score = self.decoder(self.embeddings,
                                 train_batch.pos_edge_index_dict)
        neg_score_list, neg_score = self.decoder(self.embeddings,
                                 train_batch.neg_edge_index_dict)

        # pos_score = self.decoder(self.embeddings, pos_index,
        #                          self.data.dd_train_et)
        # neg_score = self.decoder(self.embeddings, neg_index,
        #                          self.data.dd_train_et)

        # pos_score = torch.cat(pos_score_list, dim=0)
        # neg_score = torch.cat(neg_score_list, dim=0)

        pos_loss = -torch.log(pos_score + EPS).mean()
        neg_loss = -torch.log(1 - neg_score + EPS).mean()
        loss = pos_loss + neg_loss

        record =  self.compute_auprc_auroc_ap_by_et(pos_score_list,
                                                    neg_score_list)

        [auprc, auroc, ap] = record.sum(axis=1) / self.data_module.n_dd_et
        print('\nOn training set: loss:{:0.4f}  auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    '.
            format(loss, auprc, auroc, ap))

        metrics = {"loss": loss, "train_auprc": auprc,
                   "train_auroc": auroc, "train_ap": ap}

        # since we use full batch gd, then step == epoch
        self.log_dict(metrics, logger=True, on_step=True,
                      on_epoch=True, prog_bar=False)

        # epochs = [0,1,2,3,10,20,30,40,50,60,70,80,90,98]
        # epochs = [0,1,2,98]
        # if self.current_epoch in epochs:
        #     self.add_embeddings(tag="embeddings",
        #                         embeddings=self.embeddings,
        #                         labels=self.data_module.n_drug)
        #     self.add_embeddings(tag="protein_as_drugs",
        #                         embeddings=self.protein_drug_embs,
        #                         labels=self.data_module.n_drug)
            # self.add_embeddings(tag="proteins",
            #                     embeddings=self.protein_embs,
            #                     labels=self.data_module.n_prot)
            # self.add_embeddings(tag="drugs",
            #                     embeddings=self.drug_embs,
            #                     labels=self.data_module.n_drug)

        return metrics

    def add_embeddings(self, tag, embeddings, labels):

        labels_list = [i for i in range(labels)]
        self.logger.experiment.add_embedding(tag=tag,
                                             mat=embeddings,
                                             metadata=labels_list,
                                             global_step=self.current_epoch)

    def training_epoch_end(self, metrics):

        # since we use full batch gd, then step == epoch
        metrics = metrics[0]
        metrics = {"loss": metrics["loss"].item(),
                   "train_auprc": metrics["train_auprc"],
                   "train_auroc": metrics["train_auroc"],
                   "train_ap": metrics["train_ap"]}
        # since we use full batch gd, then step == epoch
        self.log_dict(metrics, logger=True, on_step=False,
                      on_epoch=True, prog_bar=False)

        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)


    def validation_step(self, train_batch, batch_idx):

        self.testb = train_batch.clone()
        # since PL starts with a validation loop, do a forward pass with
        # model.eval() and torch.grad in the background but with the training
        # edges == message passing edges, only when the algorithm starts
        if self.embeddings == None:
            # self.embeddings, self.protein_drug_embs, self.protein_embs, \
            # self.drug_embs = self.encoder(
            #     self.data_module.initial_training_data.to(self.device))
            # self.embeddings, self.protein_drug_embs, self.protein_embs = \
            #     self.encoder( self.data_module.initial_training_data.to(self.device))
            self.embeddings = self.encoder( self.data_module.initial_training_data.to(self.device))

        pos_score_list, pos_score = self.decoder(self.embeddings,
                                 train_batch.pos_edge_index_dict)
        neg_score_list, neg_score = self.decoder(self.embeddings,
                                 train_batch.neg_edge_index_dict)
        # pos_score = self.decoder(self.embeddings, self.data.dd_test_idx,
        #                          self.data.dd_test_et)
        # neg_score = self.decoder(self.embeddings,
        #                          self.data_module.test_neg_index,
        #                          self.data.dd_test_et)

        # pos_score = torch.cat(pos_score_list, dim=0)
        # neg_score = torch.cat(neg_score_list, dim=0)

        pos_loss = -torch.log(pos_score + EPS).mean()
        neg_loss = -torch.log(1 - neg_score + EPS).mean()
        loss = pos_loss + neg_loss

        record =  self.compute_auprc_auroc_ap_by_et(pos_score_list,
                                                    neg_score_list)

        [auprc, auroc, ap] = record.sum(axis=1) / self.data_module.n_dd_et
        print('\nOn test set: loss:{:0.4f}  auprc:{:0.4f}   auroc:{:0.4f}   ap@50:{:0.4f}    '.
            format(loss, auprc, auroc, ap))

        metrics = {"test_loss": loss, "test_auprc": auprc,
                   "test_auroc": auroc, "test_ap": ap}

        # since we use full batch gd, then step == epoch
        self.log_dict(metrics, logger=True, on_step=True,
                      on_epoch=True, prog_bar=False)

        if self.current_epoch == self.last_epoch - 1:
            self.record = record
            record_per_se_degree = self.compute_auprc_auroc_ap_by_et_bin(
                pos_score_list, neg_score_list,
                train_batch.pos_edge_index_dict,
                train_batch.neg_edge_index_dict,
                self.data_module.degree_per_node_undirected
            )
            self.record_per_se_degree = self.default_to_regular(record_per_se_degree)

        return metrics

    def validation_epoch_end(self, metrics):

        # since we use full batch gd, then step == epoch
        metrics = metrics[0]
        metrics = {"test_loss": metrics["test_loss"].item(),
                   "test_auprc": metrics["test_auprc"],
                   "test_auroc": metrics["test_auroc"],
                   "test_ap": metrics["test_ap"]}
        self.log_dict(metrics, logger=True, on_step=False,
                      on_epoch=True, prog_bar=False)

    def compute_auprc_auroc_ap_by_et(self, pos_score_list, neg_score_list):
        record = np.zeros((3, self.data_module.n_dd_et))  # auprc, auroc, ap

        i = 0
        for pos, neg in zip(pos_score_list, neg_score_list):

            pos_target = torch.ones(pos.shape[0])
            neg_target = torch.zeros(neg.shape[0])

            score = torch.cat([pos, neg])
            target = torch.cat([pos_target, neg_target])

            record[0, i], record[1, i], record[2, i] = auprc_auroc_ap(target,
                                                                      score)
            i += 1

        return record


    def compute_auprc_auroc_ap_by_et_bin(self, pos_score_list, neg_score_list,
                                         pos_edge_index_dict, neg_edge_index_dict,
                                         deg_cluster_per_se):

        pos_idx_list_pairs = defaultdict(lambda: defaultdict(list))
        neg_idx_list_pairs = defaultdict(lambda: defaultdict(list))

        # results_per_se_node = defaultdict(
        #     lambda: defaultdict(lambda: defaultdict(list)))
        results_per_se_degree = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list)))
        count_pairs_per_se_degree = defaultdict(lambda: defaultdict(int))

        cnt = 0
        for pos, neg in zip(pos_score_list, neg_score_list):

            edge_type = 'drug', f'side_effect_{cnt}', 'drug'
            cnt += 1
            pos_edge_index = pos_edge_index_dict[edge_type]
            neg_edge_index = neg_edge_index_dict[edge_type]
           #edge_index = torch.cat([pos_edge_index, neg_edge_index], dim=1)
            pos_edge_type_list = pos_edge_index.T.detach().cpu().tolist()
            neg_edge_type_list = neg_edge_index.T.detach().cpu().tolist()

            pos_target = torch.ones(pos.shape[0])
            neg_target = torch.zeros(neg.shape[0])

            # score = torch.cat([pos, neg])
            # target = torch.cat([pos_target, neg_target])

            for i, pair in enumerate(pos_edge_type_list):
                pos_idx_list_pairs[edge_type][pair[0]].append(i)
                pos_idx_list_pairs[edge_type][pair[1]].append(i)

            for i, pair in enumerate(neg_edge_type_list):
                neg_idx_list_pairs[edge_type][pair[0]].append(i)
                neg_idx_list_pairs[edge_type][pair[1]].append(i)

            for node_id in range(self.data_module.n_drug):
                #deg_cluster_idx = deg_cluster_per_se[edge_type][node_id]
                deg_cluster_idx = deg_cluster_per_se[node_id]

                pos_idx_pairs = pos_idx_list_pairs[edge_type][node_id]
                neg_idx_pairs = neg_idx_list_pairs[edge_type][node_id]

                pos_node_score = pos[pos_idx_pairs].tolist()
                pos_node_target = pos_target[pos_idx_pairs].tolist()
                neg_node_score = neg[neg_idx_pairs].tolist()
                neg_node_target = neg_target[neg_idx_pairs].tolist()

                #results_per_se_node[edge_type][node_id]['y_pred'] = node_score
                #results_per_se_node[edge_type][node_id]['y_true'] = node_target

                results_per_se_degree[edge_type][deg_cluster_idx]['y_pred'].extend(pos_node_score)
                results_per_se_degree[edge_type][deg_cluster_idx]['y_true'].extend(pos_node_target)
                results_per_se_degree[edge_type][deg_cluster_idx]['y_pred'].extend(neg_node_score)
                results_per_se_degree[edge_type][deg_cluster_idx]['y_true'].extend(neg_node_target)
                results_per_se_degree[edge_type][deg_cluster_idx]['num_pos_edges_per_node'].append(
                    len(pos_node_target))
                results_per_se_degree[edge_type][deg_cluster_idx]['num_neg_edges_per_node'].append(
                    len(neg_node_target))
                count_pairs_per_se_degree[edge_type][deg_cluster_idx] += 1

            for degree_bin, res in results_per_se_degree[edge_type].items():
                if len(res['y_true']) == 0 or len(res['y_pred']) == 0:
                    results = {}
                else:
                    try:
                        auroc = metrics.roc_auc_score(res['y_true'], res['y_pred'])
                    except ValueError:
                        auroc = None
                    ap = metrics.average_precision_score(res['y_true'],res['y_pred'])
                    y, xx, _ = metrics.precision_recall_curve(res['y_true'],res['y_pred'])
                    auprc = metrics.auc(xx, y)
                    results = {'auroc': auroc, 'auprc': auprc, 'ap': ap}

                results_per_se_degree[edge_type][degree_bin]['results'] = results
                results_per_se_degree[edge_type][degree_bin]['num_pos_edges'] = \
                    sum(results_per_se_degree[edge_type][degree_bin]['num_pos_edges_per_node'])/2
                results_per_se_degree[edge_type][degree_bin]['num_neg_edges'] = \
                    sum(results_per_se_degree[edge_type][degree_bin]['num_neg_edges_per_node'])/2
                res.pop('y_true', None)
                res.pop('y_pred', None)

            for degree_bin, num_pairs in count_pairs_per_se_degree[edge_type].items():
                results_per_se_degree[edge_type][degree_bin]['num_nodes'] = num_pairs

        return results_per_se_degree

    def default_to_regular(self, d):
        if isinstance(d, defaultdict):
            d = {k: self.default_to_regular(v) for k, v in d.items()}
        return d