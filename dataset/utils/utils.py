from collections import defaultdict
import pickle
import torch
import pandas as pd
from scipy import sparse as sp
from sklearn.model_selection import KFold
from torch_geometric.utils import degree

torch.manual_seed(0)


def save_to_pkl(path, obj):
    with open(path, 'wb') as g:
        pickle.dump(obj, g)


def get_drug_index_from_text(code):
    return int(code.split('D')[-1])


def get_side_effect_index_from_text(code):
    return int(code.split('C')[-1])


def degrees_per_node_train(train_data, num_nodes):
## For each node, find both degrees for each se as well as total degree

    degrees_per_se = defaultdict(list)
    deg_cluster_per_se = defaultdict(lambda: defaultdict(int))

    degrees_tot = [ 0 for _ in range(num_nodes)]
    deg_cluster_tot = defaultdict(int)
    count_pairs_per_degree = defaultdict(int)

    for edge_name, pos_train_edges in train_data.pos_edge_index_dict.items():

        neg_train_edges = train_data.neg_edge_index_dict[edge_name]
        train_edges = torch.cat([pos_train_edges, neg_train_edges], dim=1)

        row, col = train_edges
        deg = degree(row, num_nodes)
        deg = deg.int().tolist()
        degrees_per_se[edge_name] = deg
        degrees_tot = [x + y for x, y in zip(deg, degrees_tot)]

        # clusters = [0,5,10,20,50]
        clusters = [0,50]
        for node_id, degr in enumerate(deg):
            for i, degree_bound in enumerate(clusters):
                if degr <= degree_bound:
                    deg_cluster_per_se[edge_name][node_id] = i
                    break
                else:
                    if i == len(clusters) - 1:
                        deg_cluster_per_se[edge_name][node_id] = i+1

    deg_clusters, binsq = pd.qcut(degrees_tot, 3, labels=False, retbins=True)
    for node_id, degr in enumerate(deg_clusters):
        deg_cluster_tot[node_id] = degr

    # clusters = [100,1000,2000,3000,5000]
    # for node_id, degr in enumerate(degrees_tot):
    #     for i, degree_bound in enumerate(clusters):
    #         if degr <= degree_bound:
    #             deg_cluster_tot[i].append(node_id)
    #             count_pairs_per_degree[i] += 1
    #             break
    #         else:
    #             if i == len(clusters) - 1:
    #                 deg_cluster_tot[i + 1].append(node_id)
    #                 count_pairs_per_degree[i + 1] += 1

    return deg_cluster_tot


def remove_bidirection(edge_index, edge_type):
    mask = edge_index[0] > edge_index[1]
    keep_set = mask.nonzero().view(-1)

    if edge_type is None:
        return edge_index[:, keep_set]
    else:
        return edge_index[:, keep_set], edge_type[keep_set]


def to_bidirection(edge_index, edge_type=None):
    tmp = edge_index.clone()
    tmp[0, :], tmp[1, :] = edge_index[1, :], edge_index[0, :]
    if edge_type is None:
        return torch.cat([edge_index, tmp], dim=1)
    else:
        return torch.cat([edge_index, tmp], dim=1), torch.cat(
            [edge_type, edge_type])


def process_edges2(raw_edge_list, drug_num, fold, n_splits):
    train_list = []
    train_idx_list = []
    test_list = []
    test_idx_list = []
    train_label_list = []
    test_label_list = []
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    for i, idx in enumerate(raw_edge_list):

        train_folds = []
        test_folds = []
        for train_index, test_index in kf.split(idx.T):

            train_folds.append(train_index)
            test_folds.append(test_index)

        train_list.append(idx.T[train_folds[fold]])
        train_idx_list.append(train_folds[fold])
        test_list.append(idx.T[test_folds[fold]])
        test_idx_list.append(test_folds[fold])

        train_label_list.append(
            torch.ones(2 * train_folds[fold].size, dtype=torch.long) * i)
        test_label_list.append(
            torch.ones(2 * test_folds[fold].size, dtype=torch.long) * i)

    train_list = [to_bidirection(idx.T) for idx in train_list]
    test_list = [to_bidirection(idx.T) for idx in test_list]
    destination_idx_train = torch.cat(train_list, dim=1).tolist()[1]

    list = []
    for i in range(drug_num):
        num = destination_idx_train.count(i)
        if num == 0:
            num = 1
        list.append([num])
    train_destinations = torch.tensor(list, dtype=torch.float32)

    return train_list, test_list, train_destinations, train_idx_list, test_idx_list


def sparse_id(n):
    idx = [[i for i in range(n)], [i for i in range(n)]]
    val = [1 for i in range(n)]
    i = torch.LongTensor(idx)
    v = torch.FloatTensor(val)
    shape = (n, n)

    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def dense_id(n):
    idx = [i for i in range(n)]
    val = [1 for i in range(n)]
    out = sp.coo_matrix((val, (idx, idx)), shape=(n, n), dtype=float)

    return torch.Tensor(out.todense())