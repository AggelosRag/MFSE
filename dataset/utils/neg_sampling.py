import numpy as np
import torch
from torch import Tensor
from torch_geometric.utils import negative_sampling

from dataset.utils.utils import to_bidirection
import random

random.seed(0)

def check_edges(negative_edges, positive_edges):

    for edge_type, p_edge_index in positive_edges.items():
        n_edge_index = negative_edges[edge_type]
        common_el, bool_list = common_elements(p_edge_index, n_edge_index)
        if bool_list:
            print(f'found common element: {edge_type}, {common_el}')

def common_elements(a,b):
    a = a.cpu().T.tolist()
    b = b.cpu().T.tolist()
    common_el = []
    bool_list = []
    for i in b:
        if i in a:
            common_el.append(i)
            bool_list.append(True)
        else:
            bool_list.append(False)
    return common_el, any(bool_list)

def neg_sampling_test(pos_edge_index_full_list, drug_num, combo_num,
                      num_t, model_test_data, test_idx_list):

    for num in range(combo_num):

        side_effect_name = f'side_effect_{num}'
        pos_edge_index_full = pos_edge_index_full_list[num]
        num_edges = pos_edge_index_full.shape[1]

        if isinstance(num_t, float):
            num_test = round(num_t * num_edges)

        # x1, x2, x3 = structured_negative_sampling(
        #     edge_index=to_bidirection(pos_edge_index_full),
        #     num_nodes=drug_num,
        #     contains_neg_self_loops=False
        # )
        # neg_index = torch.stack([x1, x3], dim=0)
        # positive_edges = pos_edge_index_full.T[test_idx_list[num]].T
        # neg_index = neg_index.T[test_idx_list[num]].T
        # neg_index = to_bidirection(neg_index)

        neg_index = negative_sampling(to_bidirection(pos_edge_index_full),
                                      num_nodes=drug_num,
                                      num_neg_samples=num_test)
        neg_index = to_bidirection(neg_index)

        model_test_data['drug', side_effect_name, 'drug'].neg_edge_index = neg_index

    return model_test_data

def neg_sampling_train(pos_edge_index_full_list, drug_num, combo_num,
                       model_train_data, num_t, model_test_data,
                       train_idx_list, device=None):

    num = 0
    for edge_name, neg_edge in model_test_data.neg_edge_index_dict.items():

        pos_edge_index_full = pos_edge_index_full_list[num]
        if device is not None:
            pos_edge_index_full = pos_edge_index_full.to(device)
        edges = torch.cat([to_bidirection(pos_edge_index_full),
                           to_bidirection(neg_edge)], dim=1)
        num_edges = pos_edge_index_full.shape[1]

        if isinstance(num_t, float):
            num_train = round(num_t * num_edges)

        # x1, x2, x3 = structured_negative_sampling(
        #     edge_index=edges,
        #     num_nodes=drug_num,
        #     contains_neg_self_loops=False
        # )
        # neg_index = torch.stack([x1, x3], dim=0)
        # positive_edges = pos_edge_index_full.T[train_idx_list[num]].T
        # neg_index = neg_index.T[train_idx_list[num]].T
        # neg_index = to_bidirection(neg_index)

        neg_index = negative_sampling(edges, num_nodes=drug_num,
                                      num_neg_samples=num_train)
        neg_index = to_bidirection(neg_index)

        model_train_data[edge_name].neg_edge_index = neg_index
        num+=1

    return model_train_data

def structured_negative_sampling(edge_index, num_nodes: int = None,
                                 contains_neg_self_loops: bool = True):
    r"""Samples a negative edge :obj:`(i,k)` for every positive edge
    :obj:`(i,j)` in the graph given by :attr:`edge_index`, and returns it as a
    tuple of the form :obj:`(i,j,k)`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
        contains_neg_self_loops (bool, optional): If set to
            :obj:`False`, sampled negative edges will not contain self loops.
            (default: :obj:`True`)

    :rtype: (LongTensor, LongTensor, LongTensor)
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    row, col = edge_index.cpu()
    pos_idx = row * num_nodes + col
    if not contains_neg_self_loops:
        loop_idx = torch.arange(num_nodes) * (num_nodes + 1)
        pos_idx = torch.cat([pos_idx, loop_idx], dim=0)

    rand = torch.randint(num_nodes, (row.size(0), ), dtype=torch.long)
    neg_idx = row * num_nodes + rand

    mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
    rest = mask.nonzero(as_tuple=False).view(-1)
    while rest.numel() > 0:  # pragma: no cover
        tmp = torch.randint(num_nodes, (rest.size(0), ), dtype=torch.long)
        rand[rest] = tmp
        neg_idx = row[rest] * num_nodes + tmp

        mask = torch.from_numpy(np.isin(neg_idx, pos_idx)).to(torch.bool)
        rest = rest[mask]

    return edge_index[0], edge_index[1], rand.to(edge_index.device)

def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))