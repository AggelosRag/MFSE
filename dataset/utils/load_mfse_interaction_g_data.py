import torch
import pickle


def load_mfse_interaction_g_data(path):

    print("loading data")

    # load graph info
    with open(path + 'index_map/statistics.pkl', 'rb') as f:
        drug_num, protein_num, combo_num = pickle.load(f)

    pp_edge_index = torch.load(path + "index_map/pp_edge_index.pt")
    dp_edge_index = torch.load(path + "index_map/dp_edge_index.pt")

    dd_edge_index = []
    for side_effect_id in range(combo_num):
        edge_index = torch.load(path +
                   f"index_map/dd_edge_index/edge_index_{side_effect_id}.pt")
        dd_edge_index.append(edge_index)

    # return a dict
    data = {'pp_edge_index': pp_edge_index,
            'dp_edge_index': dp_edge_index,
            'dd_edge_index': dd_edge_index,
            'drug_num': drug_num,
            'protein_num': protein_num,
            'combo_num': combo_num
            }

    print('data has been loaded')

    return data