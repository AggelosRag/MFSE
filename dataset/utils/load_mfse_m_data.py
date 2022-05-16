import pickle
import pandas as pd
from torch_geometric.data import HeteroData

from dataset.drug_molecular_dataset import \
    SideEffectMolecularDataset, Bond0MolecularDataset, Bond1MolecularDataset, \
    Bond2MolecularDataset, Bond3MolecularDataset


def load_mfse_m_data():

    path = './dataset/'

    print("loading molecular data")

    # load graph info
    with open(path + 'molecular/molecule_graphs_full.pkl', 'rb') as f:
        molecules_dict = pickle.load(f)

    # map the drugbank ids to the drug nodes mapping and sort the dict
    with open('./dataset/index_map/mapping_dicts/new_drug_map.pkl', 'rb') as f:
        new_drug_map = pickle.load(f)

    df = pd.read_pickle(
        './dataset/molecular/drugbank/drugbank_stitch_full.pkl')
    #df = df[df['STITCH ID'] != 271]
    df["STITCH ID"] = df["STITCH ID"].apply(lambda x: new_drug_map[x])

    train_mfse_data = HeteroData()
    test_mfse_data = HeteroData()

    drug_molecular_dict = {}
    for id, dict in molecules_dict.items():
        drug_id = df.loc[df['DrugBank ID'] == id, 'STITCH ID'].item()

        drug_molecular_dict[drug_id] = dict
        atom_nodes = f'atoms_drug_{drug_id}'

        train_mfse_data[atom_nodes].x = dict['node_feature']
        test_mfse_data[atom_nodes].x = dict['node_feature']
        train_mfse_data[atom_nodes, "bonds_with", atom_nodes].edge_index = \
            dict['edge_index']
        test_mfse_data[atom_nodes, "bonds_with", atom_nodes].edge_index = \
            dict['edge_index']

    drug_molecular_list =\
            [drug_molecular_dict[k] for k in sorted(drug_molecular_dict.keys())]

    dataset = SideEffectMolecularDataset(
        drug_molecular_list = drug_molecular_list)
    datasetb0 = Bond0MolecularDataset(
        drug_molecular_list = drug_molecular_list)
    datasetb1 = Bond1MolecularDataset(
        drug_molecular_list = drug_molecular_list)
    datasetb2 = Bond2MolecularDataset(
        drug_molecular_list = drug_molecular_list)
    datasetb3 = Bond3MolecularDataset(
        drug_molecular_list = drug_molecular_list)

    train_mfse_data["molecules"].dataset = dataset
    test_mfse_data["molecules"].dataset = dataset
    train_mfse_data["molecules"].datasetb0 = datasetb0
    test_mfse_data["molecules"].datasetb0 = datasetb0
    train_mfse_data["molecules"].datasetb1 = datasetb1
    test_mfse_data["molecules"].datasetb1 = datasetb1
    train_mfse_data["molecules"].datasetb2 = datasetb2
    test_mfse_data["molecules"].datasetb2 = datasetb2
    train_mfse_data["molecules"].datasetb3 = datasetb3
    test_mfse_data["molecules"].datasetb3 = datasetb3

    return train_mfse_data, test_mfse_data

