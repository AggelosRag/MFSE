from torchdrug.data import Molecule, Graph, PackedGraph
import pickle5 as pickle
import torch

with open('../molecular/drugbank/drugbank_stitch_full_SMILES.pkl', 'rb') as f:
    smiles_df = pickle.load(f)

#smiles_df = smiles_df[smiles_df['DrugBank ID'] != 'DB01373']
molecule_dict = {}
molecule_list = []

for index, row in smiles_df.iterrows():
    id = row["DrugBank ID"]
    smiles = row["SMILES"]
    try:
        mol = Molecule.from_smiles(smiles)
        molecule_dict[id] = mol
        molecule_list.append(mol)
    except:
        print(id)
        print(smiles)

final_dict = {}
for key,value in molecule_dict.items():
    edge_list = molecule_dict[key].edge_list.tolist()
    edge_index = {'0': [], '1': [], '2': [], '3': []}
    for edge in edge_list:
        bond_type = edge[2]
        edge_index[str(bond_type)].append(edge[0:2])
    final_dict[key] = {}
    final_dict[key]["node_feature"] = molecule_dict[key].node_feature
    final_dict[key]["edge_index"] = molecule_dict[key].edge_list[:, :2].t()
    if len(edge_index['2']) > 0:
        print(key)
    final_dict[key]["bond_edge_index"] = [torch.tensor(edge_index[k]).t() for k in sorted(edge_index.keys())]

import pickle
def save_to_pkl(path, obj):
    with open(path, 'wb') as ff:
        pickle.dump(obj, ff)

save_to_pkl("./molecule_graphs_full.pkl", final_dict)