from torch_geometric.data import Dataset, Data

class SideEffectMolecularDataset(Dataset):
    def __init__(self, drug_molecular_list, root=None,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.drug_molecular_list = drug_molecular_list

    def len(self):
        return len(self.drug_molecular_list)

    def get(self, idx):
        return Data(x = self.drug_molecular_list[idx]['node_feature'],
                    edge_index = self.drug_molecular_list[idx]['edge_list'])


class Bond0MolecularDataset(Dataset):
    def __init__(self, drug_molecular_list, root=None,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.drug_molecular_list = drug_molecular_list

    def len(self):
        return len(self.drug_molecular_list)

    def get(self, idx):
        return Data(x=self.drug_molecular_list[idx]['node_feature'],
                    edge_index=self.drug_molecular_list[idx]['bond_edge_index'][0])


class Bond1MolecularDataset(Dataset):
    def __init__(self, drug_molecular_list, root=None,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.drug_molecular_list = drug_molecular_list

    def len(self):
        return len(self.drug_molecular_list)

    def get(self, idx):
        return Data(x=self.drug_molecular_list[idx]['node_feature'],
                    edge_index=self.drug_molecular_list[idx]['bond_edge_index'][1])


class Bond2MolecularDataset(Dataset):
    def __init__(self, drug_molecular_list, root=None,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.drug_molecular_list = drug_molecular_list

    def len(self):
        return len(self.drug_molecular_list)

    def get(self, idx):
        return Data(x=self.drug_molecular_list[idx]['node_feature'],
                    edge_index=self.drug_molecular_list[idx]['bond_edge_index'][2])


class Bond3MolecularDataset(Dataset):
    def __init__(self, drug_molecular_list, root=None,
                 transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)

        self.drug_molecular_list = drug_molecular_list

    def len(self):
        return len(self.drug_molecular_list)

    def get(self, idx):
        return Data(x=self.drug_molecular_list[idx]['node_feature'],
                    edge_index=self.drug_molecular_list[idx]['bond_edge_index'][3])