from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch


class MyDataset(Dataset):
    def __init__(self, active_dir, actigate_dir, label_dir):
        """
        :param actigate_dir:
        :param active_dir:
        """

        self.active_emb = np.load(active_dir, mmap_mode='r')
        self.actigate_emb = np.load(actigate_dir, mmap_mode='r')
        self.y = pd.read_csv(label_dir)['FIC'].apply(np.float)
        self.id = pd.read_csv(label_dir)['index'].apply(np.int64)
        self.domain = pd.read_csv(label_dir)['domain'].apply(np.int64)
        self.X = np.concatenate((self.active_emb, self.actigate_emb), axis=1)

    def __len__(self):
        return self.X.shape[0]
        pass

    def __getitem__(self, idx):
        return idx, self.id[idx], self.X[idx, :], self.y[idx], self.domain[idx]
        pass

def get_iterator(split_ratio=0.8, batch_size = 32,
                active_emb_dir="./data/active_mol/agg_mol_emb.npy",
                actigate_emb_dir= "./data/actigate_mol/mol_emb.npy",
                label_dir="./data/label.csv"):

    dataset = MyDataset(active_emb_dir, actigate_emb_dir, label_dir)
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    # train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size)

    return train_dataset, test_dataloader

if __name__ == '__main__':
    train_ds, test_dl = get_iterator()

