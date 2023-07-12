import dgl
import errno
import numpy as np
import os
import torch
import pandas as pd

from dgl.nn.pytorch.glob import AvgPooling
from dgllife.model import load_pretrained
from dgllife.utils import mol_to_bigraph, PretrainAtomFeaturizer, PretrainBondFeaturizer
from rdkit import Chem
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def mkdir_p(path, log=True):
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def graph_construction_and_featurization(smiles):
    graphs = []
    success = []
    for smi in smiles:
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                success.append(False)
                continue
            g = mol_to_bigraph(mol, add_self_loop=True,
                               node_featurizer=PretrainAtomFeaturizer(),
                               edge_featurizer=PretrainBondFeaturizer(),
                               canonical_atom_order=False)
            graphs.append(g)
            success.append(True)
        except:
            success.append(False)
    return graphs, success


def collate(graphs):
    return dgl.batch(graphs)


def prertain(dataset, model_type, batch_size):
    data_loader = DataLoader(dataset, batch_size=batch_size,
                             collate_fn=collate, shuffle=False)

    model = load_pretrained(model_type).to(device)
    model.eval()
    readout = AvgPooling()

    mol_emb = []
    for batch_id, bg in enumerate(data_loader):
        bg = bg.to(device)
        nfeats = [bg.ndata.pop('atomic_number').to(device),
                  bg.ndata.pop('chirality_type').to(device)]
        efeats = [bg.edata.pop('bond_type').to(device),
                  bg.edata.pop('bond_direction_type').to(device)]
        with torch.no_grad():
            node_repr = model(bg, nfeats, efeats)
        mol_emb.append(readout(bg, node_repr))
    mol_emb = torch.cat(mol_emb, dim=0).detach().cpu().numpy()
    mol_emb_normed = (mol_emb - mol_emb.min(0)) / (mol_emb.max(0) - mol_emb.min(0)+1e-10)
    return mol_emb_normed


def active_aggregation(active_mol, active_df):

    active_df.rename({'Unnamed: 0': 'num'}, axis=1, inplace=True)
    num_pairs = active_df['index'][active_df.shape[0] - 1]

    agg_active_mol = []
    agg_active_mol_maxi = []
    for i in range(num_pairs):
        rows = active_df[active_df['index'] == i + 1]
        agg = 0
        frac = 0
        mol_arr = []
        for j in range(rows.shape[0]):
            fraction = rows['fraction'].iloc[j]
            num = rows['num'].iloc[j]
            mol = active_mol[num]
            agg += fraction * mol
            frac += fraction
            mol_arr.append(mol)
        agg_active_mol.append(agg/frac)
        mol_np = np.array(mol_arr)
        idx = mol_np.argmax(axis=0)
        agg_active_mol_maxi.append(idx)

    agg_active_mol = np.array(agg_active_mol)
    agg_active_mol_maxi = np.array(agg_active_mol_maxi)
    agg_active_mol_normed = (agg_active_mol - agg_active_mol.min(0)) / (agg_active_mol.max(0) - agg_active_mol.min(0)+1e-10)

    return agg_active_mol_maxi, agg_active_mol_normed


def pretrain_data(fpath, smi_col, model_type, batch_size, out_dir):
    mkdir_p(out_dir)
    df = pd.read_csv(fpath)
    smiles = df[smi_col].tolist()
    mols, _ = graph_construction_and_featurization(smiles)
    mol_emb = prertain(mols, model_type, batch_size)
    np.save(os.path.join(out_dir, 'mol_emb.npy'), mol_emb)
    if 'active' in fpath:
        agg_mol_emb_max, agg_mol_emb = active_aggregation(mol_emb, df)
        np.save(os.path.join(out_dir, 'agg_mol_emb.npy'), agg_mol_emb)
        np.save(os.path.join(out_dir, 'agg_mol_maxi.npy'), agg_mol_emb_max)

if __name__ == '__main__':
    pretrain_data(fpath="./data/actigate.csv", smi_col='smiles', model_type='gin_supervised_contextpred', batch_size=32, out_dir='./data/actigate_mol')



