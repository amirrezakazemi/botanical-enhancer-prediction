import pandas as pd
import numpy as np
from captum.attr import FeaturePermutation, Lime, KernelShap
import torch
from sklearn.model_selection import train_test_split
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_top(model_dir='./botanical-regressor.pth',
              active_emb_dir='../data/pretrained/active_mol/agg_mol_emb.npy',
              actigate_emb_dir='../data/pretrained/actigate_mol/mol_emb.npy',
              label_dir='../data/label.csv'):

    model = torch.load(model_dir)

    active_emb = np.load(active_emb_dir, mmap_mode='r')
    actigate_emb = np.load(actigate_emb_dir, mmap_mode='r')
    y = pd.read_csv(label_dir)['FIC'].apply(np.float)
    id = pd.read_csv(label_dir)['index'].apply(np.int64)
    domain = pd.read_csv(label_dir)['domain'].apply(np.int64).to_numpy()
    X = np.concatenate((active_emb, actigate_emb), axis=1)

    torch.manual_seed(42)
    np.random.seed(42)

    X_train, X_test, y_train, y_test, id_train, id_test, d_train, d_test = train_test_split(X, y, id, domain,
                                                                                            test_size=0.2)
    y_hat_test = model(torch.from_numpy(X_test), torch.from_numpy(d_test))

    selection = np.concatenate(
        [np.expand_dims(id_test, 1), (y_hat_test.detach().numpy() - np.expand_dims(y_test, 1)) ** 2], axis=1)
    selection = selection[selection[:, 1].argsort()][:100, 0].astype(int)

    return selection, model, X[selection, :], id[selection], domain[selection]

def actigate_interpret(actigate_dir='./data/actigate.csv'):
    selection, _, _, _, _ = get_top()
    actigate = pd.read_csv(actigate_dir)
    actigate.iloc[selection + 1, :]['actigate'].to_csv('./data/interpret/actigate_interpret.csv')

def active_interpret(method_type="FP",
                    active_emb_maxi_dir='./data/active_mol/agg_mol_maxi.npy',
                     active_dir='./data/active.csv'):
    _, model, X_test, id_test, d_test = get_top()

    active = pd.read_csv(active_dir)
    agg_active_emb_idx = np.load(active_emb_maxi_dir)

    test_active_emb_idx = agg_active_emb_idx[id_test - 1]


    if method_type == 'FP':
        feature = FeaturePermutation(model)
    elif method_type == 'KS':
        feature = KernelShap(model)
    else:
        feature = Lime(model)

    X_attr, _ = feature.attribute((torch.from_numpy(X_test), torch.from_numpy(d_test)))
    active_attr = X_attr.detach().numpy()[:, 0:300]
    sorted_attr = np.argsort(active_attr)[:, ::-1][:, 0:30]

    comp_attr = []
    for i in range(test_active_emb_idx.shape[0]):
        comp_attr.append(test_active_emb_idx[i, sorted_attr[i, :]])
    comp_attr = np.array(comp_attr)
    comp = np.array([np.bincount(comp_attr[i]).argmax() for i in range(comp_attr.shape[0])])

    comp = np.concatenate([np.expand_dims(id_test, axis=1), np.expand_dims(comp + 1, axis=1)], axis=1)
    best = []
    for i in range(comp.shape[0]):
        s = active[(active['index'] == comp[i, 0]) & (active['component_index'] == comp[i, 1])]
        best.append(s.values[0].tolist())
    best = np.array(best)[:, 3:]
    unique_best, occ_best = np.unique(best, axis=0, return_counts=True)
    unique_best = np.concatenate([unique_best, np.expand_dims(occ_best, axis=1)], axis=1)
    pd.DataFrame(unique_best).to_csv(f'./data/interpret/{method_type}_interpret.csv')


if __name__ == "main":
    actigate_interpret()
    active_interpret()



