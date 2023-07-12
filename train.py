import os
from tqdm import tqdm
from tqdm import trange
import torch
from torch import optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import torch.nn as nn
import copy
import numpy as np
from model import MT
from sklearn.metrics import r2_score

os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"


def train(train_dataloader, val_dataloader, model, optimizer, n_epoch=50, verbose=True):
    device = torch.device("cpu")
    model.train()
    for epoch in trange(1, n_epoch + 1, desc='Epochs', leave=False):
        train_loss = 0
        for batch_idx, (idx, id, x, y, d) in enumerate(tqdm(train_dataloader, desc='Batches', leave=False, disable=True)):
            x = x.flatten(start_dim=1).to(device).float()
            y = y.flatten(start_dim=0).to(device).float()
            optimizer.zero_grad()
            y_hat = model(x, d)
            optimizer.zero_grad()
            loss = model.get_loss(y, y_hat)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= (batch_idx + 1)
        if verbose:
            print('\nEpoch %d:' % epoch)
            print('==> Train | Average loss: %.4f' % train_loss)

        val_loss, r2 = test(val_dataloader, model)
        if verbose:
            print('==> Validation | Average loss: %.4f' % val_loss)
            print('===> r2 score %.4f' % r2)

    return val_loss, r2


def test(dataloader, model):
    device = torch.device("cpu")
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_idx, (idx, id, x, y, d) in enumerate(tqdm(dataloader, desc='Batches', leave=False)):
            x = x.flatten(start_dim=1).to(device).float()
            y = y.flatten(start_dim=0).to(device).float()
            y_hat = model(x, d)
            loss = model.get_loss(y, y_hat)
            test_loss += loss.item()
            r2 = r2_score(y, y_hat)

    return test_loss, r2


def cross_validation(train_ds, config, k_folds=5):
    torch.manual_seed(42)
    np.random.seed(42)
    n_epoch = 200
    model = MT(config['input_dim'], shared_layers_dim=config['shared_layers_dim'], hidden_dim=config['hidden_dim'], spec_layers_dim=config['spec_layers_dim'],
               output_dim=config['output_dim'], domain_n=config['domain_n'])
    model.loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    kfold = KFold(n_splits=k_folds, shuffle=True)
    models, results, r2dic = {}, {}, {}

    for fold, (train_ids, val_ids) in enumerate(kfold.split(train_ds)):
        print(f'FOLD {fold}')
        print('--------------------------------')

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_ids)
        train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], sampler=train_subsampler)
        val_dataloader = DataLoader(train_ds, batch_size=len(val_ids), sampler=val_subsampler)

        val_loss, r2 = train(train_dataloader, val_dataloader, model, optimizer, n_epoch, True)
        models[fold] = copy.deepcopy(model)
        results[fold] = val_loss
        r2dic[fold] = r2


    mean_result = np.array(list(results.values())).mean()
    mean_r2 = np.array(list(r2dic.values())).mean()

    print("mse", mean_result)
    print("r2", mean_r2)
    return models[k_folds-1]



def run(train_ds, test_dl):
    config = {"input_dim": 600, 'shared_layers_dim': [200], 'hidden_dim': 100, 'spec_layers_dim': [],
              "output_dim": 1, 'domain_n': 3, 'lr': 5e-5, 'drop_out': 0.2, 'batch_size': 64}
    model = cross_validation(train_ds, config=config)
    torch.save(model, 'botanical-regressor.pth')
    test(test_dl, model)

