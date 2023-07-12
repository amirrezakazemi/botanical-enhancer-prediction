from train import run
from data_preprocessing import data_prep
from data_pretraining import pretrain_data
from data_iterator import get_iterator

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    data_prep()
    pretrain_data(fpath="./data/active.csv", smi_col='smiles', model_type='gin_supervised_contextpred', batch_size=32, out_dir='./data/active_mol')
    pretrain_data(fpath="./data/actigate.csv", smi_col='smiles', model_type='gin_supervised_contextpred', batch_size=32, out_dir='./data/actigate_mol')
    train_ds, test_dl = get_iterator()
    run(train_ds, test_dl)


