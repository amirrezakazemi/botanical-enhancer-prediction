import pandas as pd
import os

def data_prep(f_dir="./data", data_dir="data/botanical_train_data_updated_05_2023.csv", max_comp=5):

    data = pd.read_csv(data_dir)
    data['active'] = data['active'].str.lower()
    data = data[data['active'] != "flavonoid extract"]
    data = data[data['active'] != "hot chili oil"]

    label_data = dict()
    label_data['index'], label_data['FIC'], label_data['domain'] = list(), list(), list()

    actigate_data, active_data = dict(), dict()

    actigate_data['index'], actigate_data['smiles'], actigate_data['actigate'] = list(), list(), list()
    active_data['index'], active_data['component_index'], active_data['active'], active_data['smiles'], active_data['fraction'], active_data['component'] = list(), list(), list(), list(), list(), list()

    for i in range(data.shape[0]):
        label_data['index'].append(i + 1)
        label_data['FIC'].append(data['FIC'].iloc[i])
        if data['disease_name'].iloc[i].lower() == 'botrytis':
            label_data['domain'].append(0)
        elif data['disease_name'].iloc[i].lower() == 'fusarium':
            label_data['domain'].append(1)
        elif data['disease_name'].iloc[i].lower() == 'sclerotinia':
            label_data['domain'].append(2)


        actigate_data['index'].append(i + 1)
        actigate_data['smiles'].append(data['smiles_actigate'].iloc[i])
        actigate_data['actigate'].append(data['actigate'].iloc[i])

        for j in range(max_comp):
            if (str(data['Component' + str(j + 1)].iloc[i]) != "nan"):
                active_data['index'].append(i + 1)
                active_data['component_index'].append(j + 1)
                active_data['active'].append(data['active'].iloc[i])
                active_data['component'].append(data['Component' + str(j + 1)].iloc[i])
                active_data['smiles'].append(data['smiles' + str(j + 1)].iloc[i])
                active_data['fraction'].append(data['Amount' + str(j + 1) + "(%)"].iloc[i])


    actigate_dataframe = pd.DataFrame.from_dict(actigate_data)
    actigate_dataframe.to_csv(os.path.join(f_dir, "actigate.csv"))
    active_dataframe = pd.DataFrame.from_dict(active_data)
    active_dataframe.to_csv(os.path.join(f_dir, "active.csv"))
    label_dataframe = pd.DataFrame.from_dict(label_data)
    label_dataframe.to_csv(os.path.join(f_dir, "label.csv"))


if __name__ == '__main__':
    data_prep()
