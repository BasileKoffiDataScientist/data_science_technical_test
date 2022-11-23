import copy

import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import MaxAbsScaler


import config_file as config
import feature_extractor as fe
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from model_1 import *
from model_2 import *
from model_3 import *


# Display setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#
# data = config.path_single
# data_multi = config.path_multi
#
#
# def load_transform_data_model_1(data):
#     data_4_model = pd.read_csv(data)
#     # extract feature
#     data_4_model['smiles_features'] = data_4_model.smiles.apply(
#         lambda x: np.array(fe.fingerprint_features(x)))
#
#     list_feature = []
#     for i in range(len(data_4_model.smiles_features)):
#         list_feature.append(data_4_model.smiles_features[i])
#
#     new_df = pd.DataFrame(list(map(np.ravel, list_feature)))
#     new_df.columns = new_df.columns.map(lambda x: 'fe_' + str(x))
#
#     X_all = pd.concat([data_4_model, new_df], axis=1)
#     X_all.drop(columns=['mol_id', 'smiles', 'smiles_features'], inplace=True)
#     X = X_all.drop('P1', axis=1).values
#     y = X_all['P1'].values
#
#     return X, y
#
#
# def load_extract_transform_smile_string_model_2(data):
#     data_4_model = pd.read_csv(data)
#     # extract feature
#     data_4_model['smiles_string_features'] = data_4_model.smiles.apply(
#         lambda x: np.array(tokenizer.encode(x)))
#
#     list_simile_feature = []
#     for i in range(len(data_4_model.smiles_string_features)):
#         list_simile_feature.append(data_4_model.smiles_string_features[i])
#
#     new_df = pd.DataFrame(list(map(np.ravel, list_simile_feature)))
#     new_df.columns = new_df.columns.map(lambda x: 'ssf_' + str(x))
#
#     X_all = pd.concat([data_4_model, new_df], axis=1)
#
#     X_all.drop(columns=['mol_id', 'smiles', 'smiles_string_features'], inplace=True)
#     # Fill nan by 0
#     X_all = X_all.fillna(0)
#
#     # Normalizing data because of feeling missing data
#     scaler = MaxAbsScaler()
#     scaler.fit(X_all)
#     scaled = scaler.transform(X_all)
#     scaled_df = pd.DataFrame(scaled, columns=X_all.columns)
#     # print(scaled_df)
#     X = scaled_df.drop('P1', axis=1).values
#     y = scaled_df['P1'].values
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
#
#     # print(pd.DataFrame(X_train).fillna(0))
#
#     X_train = X_train_new
#     y_train = y_train_new
#
#     X = torch.FloatTensor(X)
#     y = torch.LongTensor(y)
#
#     X_train = torch.FloatTensor(X_train)
#     X_test = torch.FloatTensor(X_test)
#     X_valid = torch.FloatTensor(X_valid)
#
#     y_train = torch.LongTensor(y_train)
#     y_test = torch.LongTensor(y_test)
#     y_valid = torch.LongTensor(y_valid)
#
#     return X_train, X_test, X_valid, y_train, y_test, y_valid
#
#
# class ANN(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.fc1 = nn.Linear(in_features=2048, out_features=16)
#         self.fc2 = nn.Linear(in_features=16, out_features=12)
#         self.fc3 = nn.Linear(in_features=12, out_features=8)
#         self.output = nn.Linear(in_features=8, out_features=2)
#
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.output(x)
#         return x
# model = ANN()
#
#
# def splitdataset(X, y):
#     # We use sklearn's train_test_split to split data first time
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
#     # We use sklearn's train_test_split to split data second time
#     X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
#     X_train = X_train_new # copy.deepcopy(X_train_new)
#     y_train = y_train_new # copy.deepcopy(y_train_new)
#
#     # Transform data to Tensor
#     X = torch.FloatTensor(X)
#     y = torch.LongTensor(y)
#
#     X_train = torch.FloatTensor(X_train)
#     X_test = torch.FloatTensor(X_test)
#     X_valid = torch.FloatTensor(X_valid)
#
#     y_train = torch.LongTensor(y_train)
#     y_test = torch.LongTensor(y_test)
#     y_valid = torch.LongTensor(y_valid)
#
#     return X_train, X_test, X_valid, y_train, y_test, y_valid
#
#
# def save_model_on_disk(model, path=config.MODEL_OUTPUT):
#     pass
#
#
# def load_model(path):
#     pass
#
#
# def train_model():
#     pass
#

def evaluate_model(X_test, model):
    preds = []
    with torch.no_grad():
        for val in X_test:
            y_hat = model.forward(val)
            preds.append(y_hat.argmax().item())

    df = pd.DataFrame({'Y': y_test, 'YHat': preds})
    df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]

    acc = df['Correct'].sum() / len(df)
    dictionary_accuracy = {'Model Accuracy': acc}
    # print(dictionary_accuracy)
    # acc = accuracy_score(df['Y'], df['YHat'])

    return acc, dictionary_accuracy

#
# def evaluate(y_true, y_pred):
#     pass
#
#
# def test():
#     pass

def main():

    print('[INFO] DONE...')


if __name__ == "__main__":
    main()




