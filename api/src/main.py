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
# from model_1 import *
# from model_2 import *
# from model_3 import *


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
#
# def evaluate_model(X_test, model):
#     preds = []
#     with torch.no_grad():
#         for val in X_test:
#             y_hat = model.forward(val)
#             preds.append(y_hat.argmax().item())
#
#     df = pd.DataFrame({'Y': y_test, 'YHat': preds})
#     df['Correct'] = [1 if corr == pred else 0 for corr, pred in zip(df['Y'], df['YHat'])]
#
#     acc = df['Correct'].sum() / len(df)
#     dictionary_accuracy = {'Model Accuracy': acc}
#     # print(dictionary_accuracy)
#     # acc = accuracy_score(df['Y'], df['YHat'])
#
#     return acc, dictionary_accuracy

#
# def evaluate(y_true, y_pred):
#     pass
#
#
# def test():
#     pass

#
# print('[INFO: Prediction start here!!!!]')
model_1 = torch.jit.load('../models/model_scripted_1.pt')
model_1.eval()
# print(model_1)

model_2 = torch.jit.load('../models/model_scripted_2.pt')
model_2.eval()
# print(model_2)

# exit()

row_data = 'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C'


def pad_or_truncate(some_list, target_len=65):
    return some_list[:target_len] + [0]*(target_len - len(some_list))


# explicit function to normalize array
def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr)) * diff) / diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def row_for_prediction(row_data):
    lst_fp = []

    row_fp = fe.fingerprint_features(row_data).ToBitString()
    for letter in row_fp:
        lst_fp.append(int(letter))

    lst_smile = tokenizer.encode(row_data)
    new_list_smile = pad_or_truncate(lst_smile, target_len=65)

    # Normalizing data

    range_to_normalize = (0, 1)
    list_smile_normalize = normalize(new_list_smile, range_to_normalize[0], range_to_normalize[1])

    return lst_fp, list_smile_normalize


# lst_fp, list_smile_normalize = row_for_prediction(row_data)

# print(lst_fp)
# print(list_smile_normalize)
#
# print(type(lst_fp))
# print(type(list_smile_normalize))

# row_fp_tensor = torch.FloatTensor(lst_fp)
# row_smile_tensor = torch.FloatTensor(list_smile_normalize)





# make a class prediction for one row of data
def predict(row, model):
    # convert row to data
    # row = Tensor([row])
    # make prediction
    yhat = model(row)
    # retrieve numpy array
    yhat = yhat.detach().numpy()
    return yhat




# # Model 1
# yhat = predict(row_fp_tensor, model_1).argmax().item()
# dictionary_yhat = {'Model 1 prediction by fingerprint': yhat}
# # print('predicted : ')
# print(dictionary_yhat)
# print('DONE!!!!')
# print('[INFO: Prediction finished here!!!!]')
#
#
# # Model 2
# yhat_2 = predict(row_smile_tensor, model_2).argmax().item()
#
# dictionary_yhat_2 = {'Model 2 prediction by smile string charactere': yhat_2}
# # print('predicted : ')
# print(dictionary_yhat_2)
# print('DONE AGAIN !!!!')
# exit()


#
# print('*'*100)
# y_hat = model_1(row_fp_tensor).argmax().item()
# output_row = model_1(row_fp_tensor).argmax().item()
# print(y_hat)
#
# dictionary_yhat = {'Model prediction': yhat}
# print(dictionary_yhat)
# print('*'*100)
# exit()
# # exit()



def main(data):
    lst_fp, list_smile_normalize = row_for_prediction(data)
    row_fp_tensor = torch.FloatTensor(lst_fp)
    row_smile_tensor = torch.FloatTensor(list_smile_normalize)

    print('[INFO: Model 1 prediction start here!!!!]')

    prediction_1 = predict(row_fp_tensor, model_1).argmax().item()
    data_1 = {'prediction': prediction_1, 'class_name': str(prediction_1)}
    print('[INFO: Model 1 Prediction finished here!!!!]')

    print('*'*100)

    print('[INFO: Model 2 prediction start here!!!!]')

    prediction_2 = predict(row_smile_tensor, model_2).argmax().item()
    data_2 = {'prediction': prediction_2, 'class_name': str(prediction_2)}
    print('[INFO: Model 2 Prediction finished here!!!!]')


    data_final = {
        'data_1' : {'prediction': prediction_1, 'class_name': str(prediction_1)},
        'data_2': {'prediction': prediction_2, 'class_name': str(prediction_2)},

    }

    print(data_final)

    print('[INFO] DONE...DONE...DONE...DONE...DONE...DONE...DONE...DONE...DONE...')


def predict_final(data):
    lst_fp, list_smile_normalize = row_for_prediction(data)
    row_fp_tensor = torch.FloatTensor(lst_fp)
    row_smile_tensor = torch.FloatTensor(list_smile_normalize)

    # print('[INFO: Model 1 prediction start here!!!!]')
    # Model 1
    # yhat = predict(row_fp_tensor, model_1).argmax().item()
    prediction_1 = predict(row_fp_tensor, model_1).argmax().item()
    data_1 = {'prediction': prediction_1, 'class_name': str(prediction_1)}

    # print(data_1)


    # dictionary_yhat = {'Model 1 prediction by fingerprint': yhat}
    # # print('predicted : ')
    # print(dictionary_yhat)
    # print('DONE!!!!')
    # print('[INFO: Model 1 Prediction finished here!!!!]')

    # print('*'*100)

    # print('[INFO: Model 2 prediction start here!!!!]')

    # Model 2
    # yhat_2 = predict(row_smile_tensor, model_2).argmax().item()
    prediction_2 = predict(row_smile_tensor, model_2).argmax().item()
    data_2 = {'prediction': prediction_2, 'class_name': str(prediction_2)}

    # print(data_2)

    # dictionary_yhat_2 = {'Model 2 prediction by smile string charactere': yhat_2}
    # # print('predicted : ')
    # print(dictionary_yhat_2)
    # print('DONE AGAIN !!!!')
    # print('[INFO: Model 2 Prediction finished here!!!!]')
    # exit()

    data_final = {
        'model_1' : {'prediction': prediction_1, 'class_name': str(prediction_1)},
        'model_2': {'prediction': prediction_2, 'class_name': str(prediction_2)},

    }

    # print(data_final)

    # print('[INFO] DONE...DONE...DONE...DONE...DONE...DONE...DONE...DONE...DONE...')
    return data_final


def predict_for_api(data):
    lst_fp, list_smile_normalize = row_for_prediction(data)
    row_fp_tensor = torch.FloatTensor(lst_fp)
    row_smile_tensor = torch.FloatTensor(list_smile_normalize)

    prediction_1 = predict(row_fp_tensor, model_1).argmax().item()

    prediction_2 = predict(row_smile_tensor, model_2).argmax().item()
    return prediction_1, prediction_2

if __name__ == "__main__":
    print('Entrez votre molecule ici :')
    # data = input()
    # main('Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C')
    data = 'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C'
    output = predict_final(data)
    print(output)




