import os
import config_file as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import feature_extractor as fe
from sklearn.model_selection import train_test_split

import pandas_profiling as pp

from torch import Tensor
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
from sklearn.preprocessing import MaxAbsScaler
# Display setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#
# row_data = 'Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C'
#
#
# def pad_or_truncate(some_list, target_len=65):
#     return some_list[:target_len] + [0]*(target_len - len(some_list))
#
#
# # explicit function to normalize array
# def normalize(arr, t_min, t_max):
#     norm_arr = []
#     diff = t_max - t_min
#     diff_arr = max(arr) - min(arr)
#     for i in arr:
#         temp = (((i - min(arr)) * diff) / diff_arr) + t_min
#         norm_arr.append(temp)
#     return norm_arr
#
#
# def row_for_prediction(row_data):
#     lst_fp = []
#
#     row_fp = fe.fingerprint_features(row_data).ToBitString()
#     for letter in row_fp:
#         lst_fp.append(int(letter))
#
#     lst_smile = tokenizer.encode(row_data)
#     new_list_smile = pad_or_truncate(lst_smile, target_len=65)
#
#     # Normalizing data
#
#     range_to_normalize = (0, 1)
#     list_smile_normalize = normalize(new_list_smile, range_to_normalize[0], range_to_normalize[1])
#
#     return lst_fp, list_smile_normalize
#
#
# lst_fp, list_smile_normalize = row_for_prediction(row_data)
#
# # print(lst_fp)
# # print(list_smile_normalize)
# #
# # print(type(lst_fp))
# # print(type(list_smile_normalize))
#
# row_fp_tensor = torch.FloatTensor(lst_fp)
# row_smile_tensor = torch.FloatTensor(list_smile_normalize)
#


# exit()


data=config.path_single

def LoadExtract_transform(data):
    data_4_model = pd.read_csv(data)
    # extract feature
    data_4_model['smiles_features'] = data_4_model.smiles.apply(
        lambda x: np.array(fe.fingerprint_features(x)))

    # Data Exploration
    # data_4_model_model_1 = pp.ProfileReport(data_4_model)
    # data_4_model_model_1.to_file('profile_report_data_4_model_model_1.html')

    list_feature = []
    for i in range(len(data_4_model.smiles_features)):
        list_feature.append(data_4_model.smiles_features[i])

    new_df = pd.DataFrame(list(map(np.ravel, list_feature)))
    new_df.columns = new_df.columns.map(lambda x: 'fe_' + str(x))

    X_all = pd.concat([data_4_model, new_df], axis=1)

    # Data Exploration
    # report_X_all_model_1 = pp.ProfileReport(X_all)
    # report_X_all_model_1.to_file('templates/profile_report_X_all_model_1.html')

    X_all.drop(columns=['mol_id', 'smiles', 'smiles_features'], inplace=True)
    X = X_all.drop('P1', axis=1).values
    y = X_all['P1'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    X_train = X_train_new
    y_train = y_train_new

    X = torch.FloatTensor(X)
    y = torch.LongTensor(y)

    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    X_valid = torch.FloatTensor(X_valid)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    y_valid = torch.LongTensor(y_valid)

    return X_train, X_test, X_valid, y_train, y_test, y_valid

# data = './dataset_single.csv'
X_train, X_test, X_valid, y_train, y_test, y_valid = LoadExtract_transform(data)

# Convert all of our data into torch tensors, the required datatype for Pytorch model
# train_inputs = torch.tensor(X_train)
# train_labels = torch.tensor(y_train)
#
# validation_inputs = torch.tensor(X_valid)
# validation_labels = torch.tensor(y_valid)
#
# test_inputs = torch.tensor(X_test)
# test_labels = torch.tensor(y_test)

class ANN(nn.Module):
    # I define the model element below
    def __init__(self):
        super().__init__()
        # The first hidden layer
        self.fc1 = nn.Linear(in_features=2048, out_features=16)
        # The second hidden layer
        self.fc2 = nn.Linear(in_features=16, out_features=12)
        # The third hidden layer
        self.fc3 = nn.Linear(in_features=12, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=2)

    # forward propagate input
    def forward(self, x):
        # input to first hidden layer
        x = F.relu(self.fc1(x))
        # input to second hidden layer
        x = F.relu(self.fc2(x))
        # input to third hidden layer
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x


model = ANN()


############################# To make it generic : ##############################################

n_inputs = 2048
# ---------------------------------------------------
# You can change it by uncomment the two rows below
# ---------------------------------------------------
# print('Enter the number of feature :')
# n_inputs = input()

class ANN_generic(nn.Module):
    # I define the model element below
    def __init__(self, n_inputs):
        super().__init__()
        # The first hidden layer
        self.fc1 = nn.Linear(n_inputs, out_features=16)
        # The second hidden layer
        self.fc2 = nn.Linear(in_features=16, out_features=12)
        # The third hidden layer
        self.fc3 = nn.Linear(in_features=12, out_features=8)
        self.output = nn.Linear(in_features=8, out_features=2)

    # forward propagate input
    def forward(self, x):
        # input to first hidden layer
        x = F.relu(self.fc1(x))
        # input to second hidden layer
        x = F.relu(self.fc2(x))
        # input to third hidden layer
        x = F.relu(self.fc3(x))
        x = self.output(x)
        return x

# model_generic = ANN_generic()


################################# Model Training ################################################
def train_model(X_train, model):
    # Model Training
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 500
    loss_arr = []

    for i in range(epochs):
        y_hat = model.forward(X_train)
        loss = criterion(y_hat, y_train)
        loss_arr.append(loss)

        if i % 10 == 0:
            print(f'Epoch: {i} Loss: {loss}')

        optimizer.zero_grad()
        # loss.backward()
        # optimizer.step()

    print(loss_arr)

################################# Model Evaluation ################################################


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


# acc = evaluate_model(row_fp_tensor, model)
# print('###### ACCURACY AFTER EVALUATION!!! #####')
# print('Model Accuracy :')
# print(acc)
#
# exit()
################################ Model Testing ################################################


def testing_model(model, X_valid, y_valid):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # track test loss
    test_loss = 0.0

    model.eval()
    i = 1
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(X_valid)
    # calculate the batch loss
    loss = criterion(output, y_valid)
    # update test loss
    test_loss += loss.item() * len(X_valid)
    # convert output probabilities to predicted class
    _, pred = torch.max(output, 1)
    # compare predictions to true label
    correct_tensor = pred.eq(y_valid.data.view_as(pred))
    # print(correct_tensor)
    correct = np.squeeze(correct_tensor.numpy())

    # print(correct)
    df_correct = pd.DataFrame(correct, columns=['pred_testing'])
    df_correct.pred_testing = df_correct.pred_testing.replace({True: 1, False: 0})
    # print(df_correct)
    # g = df_correct.groupby(['pred_testing']).count()
    print('###### CONFUSION MATRIX AFTER TESTING!!! #####')
    confusion_matrix = df_correct['pred_testing'].value_counts().to_frame()
    confusion_matrix_dict = confusion_matrix.to_dict()
    # print(df_correct['pred_testing'].value_counts())
    print(confusion_matrix)
    print(confusion_matrix_dict)
    # average test loss
    test_loss = test_loss / len(X_valid)
    print('Test Loss: {:.6f}\n'.format(test_loss))
    return df_correct


# df_correct = testing_model(model, X_valid, y_valid)

# print(df_correct)


############################# Save Model ####################################

def save_model(model):

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    # model_scripted.save('../models/model_scripted_1.pt')  # Save

    # # Save and load the model via state_dict :
    # # Let’s save and load our model using just state_dict.
    #
    # # Specify a path
    # filepath = "../models/state_dict_model_1.pt"
    # # Save
    # torch.save(model.state_dict(), filepath)
    #
    # # Save and load entire model : Now let’s try the same thing with the entire model.
    # # Specify a path
    # filepath_entire = "../models/entire_model_1.pt"
    #
    # # Save
    # torch.save(model, filepath_entire)
    #
    # # # Load
    # # model = torch.load(PATH)
    # # model.eval()
    #
    # # # model = ANN()
    # # model.load_state_dict(torch.load(PATH))
    # # model.eval()
    # # print(model.eval())


############################# MAIN() ####################################

def main_model_1():
    X_train, X_test, X_valid, y_train, y_test, y_valid = LoadExtract_transform(data)
    model = ANN()

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    acc, dictionary_accuracy = evaluate_model(X_test, model)
    print('###### ACCURACY AFTER EVALUATION!!! #####')
    print('Model Accuracy :')
    print(acc)
    print(dictionary_accuracy)

    df_correct = testing_model(model, X_valid, y_valid)
    # print(df_correct)

    # save_model(model)


#
# # make a class prediction for one row of data
# def predict(row, model):
#     # convert row to data
#     # row = Tensor([row])
#     # make prediction
#     yhat = model(row)
#     # print(yhat)
#     # retrieve numpy array
#     yhat = yhat.detach().numpy()
#     return yhat
#
#
#
#
#
# yhat = predict(row_fp_tensor, model).argmax().item()
#
# dictionary_yhat = {'Model prediction': yhat}
# # print('predicted : ')
# print(dictionary_yhat)
# print('DONE!!!!')
# exit()
#
#
#
# print('*'*100)
# y_hat = model(row_fp_tensor).argmax().item()
# output_row = model(row_fp_tensor).argmax().item()
# print(y_hat)
# print('*'*100)
# exit()

if __name__ == "__main__":
    main_model_1()

