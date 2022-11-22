import os

# import wandb
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MaxAbsScaler
import config_file as config
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import feature_extractor as fe
from sklearn.model_selection import train_test_split

from torch.optim import AdamW
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import copy
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import time
import copy
#
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#

from deepchem.feat.smiles_tokenizer import SmilesTokenizer
from transformers import BertTokenizer
# current_dir = os.path.dirname(os.path.realpath(__file__))
# vocab_path = os.path.join(current_dir, 'tests/data', 'vocab.txt')
# tokenizer = SmilesTokenizer(vocab_path)

# Display setting
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


data=config.path_multi

# model = ANN(*args, **kwargs)
# model.load_state_dict(torch.load(PATH))
# model.eval()
# best_model_wts = copy.deepcopy(model.state_dict())
# print(model)
# exit()

def target_data(data):
    df = pd.read_csv(data)
    df_target = df[['P2',  'P1',  'P3',  'P4',  'P5',  'P6',  'P7',  'P8',  'P9']]

    return df_target

df_target = target_data(data)
print(df_target)

#
# exit()

def load_extract_transform_smile_string(data):
    data_4_model = pd.read_csv(data)
    # extract feature
    data_4_model['smiles_string_features'] = data_4_model.smiles.apply(
        lambda x: np.array(tokenizer.encode(x)))

    list_simile_feature = []
    for i in range(len(data_4_model.smiles_string_features)):
        list_simile_feature.append(data_4_model.smiles_string_features[i])

    new_df = pd.DataFrame(list(map(np.ravel, list_simile_feature)))
    new_df.columns = new_df.columns.map(lambda x: 'ssf_' + str(x))

    X_all = pd.concat([data_4_model, new_df], axis=1)

    # print(X_all)
    # exit()

    X_all.drop(columns=['mol_id', 'smiles', 'smiles_string_features'], inplace=True)
    X_all = X_all.fillna(0)
    print(X_all)
    print(len(X_all.columns))
    exit()

    # Normalizing data
    scaler = MaxAbsScaler()
    scaler.fit(X_all)
    scaled = scaler.transform(X_all)
    scaled_df = pd.DataFrame(scaled, columns=X_all.columns)
    # print(scaled_df)
    X = scaled_df.drop('P1', axis=1).values
    y = scaled_df['P1'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_new, X_valid, y_train_new, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

    # print(pd.DataFrame(X_train).fillna(0))

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


X_train, X_test, X_valid, y_train, y_test, y_valid = load_extract_transform_smile_string(data)

def extractfeatures(data):
    data['smiles_features'] = data.smiles.apply(
        lambda x: np.array(fe.fingerprint_features(x)))

    list_feature = []
    for i in range(len(data.smiles_features)):
        list_feature.append(data.smiles_features[i])

    new_df = pd.DataFrame(list(map(np.ravel, list_feature)))
    new_df.columns = new_df.columns.map(lambda x: 'fe_' + str(x))
    print(new_df)

    return new_df

new_df = extractfeatures(pd.read_csv(data))

# print(new_df)
#
# exit()
def load_extract_transform(data):
    data_4_model = pd.read_csv(data)
    # extract feature
    new_df = extractfeatures(data_4_model)
    # data_4_model['smiles_features'] = data_4_model.smiles.apply(
    #     lambda x: np.array(fe.fingerprint_features(x)))
    #
    # list_feature = []
    # for i in range(len(data_4_model.smiles_features)):
    #     list_feature.append(data_4_model.smiles_features[i])
    #
    # new_df = pd.DataFrame(list(map(np.ravel, list_feature)))
    # new_df.columns = new_df.columns.map(lambda x: 'fe_' + str(x))
    # print(new_df)
    # exit()
    X_all = pd.concat([data_4_model, new_df], axis=1)
    X_all.drop(columns=['mol_id', 'smiles', 'smiles_features'], inplace=True)
    print(X_all)
    exit()
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
X_train, X_test, X_valid, y_train, y_test, y_valid = load_extract_transform(data)



# Convert all of our data into torch tensors, the required datatype for Pytorch model
train_inputs = torch.tensor(X_train)
train_labels = torch.tensor(y_train)

validation_inputs = torch.tensor(X_valid)
validation_labels = torch.tensor(y_valid)

test_inputs = torch.tensor(X_test)
test_labels = torch.tensor(y_test)