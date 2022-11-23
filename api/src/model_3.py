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
import pandas_profiling as pp

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

# Extract all the targets in a dataframe
def target_data(data):
    df = pd.read_csv(data)
    df_target = df[['P2',  'P3',  'P3',  'P4',  'P5',  'P6',  'P7',  'P8',  'P9']]

    return df_target

df_target = target_data(data)
# print(df_target)

#
# exit()

def load_extract_transform_smile_string(data):
    data_4_model_multi = pd.read_csv(data)
    # extract feature
    data_4_model_multi['smiles_string_features'] = data_4_model_multi.smiles.apply(
        lambda x: np.array(tokenizer.encode(x)))

    # Data Exploration
    # report_data_4_model_multi_model_3 = pp.ProfileReport(data_4_model_multi)
    # report_data_4_model_multi_model_3.to_file('profile_report_data_4_model_multi_model_3.html')

    list_simile_feature = []
    for i in range(len(data_4_model_multi.smiles_string_features)):
        list_simile_feature.append(data_4_model_multi.smiles_string_features[i])

    new_df = pd.DataFrame(list(map(np.ravel, list_simile_feature)))
    new_df.columns = new_df.columns.map(lambda x: 'ssf_' + str(x))

    X_all = pd.concat([data_4_model_multi, new_df], axis=1)

    # print(X_all)
    # exit()

    X_all.drop(columns=['mol_id', 'smiles', 'smiles_string_features'], inplace=True)
    X_all = X_all.fillna(0)
    # print(X_all)
    # print(len(X_all.columns))
    # exit()

    # Normalizing data
    scaler = MaxAbsScaler()
    scaler.fit(X_all)
    scaled = scaler.transform(X_all)
    scaled_df = pd.DataFrame(scaled, columns=X_all.columns)
    # print(scaled_df)
    X = scaled_df.drop('P3', axis=1).values
    y = scaled_df['P3'].values
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
    # print(new_df)

    return new_df

new_df = extractfeatures(pd.read_csv(data))

# print(new_df)
#
# exit()
def load_extract_transform(data):
    data_4_model_multi = pd.read_csv(data)
    # extract feature
    new_df = extractfeatures(data_4_model_multi)
    # data_4_model_multi['smiles_features'] = data_4_model_multi.smiles.apply(
    #     lambda x: np.array(fe.fingerprint_features(x)))
    #
    # list_feature = []
    # for i in range(len(data_4_model_multi.smiles_features)):
    #     list_feature.append(data_4_model_multi.smiles_features[i])
    #
    # new_df = pd.DataFrame(list(map(np.ravel, list_feature)))
    # new_df.columns = new_df.columns.map(lambda x: 'fe_' + str(x))
    # print(new_df)
    # exit()
    X_all = pd.concat([data_4_model_multi, new_df], axis=1)
    X_all.drop(columns=['mol_id', 'smiles', 'smiles_features'], inplace=True)
    # print(X_all)
    # exit()
    X = X_all.drop('P3', axis=1).values
    y = X_all['P3'].values
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

# All the data are already Tensor data
X_train, X_test, X_valid, y_train, y_test, y_valid = load_extract_transform(data)


# # Convert all of our data into torch tensors, the required datatype for Pytorch model
# train_inputs = torch.tensor(X_train)
# train_labels = torch.tensor(y_train)
#
# validation_inputs = torch.tensor(X_valid)
# validation_labels = torch.tensor(y_valid)
#
# test_inputs = torch.tensor(X_test)
# test_labels = torch.tensor(y_test)


class SingleTaskModel(nn.Module):
    """Single task model custom class"""

    def __init__(self, model_1, out_features):  # add conf file
        super(SingleTaskModel, self).__init__()

        for param in model_1.parameters():
            param.requires_grad = False

        in_features = model_1.fc.in_features
        self.model_1 = model_1
        self.model_1.fc = nn.Linear(in_features, 2048)
        self.fc = nn.Linear(in_features=2048, out_features=out_features)

    def forward(self, x):
        return self.fc(self.model_1(x))


############################# Multitask model based on our MODEL 1 : The extension of our #############################


print('[INFO : Multitask Model start bellow!!!]')


############################ Multitask model based on our MODEL 1 : The extension of our ##############################

class MultiTaskModel(nn.Module):
    """Custom multi task model class"""

    def __init__(self, model_1):  # add conf file
        super(MultiTaskModel, self).__init__()

        # making sure we don't update the gradiente for the resnet model
        for param in model_1.parameters():
            param.requires_grad = False

        num_ftrs = model_1.fc.in_features
        self.model_1 = model_1
        self.model_1.fc = nn.Linear(num_ftrs, 2048)

        # self.P1 = nn.Linear(in_features=2048, out_features=1)
        self.P3 = nn.Linear(in_features=2048, out_features=5)
        self.P2 = nn.Linear(in_features=2048, out_features=2)

    def forward(self, x):
        # P1 = self.P1(self.model_1(x))
        P2 = self.P2(self.model_1(x))
        P3 = self.P3(self.model_1(x))

        return P2, P3


# We need a spacial loss function for our multitask model
def multitask_loss(outputs, target):
  cross_entropy = nn.CrossEntropyLoss()

  target_P2, target_P3 = target[:,0], target[:, 1]
  output_P2, output_P3 = outputs[0], outputs[1]

  loss_P3 = cross_entropy(output_P3, target_P3.long())
  loss_P2 = cross_entropy(output_P2, target_P2.long())
  # Due to my experience, after different experiment, we will have different factors associate to loss_P2 and to loss_P3
  # Let's say they are 0.2 and 0.5 respectively
  return loss_P2/(0.2) + loss_P3/(0.5)


print('[INFO : WORK ON PROGRESS!!!]')

# Custom function to calculate the accurance for the multi task model
def multitask_accuracy(outputs, labels):
    pass


# Help to build a prediction text for a multitask model
def multi_prediction_text_fn(outputs, idx):
    pass


