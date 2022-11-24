# data_science_technical_test
# !!! CAUTION !!! : Because this project is a Test, I put a lot of details. Otherwise, I'll keep it simple like in the real life by giving Description, Installation and the name of Functionalities scripts and what each one do. !!! CAUTION !!!
# Also, the evaluation was based on the documentation. So, I put maximum details to explain my job
[comment]: <> (**Table of content**)

[comment]: <> (- [Documentation]&#40;#documentation&#41;)

[comment]: <> (- [Documentation about Application]&#40;#application-structure&#41;)

[comment]: <> (  - [Objectif :]&#40;#objectif-of-this-application-&#41;)

[comment]: <> (  - [Application Structure]&#40;#application-structure&#41;)

[comment]: <> (  - [Project Structure]&#40;#project-structure&#41;)

[comment]: <> (- [Deep Learning Model]&#40;#deep-learning-model&#41;)

[comment]: <> (  - [Data Exploration]&#40;#data-exploration&#41;)

[comment]: <> (  - [Model_1 ANN]&#40;#Model_1 ANN with extracted features of a molecule &#41;)

[comment]: <> (  - [Model_2 ANN]&#40;#Model_2 ANN with smile string character as input&#41;)

[comment]: <> (  - [Model_3 Multitask]&#40;#Model_3 Multitask-Extension of Model1&#41;)

[comment]: <> (- [Dockerization]&#40;#dockerization&#41;)

[comment]: <> (- [Flask API]&#40;#flask-api&#41;)

[comment]: <> (- [Setup]&#40;#setup-2&#41;)

# Documentation
## Context 
The prediction of a drug molecule properties plays an important role in the drug design process. The molecule properties are the causeof failure for 60% of all drugs in the clinical phases. A multi parameters optimization using machine learning methods can be used tochoose an optimized molecule to be subjected to more extensive studies and to avoid any clinical phase failure.

So, the objective of this exercise is to develop a deep learning model to predict one or more basic properties of a molecule.

This project predict basic molecule's properties, from its fingerprint features, using deep learning

## The functionalities :

- Setup python environment using `miniconda`
- Create git repository
- Load a dataset and perform feature extraction
- Split the dataset
- Train a classification Deep Learning model : ANN
- Save the pretrained model
- Predict the molecule properties using pretrained model
- Evaluate the model
- Test the model
- Package the app inside a docker image
- Access the `predict` method via a `Flask` API
- Make a setup.py

# Details of the functionalities
## App Structure

This project is organized as follow

```
.
├── api
│   ├── cli.py
│   ├── data
│   │   ├── dataset_multi.csv
│   │   ├── dataset_single.csv
│   │   └── dataset_single_folds.csv
│   ├── __init_.py
│   ├── media
│   │   ├── P1.png
│   │   ├── test_api.png
│   │   ├── test_flask.png
│   │   └── test_terminal.png
│   ├── models
│   │   ├── entire_model_1.pt
│   │   ├── entire_model_2.pt
│   │   ├── state_dict_model_1.pt
│   │   └── state_dict_model_2.pt
│   └── src
│       ├── templates
│       │   ├── index.html
│       │   ├── profile_report_data_4_model_model_1.html
│       │   └── profile_report_data_4_model_multi_model_3.py
│       ├── api_flask.py
│       ├── config_file.py 
│       ├── feature_extractor.py
│       ├── main.py
│       ├── model_1.py
│       ├── model_2.py
│       └── model_3.py
├── test
│   ├── __init__.py
│   └── test.py
├── .gitignore
├── Cloud_Environment_Questions_Answers.pdf
├── Dockerfile
├── LICENCE
├── README.md
├──requirements.txt 
├──run_flask_api.sh 
├── servier.yaml
└── setup.py
```

## Installation : Setup environment
I use the save as been given
```python
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda
export PATH=~/miniconda/bin:$PATH
conda update -n base conda
conda create -y --name servier python=3.6
conda activate servier
conda install -c conda-forge rdkit
```
# Git repository
I create a repository for the versionning of my App

## Deep learning model

First of all, we nead to load and transform and split data in to train, validation and test set

### Laod, extract and transform data
For Model 1 
```python
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

```
 For the model 2
```python

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data=config.path_single
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

    # Data Exploration
    # report_X_all_model_2 = pp.ProfileReport(X_all)
    # report_X_all_model_2.to_file('templates/profile_report_X_all_model_2.html')


    X_all.drop(columns=['mol_id', 'smiles', 'smiles_string_features'], inplace=True)
    X_all = X_all.fillna(0)
    
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

```

### Data Exploration

`dataset_single.csv`

```python
    P1	mol_id	    smiles
0	1	CID2999678	Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C
1	0	CID2999679	Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1
2	1	CID2999672	COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O...
3	0	CID5390002	O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1cc...
4	1	CID2999670	NC(=O)NC(Cc1ccccc1)C(=O)O
```

1. Overview of our data
   ![Overview](api/media/img_overview.png)
2. Variable of our data
   ![Variables](api/media/img_variables.png)
3. Check missing values of our data
   ![MissingValue](api/media/img_missing_values.png)
4. Samples of our data
   ![Samples](api/media/img_sample.png)

`dataset_multi.csv`
```python
      P2  P1  P3  P4  P5  P6  P7  P8  P9      mol_id                                             smiles                             smiles_string_features
0      1   1   1   1   1   1   1   0   1  CID2999678    Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C  [101, 10507, 2487, 9468, 9468, 1006, 1050, 247...
1      0   0   1   1   0   0   0   1   1  CID2999679                Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1  [101, 27166, 2487, 9468, 12273, 2487, 11020, 2...
2      1   1   0   1   1   1   1   1   1  CID2999672  COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O...  [101, 2522, 2278, 2487, 9468, 2475, 2278, 1006...
3      1   0   0   1   1   0   1   1   1  CID5390002  O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1cc...  [101, 1051, 1027, 27723, 1013, 1039, 1006, 102...
4      0   1   1   1   0   1   0   0   1  CID2999670                          NC(=O)NC(Cc1ccccc1)C(=O)O  [101, 13316, 1006, 1027, 1051, 1007, 13316, 10...
```

A you can see below, we have nan values because all the molecules have not the same length
So we need to fill missing value by 0. And after that, because of a great value, we need to normalize our data set.
```python
     P2  P1  P3  P4  P5  P6  P7  P8  P9      mol_id                                             smiles                             smiles_string_features  ssf_0  ssf_1  ssf_2  ssf_3  ssf_4  ssf_5  ssf_6  ssf_7  ssf_8  ssf_9  ssf_10   ssf_11   ssf_12   ssf_13  ssf_14   ssf_15   ssf_16   ssf_17  ssf_18  ssf_19  ssf_20   ssf_21   ssf_22   ssf_23   ssf_24   ssf_25   ssf_26  ssf_27   ssf_28   ssf_29  ssf_30   ssf_31   ssf_32   ssf_33  ssf_34  ssf_35   ssf_36  ssf_37  ssf_38  ssf_39  ssf_40  ssf_41  ssf_42   ssf_43  ssf_44  ssf_45  ssf_46  ssf_47  ssf_48  ssf_49  ssf_50  ssf_51  ssf_52  ssf_53  ssf_54  ssf_55  ssf_56  ssf_57  ssf_58  ssf_59  ssf_60  ssf_61  ssf_62  ssf_63  ssf_64
0      1   1   1   1   1   1   1   0   1  CID2999678    Cc1cccc(N2CCN(C(=O)C34CC5CC(CC(C5)C3)C4)CC2)c1C  [101, 10507, 2487, 9468, 9468, 1006, 1050, 247...    101  10507   2487   9468   9468   1006   1050   2475   9468   2078    1006   1039.0   1006.0   1027.0  1051.0   1007.0   1039.0  22022.0  9468.0  2629.0  9468.0   1006.0  10507.0   1006.0   1039.0   2629.0   1007.0  1039.0   2509.0   1007.0  1039.0   2549.0   1007.0  10507.0  2475.0  1007.0  27723.0  2278.0   102.0     NaN     NaN     NaN     NaN      NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
1      0   0   1   1   0   0   0   1   1  CID2999679                Cn1ccnc1SCC(=O)Nc1ccc(Oc2ccccc2)cc1  [101, 27166, 2487, 9468, 12273, 2487, 11020, 2...    101  27166   2487   9468  12273   2487  11020   2278   1006   1027    1051   1007.0  13316.0   2487.0  9468.0   2278.0   1006.0   1051.0  2278.0  2475.0  9468.0   9468.0   2278.0   2475.0   1007.0  10507.0   2487.0   102.0      NaN      NaN     NaN      NaN      NaN      NaN     NaN     NaN      NaN     NaN     NaN     NaN     NaN     NaN     NaN      NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
2      1   1   0   1   1   1   1   1   1  CID2999672  COc1cc2c(cc1NC(=O)CN1C(=O)NC3(CCc4ccccc43)C1=O...  [101, 2522, 2278, 2487, 9468, 2475, 2278, 1006...    101   2522   2278   2487   9468   2475   2278   1006  10507   2487   12273   1006.0   1027.0   1051.0  1007.0  27166.0   2487.0   2278.0  1006.0  1027.0  1051.0   1007.0  13316.0   2509.0   1006.0  10507.0   2278.0  2549.0   9468.0   9468.0  2278.0  23777.0   1007.0  27723.0  1027.0  1051.0   1007.0  1051.0  2278.0  2487.0  9468.0  9468.0  2278.0  12521.0   102.0     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
3      1   0   0   1   1   0   1   1   1  CID5390002  O=C1/C(=C/NC2CCS(=O)(=O)C2)c2ccccc2C(=O)N1c1cc...  [101, 1051, 1027, 27723, 1013, 1039, 1006, 102...    101   1051   1027  27723   1013   1039   1006   1027   1039   1013   13316   2475.0   9468.0   2015.0  1006.0   1027.0   1051.0   1007.0  1006.0  1027.0  1051.0   1007.0  29248.0   1007.0  29248.0   9468.0   9468.0  2278.0   2475.0   2278.0  1006.0   1027.0   1051.0   1007.0  1050.0  2487.0   2278.0  2487.0  9468.0  9468.0  2278.0  2487.0   102.0      NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
4      0   1   1   1   0   1   0   0   1  CID2999670                          NC(=O)NC(Cc1ccccc1)C(=O)O  [101, 13316, 1006, 1027, 1051, 1007, 13316, 10...    101  13316   1006   1027   1051   1007  13316   1006  10507   2487    9468   9468.0   2278.0   2487.0  1007.0   1039.0   1006.0   1027.0  1051.0  1007.0  1051.0    102.0      NaN      NaN      NaN      NaN      NaN     NaN      NaN      NaN     NaN      NaN      NaN      NaN     NaN     NaN      NaN     NaN     NaN     NaN     NaN     NaN     NaN      NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN     NaN
```
For the model 3, because of an extension of model 1, I extrat data differently.
I start to extract target before: 
```python
data=config.path_multi
def target_data(data):
    df = pd.read_csv(data)
    df_target = df[['P2',  'P3',  'P3',  'P4',  'P5',  'P6',  'P7',  'P8',  'P9']]

    return df_target

df_target = target_data(data)
```

After that, we can start to build our models

### MODEL 1 & 2 : ANN

```python

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

```
### Train the model:
```python
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

```

### Evaluate the model
```python
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


```


### Testing the model
```python
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

```


### Save the model
```python
############################# Save Model ####################################

def save_model(model):

    model_scripted = torch.jit.script(model)  # Export to TorchScript
    model_scripted.save('../models/model_scripted_1.pt')  # Save


```

### MODEL 3 : The multitask model base on model 1
I didn't have the time to finish it. But, base on my experience, I propose this architecture below. Anyway, I'll continue this project 


## Package the app inside a docker image
I creat a Dorkerfile file. In Docker, it's very difficult to setup environment like in a computer. The problem is when you create an environment, you can't activate it like in our computer.
So, I use this code below :

#### The single task model
```python
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
```
#### The multitask model based on the single task model like my model 1
```python
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
```

#### We need a spacial loss function for our multitask model
```python

def multitask_loss(outputs, target):
  cross_entropy = nn.CrossEntropyLoss()

  target_P2, target_P3 = target[:,0], target[:, 1]
  output_P2, output_P3 = outputs[0], outputs[1]

  loss_P3 = cross_entropy(output_P3, target_P3.long())
  loss_P2 = cross_entropy(output_P2, target_P2.long())
  # Due to my experience, after different experiment, we will have different factors associate to loss_P2 and to loss_P3
  # Let's say they are 0.2 and 0.5 respectively
  return loss_P2/(0.2) + loss_P3/(0.5)

```

```python

```

```python
FROM continuumio/miniconda3
# Make RUN commands use `bash --login`:
SHELL ["/bin/bash", "--login", "-c"]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN apt-get update

RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda --version

RUN conda update -n base conda

RUN conda create -y --name servier python=3.6

SHELL ["conda", "run", "-n", "servier", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure flask is installed:"

WORKDIR /api

COPY api .

COPY test .

RUN conda install -c conda-forge rdkit

COPY requirements.txt .

RUN conda install --file requirements.txt
```

### Access the `predict` method via a `Flask` API

```python
# print('[INFO: Prediction start here!!!!]')
model_1 = torch.jit.load('../models/model_scripted_1.pt')
model_1.eval()
# print(model_1)

model_2 = torch.jit.load('../models/model_scripted_2.pt')
model_2.eval()
# print(model_2)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/molecule', methods=['POST', 'GET'])
def predict():
    if request.method=='POST':
        molecule = request.form['molecule']
        output = predict_final(molecule)
        return redirect(url_for("models", pred=output))

    else:
        return render_template('prediction.html')


@app.route("/<pred>", methods=['POST', 'GET'])
def models(pred):
    return f"<h1>{pred}</h1>"
```

Additional page was in the template to explore in more details our data.
#### You can also make prediction by click on `prediction` in the navbar in Home

### Make a setup.py

```python
import setuptools


# CAUTION : At this point, if we make a setup, it will not work. But to use functions in each models, move this file in api/src and run this file : python setpu.py
setuptools.setup(
    include_package_data = True,
    name = 'ServierMoleculePropertyPrediction',
    version = '1.0.0',
    licence = 'MIT',
    description = 'Servier Molecule Properties Prediction python module',
    url = 'https://github.com/BasileKoffiDataScientist/data_science_technical_test',
    author = 'Basile Koffi',
    author_email = 'koffibasile@gmail.com',
    # I can parse here the requirement.txt file. But I'm in Apha version. So I keet it simple
    install_requires = ["pandas", "rdkit", "transformers"],
    long_description = "This application predict basic molecule's properties, from its fingerprint features and smile string, using deep learning",
    long_description_content_type = "text/markdown",
    classifiers = [
        "Programming Language  :: Python ::3",
        "Operating System  :: OS Independant",
    ],
    entry_points='''
        http://192.168.1.4:5000/final/data
        ''',
)
```
# Conclusion: 
###I have never worked with this kind of data. But, it was a real experience to work with. Anyway, I will finish this work with Django because Imaster it better. It's over 5 years since I last touched Flask.