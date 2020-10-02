import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ipdb
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler    
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


df = pd.read_csv("../data/winequality-white.csv",sep=';')

df['quality'] = df['quality'].astype('category')
encode_map = {
    3.0: 0,
    4.0: 0,
    5.0: 0,
    6.0: 1,
    7.0: 1,
    8.0: 1,
    9.0: 1
}

df['quality'].replace(encode_map, inplace=True)

X = df.iloc[:, :11]
y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=69)
y_train = y_train.values
y_test = y_test.values


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def shaping_input(*datas):
    result = []
    for data in datas:
        row = data.shape[0]
        col = data.shape[1]
        result.append(data.reshape(row, 1, col))
    return result

X_train, X_test = shaping_input(X_train, X_test)

EPOCHS = 5000
BATCH_SIZE = 256
LEARNING_RATE = 1e-4


## train data
class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

train_data = trainData(torch.FloatTensor(X_train), torch.LongTensor(y_train))
test_data = trainData(torch.FloatTensor(X_test), torch.LongTensor(y_test))
train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels = 1,out_channels = 16,kernel_size = 2)
        self.test_conv2 = nn.Conv1d(16, 32, 2)
        self.test_conv3 = nn.Conv1d(32, 64, 1)
        self.batch1 = nn.BatchNorm1d(16*10)
        self.batch2 = nn.BatchNorm1d(32*4)
        self.batch3 = nn.BatchNorm1d(64*2)
        self.fc1 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, encoder_outputs):
        cnn_out = self.conv1(encoder_outputs)
        cnn_out = cnn_out.reshape(len(cnn_out), 16*10)
        cnn_out = self.batch1(cnn_out)
        cnn_out = cnn_out.reshape(len(cnn_out), 16, 10)
        cnn_out = F.max_pool1d(F.relu(cnn_out), 2)
        cnn_out = self.dropout(cnn_out)

        cnn_out = self.test_conv2(cnn_out)
        cnn_out = cnn_out.reshape(len(cnn_out), 32*4)
        cnn_out = self.batch2(cnn_out)
        cnn_out = cnn_out.reshape(len(cnn_out), 32, 4)
        cnn_out = F.max_pool1d(F.relu(cnn_out), 2)
        cnn_out = self.dropout(cnn_out)

        cnn_out = self.test_conv3(cnn_out)
        cnn_out = cnn_out.reshape(len(cnn_out), 64*2)
        cnn_out = self.batch3(cnn_out)
        cnn_out = cnn_out.reshape(len(cnn_out), 64, 2)
        cnn_out = F.max_pool1d(F.relu(cnn_out), 2)
        cnn_out = self.dropout(cnn_out)

        cnn_out = cnn_out.squeeze()
        output = self.fc1(cnn_out)
        return output

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CNN()
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

def binary_acc(y_pred, y_test):
    y_pred_tag = torch.argmax(y_pred, dim=1)
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum/y_test.shape[0]
    acc = torch.round(acc * 100)
    return acc

for e in range(1, EPOCHS+1):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        
        y_pred = model(X_batch)
        loss = criterion(y_pred, y_batch)
        acc = binary_acc(y_pred, y_batch)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    if e % 50 == 0:
        print(f'Epoch {e+0:04}: | Loss: {epoch_loss/len(train_loader):.5f} | Acc: {epoch_acc/len(train_loader):.3f}')

    if e % 100 == 0:
        model.eval()
        epoch_acc = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_test_pred = model(X_batch)
                acc = binary_acc(y_test_pred, y_batch)
                epoch_acc += acc.item()
        print(f'Test Acc: {epoch_acc/len(test_loader):.3f}')

# without batchnorm
# Epoch 1000: | Loss: 0.48713 | Acc: 76.923
# Test Acc: 74.143

#with batchnorm
# Epoch 0550: | Loss: 0.26266 | Acc: 89.308
# Epoch 0600: | Loss: 0.25085 | Acc: 89.692
# Test Acc: 79.000
# Epoch 4950: | Loss: 0.04238 | Acc: 99.077
# Epoch 5000: | Loss: 0.04482 | Acc: 98.692
# Test Acc: 78.714

# with batchnorm and 1 dropout
# Epoch 3550: | Loss: 0.13402 | Acc: 95.000
# Epoch 3600: | Loss: 0.14369 | Acc: 94.462
# Test Acc: 78.571
