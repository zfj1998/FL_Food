import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
import pandas as pd

epi_datas = pd.read_csv('./epi_r_filtered_5.csv')
epi_datas = epi_datas.values
x = epi_datas[:,1:]
y = epi_datas[:,0]
y[y==0.0]=0
y[y==1.25]=0
y[y==1.875]=0
y[y==2.5]=0
y[y==3.125]=0
y[y==3.75]=0
y[y==4.375]=1
y[y==5.0]=1
standar_data = StandardScaler()
standar_data.fit(x)
standar_datax = standar_data.transform(x)
x_train,x_test,y_train,y_test = train_test_split(standar_datax,y,test_size=0.3)
x_train = torch.from_numpy(x_train).type(torch.FloatTensor)
x_test = torch.from_numpy(x_test).type(torch.FloatTensor)
y_train = torch.from_numpy(y_train).type(torch.LongTensor)
y_test = torch.from_numpy(y_test).type(torch.LongTensor)

class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier,self).__init__()
        self.fc1 = nn.Linear(4 ,8)
        self.fc2 = nn.Linear(8, 2)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
             
    def predict(self,x):
        pred = self.forward(x)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

model = MyClassifier()
criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
epochs = 10000
losses = []
train_accuracy = []
test_accuracy = []

for i in range(epochs):
    x_pred = model.forward(x_train)
    loss = criterion(x_pred,y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i%50 == 0:
        loss_v = loss.item()
        tr_acc = accuracy_score(model.predict(x_train), y_train)
        te_acc = accuracy_score(model.predict(x_test), y_test)
        losses.append(loss_v)
        train_accuracy.append(tr_acc)
        test_accuracy.append(te_acc)
        print('loss:{} train acc:{}  test acc:{}'.format(loss_v, tr_acc, te_acc))

min_loss = min(losses)
max_train = max(train_accuracy)
max_test = max(test_accuracy)

print('minloss:{}  maxtrain:{}  maxtest:{}'.format(min_loss, max_train, max_test))
