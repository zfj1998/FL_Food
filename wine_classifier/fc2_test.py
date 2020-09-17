import sklearn.datasets
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


# X,y = sklearn.datasets.make_moons(200,noise=0.2)
# X = torch.from_numpy(X).type(torch.FloatTensor)
# y = torch.from_numpy(y).type(torch.LongTensor)

redwine_data = pd.read_csv('winequality-red.csv',sep=';')
Redwine_datas = redwine_datas.values

x = Redwine_datas[:,:11]
y = Redwine_datas[:,11]
y[y<6]=0
y[y>5]=1

x_train,x_test,y_train,y_test = train_test_split(true_redwine_datas,y,test_size=0.3)
standard_datas = StandardScaler()
standard_datas.fit(x)
true_redwine_datas=standard_datas.transform(x)

class MyClassifier(nn.Module):
    def __init__(self):
        super(MyClassifier,self).__init__()
        self.fc1 = nn.Linear(11,6)
        self.fc2 = nn.Linear(6,4)
        self.fc3 = nn.Linear(4,2)
        
    def forward(self,x):
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        x = torch.tanh(x)
        x = self.fc3(x)
        return x
             
    def predict(self,x):
        pred = F.softmax(self.forward(x), dim=0)
        ans = []
        for t in pred:
            if t[0]>t[1]:
                ans.append(0)
            else:
                ans.append(1)
        return torch.tensor(ans)

model = MyClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 1000
losses = []


for i in range(epochs):
    y_pred = model.forward(X)
    loss = criterion(y_pred,y)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

ipdb.set_trace()
print(accuracy_score(model.predict(X),y))