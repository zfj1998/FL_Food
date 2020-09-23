import numpy as np
import torch
from torch import nn
from torch.autograd import Variable

torch.manual_seed(42)
class SVR(nn.Module):
    def __init__(self):
        super(SVR,self).__init__()
        self.linearModel=nn.Linear(100,1)
        
    def forward(self,x):
        x = self.linearModel(x)
        return x
    
model=SVR()
def hingeLoss(outputVal,dataOutput,model):
    loss1=torch.sum(torch.clamp(1 - torch.matmul(outputVal.t(),dataOutput),min=0))
    loss2=torch.sum(model.linearModel.weight ** 2)  # l2 penalty
    totalLoss=loss1+loss2
    return(totalLoss)

optimizer=torch.optim.SGD(model.parameters(),lr=0.001)

X=np.random.rand(1000,100).astype(np.float32)
Y=np.random.randint(2,size=(1000)).reshape(1000,1).astype(np.float32)

for epoch in range(10000):
    inputVal=Variable(torch.from_numpy(X))
    outputVal=Variable(torch.from_numpy(Y))
    optimizer.zero_grad()
    modelOutput = model(inputVal)
    totalLoss=hingeLoss(outputVal,modelOutput,model)
    totalLoss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, 100, totalLoss))