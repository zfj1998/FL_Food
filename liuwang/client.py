
import torch
from torch import optim
import syft
from syft.workers.websocket_client import WebsocketClientWorker
hook = syft.TorchHook(torch)
# create a client worker mapping to the server worker in remote machine
remote_client = WebsocketClientWorker(
                            host = '192.168.0.102', # the host of remote machine, the same a the Server host
                            hook=hook,
                            id='liuwang',
                            port=8182)
print('>>> remote_client',  remote_client)

# get the data pointers which point to the real data in remote machine for training model locally
features = remote_client.search(["toy", "features"])
labels = remote_client.search(["toy", "labels"])
print('>>> x:', features)
print('>>> y:', labels)

# a toy model
model = torch.nn.Linear(2, 1)
remote_model = model.copy().send(remote_client)

def train(x, y, N)->torch.nn.Module:
    # Training Logic
    opt = optim.SGD(params=remote_model.parameters(),lr=0.1)
    for iter in range(N):
        # 1) erase previous gradients (if they exist)
        opt.zero_grad()
        # 2) make a prediction
        pred = remote_model(x)
        # 3) calculate how much we missed
        loss = ((pred - y)**2).sum()
        # 4) figure out which weights caused us to miss
        loss.backward()
        # 5) change those weights
        opt.step()
        # 6) print our progress
        print(loss.get())
    return remote_model.get()

trained_model = train(features[0], labels[0], 20)
print(trained_model)
