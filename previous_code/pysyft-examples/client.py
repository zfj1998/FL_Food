
import torch
from torch import optim
import syft
from syft.grid.public_grid import PublicGridNetwork
from syft.workers.websocket_client import WebsocketClientWorker
hook = syft.TorchHook(torch)

def train(model, datasets, ITER=20)->torch.nn.Module:
    """
    :param model: the torch model
    :param datasets: the datasets pointers about server workers
            with the format as [(data_ptr, target_ptr), (data_ptr, target_ptr), ...]
    :param ITER: the number of iteration
    :return:
    """
    model_c = model.copy()
    # Training Logic
    for iter in range(ITER):
        for data, target in datasets:
            # 1) send model to correct worker
            model_c = model_c.send(data.location)
            # 2) Call the optimizer for the worker using get_optim
            opt = optim.SGD(params=model_c.parameters(),lr=0.1)
            # 3) erase previous gradients (if they exist)
            opt.zero_grad()
            # 4) make a prediction
            pred = model_c(data)
            # 5) calculate how much we missed
            loss = ((pred - target)**2).sum()
            # 6) figure out which weights caused us to miss
            loss.backward()
            # 7) change those weights
            opt.step()
            # 8) get model (with gradients)
            model_c = model_c.get()
            # 9) print our progress
            print(data.location.id, loss.get())
    return model_c

if __name__ == '__main__':
    # create a client workers mapping to the server workers in remote machines
    # remote_client_1 = PublicGridNetwork(hook=hook, gateway_url='http://localhost:8182')
    # remote_client_2 = PublicGridNetwork(hook=hook, gateway_url='http://localhost:8183')
    remote_client_1 = WebsocketClientWorker(
        host='localhost',
        # host = '192.168.0.102', # the host of remote machine, the same as the Server host
        hook=hook,
        id='server1',
        port=8182)
    remote_client_2 = WebsocketClientWorker(
        host='localhost',
        # host = '192.168.0.102', # the host of remote machine, the same as the Server host
        hook=hook,
        id='server2',
        port=8183)
    remote_clients_list = [remote_client_1, remote_client_2]
    print('>>> remote_client_1', remote_client_1)
    print('>>> remote_client_2', remote_client_2)

    # get the data pointers which point to the real data in remote machines for training model
    datasets = []
    for remote_client in remote_clients_list:
        data = remote_client.search(["toy", "data"])[0]
        target = remote_client.search(["toy", "target"])[0]
        print('>>>data: ', data)
        print('>>>target: ', target)
        datasets.append((data, target))
    # exit(0)
    # define torch model
    model = torch.nn.Linear(2, 1)
    print('>>> untrained model: ', model.state_dict())
    # train model
    trained_model = train(model, datasets, ITER=10)
    print('>>> trained model: ', trained_model.state_dict())
