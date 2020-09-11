
import torch
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker
import sys

try:
    host = sys.argv[1]
    id = sys.argv[2]
    port = sys.argv[3]
    print(host, id, port)
except Exception as e:
    host, id, port = None, None, None
    print(str(e))
    print('run the server by: "python server.py host id port"')
    print('for example: "python server.py localhost server1 8182"')
    exit(-1)

hook = sy.TorchHook(torch)
server_worker = WebsocketServerWorker(host=host,  # host="192.168.2.101", # the host of server machine
                                      hook=hook, id=id, port=port)
# hook = sy.TorchHook(torch, local_worker=server_worker)

# data in server
x = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True).tag("toy", "data")
y = torch.tensor([[0],[0],[1],[1.]], requires_grad=True).tag("toy", "target")
# x.private, x.private = True, True

x_ptr = x.send(server_worker)
y_ptr = y.send(server_worker)
print(x_ptr, y_ptr)

# x = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=False)
# y = torch.tensor([[0],[0],[1],[1.]], requires_grad=False)
# server_worker.add_dataset(sy.BaseDataset(data=x, targets=y), key="vectors")

print('>>> server_worker:', server_worker)
print('>>> server_worker.list_objects():', server_worker.list_objects())
print('>>> server_worker.list_tensors():', server_worker.list_tensors())

server_worker.start()  # Might need to interrupt with `CTRL-C` or some other means

print('>>> server_worker.list_objects()', server_worker.list_objects())
print('>>> server_worker.objects_count()', server_worker.objects_count())
print('>>> server_worker.list_tensors():', server_worker.list_tensors())
print('>>> server_worker.host', server_worker.host)
print('>>> server_worker.port', server_worker.port)