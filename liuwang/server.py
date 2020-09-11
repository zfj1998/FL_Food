
import torch
import syft as sy
from syft.workers.websocket_server import WebsocketServerWorker

hook = sy.TorchHook(torch)
local_worker = WebsocketServerWorker(
                            # host="localhost",
                            host="192.168.2.101", # the host of server machine
                            hook=hook,
                            id='liuwang',
                            port=8182)
hook = sy.TorchHook(torch, local_worker=local_worker)

# data in server
x = torch.tensor([[0,0],[0,1],[1,0],[1,1.]], requires_grad=True).tag("toy", "features")
y = torch.tensor([[0],[0],[1],[1.]], requires_grad=True).tag("toy", "labels")
x_ptr = x.send(local_worker)
y_ptr = y.send(local_worker)
print('>>> local_worker:', local_worker)
print('>>> local_worker.list_objects():', local_worker.list_objects())
print('>>> local_worker.list_tensors():', local_worker.list_tensors())

local_worker.start()  # Might need to interrupt with `CTRL-C` or some other means

print('>>> local_worker.list_objects()', local_worker.list_objects())
print('>>> local_worker.objects_count()', local_worker.objects_count())
print('>>> local_worker.list_tensors():', local_worker.list_tensors())
print('>>> local_worker.host', local_worker.host)
print('>>> local_worker.port', local_worker.port)