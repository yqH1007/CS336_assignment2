import torch
import torch.distributed as dist
import torch.nn as nn


class DDP(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module = module
        for p in self.module.parameters():
            dist.broadcast(p.data,src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data,src=0)


    def forward(self,*inputs,**kwargs):
        return self.module(*inputs,**kwargs)

    def finish_gradient_synchronization(self):
        world_size = dist.get_world_size()
        for p in self.module.parameters():
            if p.grad is not None:
                dist.all_reduce(p.grad,op=dist.ReduceOp.SUM) ## 将不同GPU上的梯度累加在一起
                p.grad /= world_size


class DDPOverlap(nn.Module):
    def __init__(self,module):
        super().__init__()
        self.module = module
        self.world_size = dist.get_world_size()
        for p in self.module.parameters():
            dist.broadcast(p.data,src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data,src=0)
        self.handles = []
        def my_hook(param):
            param.grad /= self.world_size
            handle = dist.all_reduce(param.grad,op=dist.ReduceOp.SUM,async_op=True)
            self.handles.append(handle)
        for p in self.module.parameters():
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(my_hook)



    def forward(self,*inputs,**kwargs):
        return self.module(*inputs,**kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []
