import torch.nn as nn
import torch.distributed as dist
import torch
from cs336_basics.model import Linear,Embedding

class FSDP(nn.Module):
    def __init__(self,module,compute_dtype=None):
        super().__init__()
        self.compute_dtype = compute_dtype
        self.module = module
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.handles = []
        self.sharded_modules = []
        self.param_to_module = {}
        def _gather(module):
            shard = module._fsdp_sharded
            if self.compute_dtype is not None:
                dt = self.compute_dtype
            else:
                dt = torch.float32
            shard_cast = shard.to(dt)
            full = torch.empty(module._fsdp_full_shape,dtype=dt,device=shard.device)
            dist.all_gather_into_tensor(full,shard_cast)
            module.weight.data = full
        def forward_pre(module,args):
            _gather(module=module)

        def forward_post(module,args,output):
            module.weight.data = module._fsdp_sharded
            module.weight.grad = None

        def backward_pre(module,args):
            module.weight.grad = None
            _gather(module=module)

        def grad_hook(param):
            m = self.param_to_module[param]
            g = param.grad
            g = g.to(torch.float32) / self.world_size
            param.data = m._fsdp_sharded
            grad_sharded = torch.empty_like(m._fsdp_sharded)
            dist.reduce_scatter_tensor(grad_sharded,g.contiguous(),op=dist.ReduceOp.SUM) ## 
            param.grad = grad_sharded

        def grad_hook_replicated(param):
            param.grad /= self.world_size
            dist.all_reduce(param.grad,op=dist.ReduceOp.SUM)
            
        for p in self.module.parameters():
            dist.broadcast(p.data,src=0)
        for b in self.module.buffers():
            dist.broadcast(b.data,src=0)
        for m in self.module.modules():
            if isinstance(m,(Linear,Embedding)):
                self.sharded_modules.append(m)
                m._fsdp_full_shape = m.weight.shape
                D0 = m._fsdp_full_shape[0]
                sharded = D0 // self.world_size
                start = self.rank * sharded
                m.weight.data = m.weight.data[start:start+sharded].clone()
                m._fsdp_sharded = m.weight.data
                self.param_to_module[m.weight] = m
                m.register_forward_pre_hook(forward_pre)
                m.register_forward_hook(forward_post)
                m.register_full_backward_pre_hook(backward_pre)
                m.weight.register_post_accumulate_grad_hook(grad_hook)
        for param in self.module.parameters():
            if param not in self.param_to_module.keys():
                param.register_post_accumulate_grad_hook(grad_hook_replicated)

    def forward(self,*input,**kwargs):
        return self.module(*input,**kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles = []


            
