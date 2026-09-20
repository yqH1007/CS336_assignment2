import torch
import torch.nn as nn
import torch.distributed as dist
from torch.optim import Optimizer

class ShardedOptimizer(Optimizer):
    def __init__(self,params,optimizer_cls,**kwargs):
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = kwargs
        self.inner_optimizer = None
        self.param_to_rank = {}
        self.count = 0
        super().__init__(params,kwargs)

    def add_param_group(self,param_group):
        super().add_param_group(param_group)
        collected_param = []
        for p in param_group["params"]:
            ## 按参数顺序分配参数给不同rank
            owner = self.count % self.world_size
            self.param_to_rank[p] = owner
            ## 参数张量的维度不一样，每个参数对应优化器状态也不一样，这种切分不能做到平均分配
            if owner==self.rank:
                collected_param.append(p)     
            self.count +=1      
        if collected_param:
            if self.inner_optimizer is None:
                self.inner_optimizer = self.optimizer_cls(collected_param,**self.optimizer_kwargs)
            else:
                ## 记录内部优化器需要优化的参数列表
                self.inner_optimizer.add_param_group({"params": collected_param})

    def step(self,closure=None,**kwargs):
        result = self.inner_optimizer.step(closure)
        ## 不同rank把自己存储的优化器状态广播给其他rank
        for group in self.param_groups:
            for p in group["params"]:
                dist.broadcast(p.data, src=self.param_to_rank[p])
        return result


