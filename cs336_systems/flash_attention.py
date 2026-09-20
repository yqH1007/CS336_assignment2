import torch
from einops import einsum

def _backward(q,k,v,O,L,is_causal,grad_out):
    D = (O * grad_out).sum(dim=-1)
    scale = q.shape[2]**(-0.5)
    S = einsum(q,k,"... q d,... k d -> ... q k")*scale
    if is_causal:
        mask = torch.arange(q.shape[1],device=q.device)[:,None] >= torch.arange(k.shape[1],device=k.device)[None,:]
        S = S.masked_fill(~mask,-1e6)
    P = torch.exp(S - L.unsqueeze(-1)) ## (exp S / exp max ) / sum(exp qk)
    P = P.to(grad_out.dtype)
    dV = einsum(P,grad_out,"... q k,... q d -> ... k d") ## O = P @ V, dV = P.t @ dO
    dP = einsum(grad_out,v,"... q d,... k d -> ... q k") ## dP = dO @ V.T
    dS = P * (dP - D.unsqueeze(-1))
    dQ = einsum(dS,k,"... q k,... k d -> ... q d")*scale ## S = Q @ K.T，dQ = dS @ K
    dK = einsum(dS,q,"... q k,... q d -> ... k d")*scale ## S = Q @ K.T, dK = dS.T @ Q
    return dQ, dK, dV, None


compiled_backward = torch.compile(_backward,fullgraph=True,dynamic=None,backend="inductor",mode=None)
    
class FlashAttentionPytorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,is_causal=True):
        batch,n_queries,d = q.shape
        n_keys = k.shape[1]
        B_q = 16
        B_k = 16
        T_q = n_queries // B_q
        T_k = n_keys // B_k
        scale = d**(-0.5)
        O = torch.empty_like(q)
        L = torch.empty(batch,n_queries,device=q.device,dtype=q.dtype)
        for i in range(T_q):
            Q_i = q[:,B_q*i:B_q*(i+1),:]
            O_i = torch.zeros_like(Q_i)
            l_i = torch.zeros(batch,B_q,device=q.device,dtype=q.dtype)
            m_i = torch.full((batch,B_q),fill_value=float("-inf"),device=q.device,dtype=q.dtype)
            for j in range(i+1 if is_causal else T_k):
                K_j = k[:,B_k*j:B_k*(j+1),:]
                V_j = v[:,B_k*j:B_k*(j+1),:]
                S_ij = einsum(Q_i,K_j,"... q d, ... k d -> ... q k")*scale ## attention(q,k)
                if is_causal and j==i:
                    mask = torch.arange(B_q,device=Q_i.device)[:,None] >= torch.arange(B_k,device=K_j.device)[None,:]
                    S_ij = S_ij.masked_fill(~mask,-1e6)
                m_new = torch.maximum(m_i,torch.amax(S_ij,dim=-1)) ## 求出最大的相似度权重
                P_i = torch.exp(S_ij - m_new.unsqueeze(-1)) ## exp 权重矩阵S - 当前权重最大值，缩放
                correction = torch.exp(m_i - m_new)## 前后 权重最大值的差值，delta m
                l_i = correction * l_i + P_i.sum(dim=-1) ##softmax 求和 ：最大值发生变化，缩放原来的分母，同时加上新的权重sum_j(exp(qi_kj))，softmax的分母部分
                O_i = O_i * correction.unsqueeze(-1) + einsum(P_i,V_j,"... q k, ... k d -> ... q d")## softmax的分子部分，包括V
                ##先用权重max的差值调整原先的O，然后加上新计算的加权求和的值
                m_i = m_new
            O_i = O_i / l_i.unsqueeze(-1) ## softmax 分数
            L_i = m_i + torch.log(l_i)## 记录一个块算出的softmax sum
            O[:,B_q*i:B_q*(i+1),:] = O_i ## (batch,Q,V)
            L[:,B_q*i:B_q*(i+1)] = L_i ## (batch,Q)
        ctx.save_for_backward(q,k,v,O,L)
        ctx.is_causal = is_causal
        return O

    @staticmethod
    def backward(ctx, grad_out):
        q,k,v,O,L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ,dK,dV,d_is_causal = compiled_backward(q,k,v,O,L,is_causal,grad_out)
        return dQ, dK, dV, d_is_causal

if __name__ == "__main__":
    print(" I'm the handsome boy! ")

