import torch
from einops import einsum
from cs336_systems.flash_attention import compiled_backward
import triton
import triton.language as tl

@triton.jit
def flash_attention_triton(
    Q_ptr, K_ptr, V_ptr,
    O_ptr, L_ptr,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    N_Q: tl.constexpr, 
    N_K: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    B_Q: tl.constexpr,
    B_K: tl.constexpr,
    B_D: tl.constexpr,
):
    i = tl.program_id(0)
    b = tl.program_id(1)

    ## grid = (n_queries // B_q, batch) 总共有batch*(t_k)个kernel并行处理
    rows = i * B_Q + tl.arange(0, B_Q) ## 索引
    cols = tl.arange(0, B_K)
    dims = tl.arange(0, B_D)

    # Q_i: [B_Q, B_D]
    q_ptrs = (
        Q_ptr
        + b * stride_qb
        + rows[:, None] * stride_qn
        + dims[None, :] * stride_qd
    )
    Q_i = tl.load(
        q_ptrs,
        mask=(rows[:, None] < N_Q) & (dims[None, :] < D),
        other=0.0,
    )

    # FP32 在线 softmax 状态
    O_i = tl.zeros((B_Q, B_D), dtype=tl.float32)
    l_i = tl.zeros((B_Q,), dtype=tl.float32)
    m_i = tl.full(
        (B_Q,), -float("inf"), dtype=tl.float32
    )

    # causal 时，当前 Q 块只需读取截至该块末尾的 K/V
    end_k = N_K
    if IS_CAUSAL:
        end_k = tl.minimum(N_K, (i + 1) * B_Q)

    for start_k in tl.range(0, end_k, B_K):
        keys = start_k + cols
        ## 取K的分块
        k_ptrs = (
            K_ptr
            + b * stride_kb
            + dims[:, None] * stride_kd
            + keys[None, :] * stride_kn
        )
        K_j_T = tl.load(
            k_ptrs,
            mask=(dims[:, None] < D) & (keys[None, :] < N_K),
            other=0.0,
        )

        # V_j: [B_K, B_D]
        v_ptrs = (
            V_ptr
            + b * stride_vb
            + keys[:, None] * stride_vn
            + dims[None, :] * stride_vd
        )
        V_j = tl.load(
            v_ptrs,
            mask=(keys[:, None] < N_K) & (dims[None, :] < D),
            other=0.0,
        )

        # S_ij = Q_i @ K_j.T / sqrt(D)
        # [B_Q, B_D] @ [B_D, B_K] -> [B_Q, B_K]
        S_ij = tl.dot(
            Q_i, K_j_T, input_precision="ieee"
        ) * SCALE

        # 越界 key 和未来位置都不能参与 softmax
        visible = keys[None, :] < N_K
        if IS_CAUSAL:
            visible = visible & (
                rows[:, None] >= keys[None, :]
            )
        S_ij = tl.where(visible, S_ij, -float("inf"))

        # 与你的 PyTorch 递推逐行对应
        m_new = tl.maximum(m_i, tl.max(S_ij, axis=1))
        P_i = tl.exp(S_ij - m_new[:, None])
        correction = tl.exp(m_i - m_new)

        l_i = correction * l_i + tl.sum(P_i, axis=1)

        O_i = O_i * correction[:, None]
        O_i = tl.dot(
            P_i.to(V_j.dtype),
            V_j,
            O_i,
            input_precision="ieee",
        )
        m_i = m_new

    # 最后统一归一化，保存自然对数形式的 LSE
    O_i = O_i / l_i[:, None]
    L_i = m_i + tl.log(l_i)

    # 启动函数分配的 O、L 都是连续张量
    o_ptrs = (
        O_ptr
        + b * N_Q * D
        + rows[:, None] * D
        + dims[None, :]
    )
    tl.store(
        o_ptrs,
        O_i,
        mask=(rows[:, None] < N_Q) & (dims[None, :] < D),
    )
    tl.store(
        L_ptr + b * N_Q + rows,
        L_i,
        mask=rows < N_Q,
    )

def get_tile_sizes(d):
    """Share launch tile sizes with benchmark reporting."""
    return 16, 16, max(16, triton.next_power_of_2(d))


def _forward(q,k,v,is_causal):
    batch, n_queries, d = q.shape
    n_keys = k.shape[1]
    O = torch.empty(
        (batch, n_queries, d),
        device=q.device,
        dtype=q.dtype,
    )
    L = torch.empty(
        (batch, n_queries),
        device=q.device,
        dtype=torch.float32,
    )
    B_q, B_k, B_d = get_tile_sizes(d)
    with torch.cuda.device(q.device):
        flash_attention_triton[(triton.cdiv(n_queries,B_q),batch)](
            q,k,v,O,L,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            N_Q = n_queries,
            N_K = n_keys,
            D = d,
            SCALE = d ** (-0.5),
            IS_CAUSAL = bool(is_causal),
            B_Q = B_q,
            B_K = B_k,
            B_D = B_d,
        )
    return O,L
    
class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,is_causal=True):
        O,L = _forward(q,k,v,is_causal)
        ctx.save_for_backward(q,k,v,O,L)
        ctx.is_causal = bool(is_causal)

        return O

    @staticmethod
    def backward(ctx, grad_out):
        q,k,v,O,L = ctx.saved_tensors
        is_causal = ctx.is_causal
        dQ, dK, dV, d_is_causal = compiled_backward(q,k,v,O,L,is_causal,grad_out)
        return dQ, dK, dV, d_is_causal


