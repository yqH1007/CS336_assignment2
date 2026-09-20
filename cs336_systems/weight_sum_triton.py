import triton
import triton.language as tl
import torch
from einops import rearrange

@triton.jit
def weighted_sum_fwd(
    x_ptr,weight_ptr,
    output_ptr,
    x_stride_row,x_stride_dim,
    weight_stride_dim,
    output_stride_row,
    NUM_ROWS,D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
    row_tile_idx = tl.program_id(0)
    x_block_ptr = tl.make_block_ptr(
        ## 描述一个矩阵如何tile，这里的row和column都进行的tile
        x_ptr,
        shape = (NUM_ROWS,D,),
        strides = (x_stride_row,x_stride_dim),
        offsets = (row_tile_idx * ROWS_TILE_SIZE,0),## row_tile_idx 决定了这个kernel操作整个矩阵的哪几行
        block_shape = (ROWS_TILE_SIZE,D_TILE_SIZE),
        order = (1,0),
    )
    weight_block_ptr = tl.make_block_ptr(
        ## weight的row进行了tile
        weight_ptr,
        shape=(D,),
        strides=(weight_stride_dim,),## tile的strides定义
        offsets=(0,),
        block_shape=(D_TILE_SIZE,),
        order=(0,),
    )
    output_block_ptr = tl.make_block_ptr(
        output_ptr,
        shape=(NUM_ROWS,),
        strides=(output_stride_row,),
        offsets=(row_tile_idx * ROWS_TILE_SIZE,),
        block_shape=(ROWS_TILE_SIZE,),
        order=(0,),
    )

    output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)
    for i in range(tl.cdiv(D, D_TILE_SIZE)):## column方向通过循环来计算，row方向通过不同kernel计算
        # Load the current block pointer
        # Since ROWS_TILE_SIZE might not divide NUM_ROWS, and D_TILE_SIZE might not divide D,
        # we need boundary checks for both dimensions
        row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
        weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
        # Compute the weighted sum of the row.
        output += tl.sum(row * weight[None, :], axis=1) ## equal: ( weight * x ).sum(axis=-1)
        # Move the pointers to the next tile.
        # These are (rows, columns) coordinate deltas
        x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE)) # 移动对应的指针，进行下一部分列的计算
        weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,)) # Move by D_TILE_SIZE
        # Write output to the output block pointer (a single scalar per row).
        # Since ROWS_TILE_SIZE might not divide NUM_ROWS, we need boundary checks
    tl.store(output_block_ptr, output, boundary_check=(0,))

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight):
        D, output_dims = x.shape[-1],x.shape[:-1]
        input_shape = x.shape
        x = rearrange(x,"... d -> (...) d")

        ctx.save_for_backward(x,weight)

        assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension
        ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
        ctx.input_shape = input_shape
        # Need to initialize empty result tensor. Note that these elements are not necessarily 0!
        y = torch.empty(output_dims, device=x.device)
        # Launch our kernel with n instances in our 1D grid.
        n_rows = y.numel()
        weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
        x, weight,
        y,
        x.stride(0), x.stride(1),
        weight.stride(0),
        y.stride(0),
        NUM_ROWS=n_rows, D=D,
        ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
        )
        return y.view(input_shape[:-1])
