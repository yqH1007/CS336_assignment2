"""Optional CUDA profiling annotations shared by models and benchmarks."""

from contextlib import nullcontext

import torch


_CUDA_AVAILABLE = torch.cuda.is_available()


def nvtx_range(message: str):
    """Annotate a CPU launch range on CUDA; otherwise use a no-op context.

    The range's duration is not GPU execution time. No synchronization is added.
    """
    if _CUDA_AVAILABLE:
        # NVTX formats its first argument; pass labels as data so braces are literal.
        return torch.cuda.nvtx.range("{}", message)
    return nullcontext()
