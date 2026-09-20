import argparse
import csv
import sys
from itertools import product

from triton.testing import do_bench
import torch

from cs336_basics.model import scaled_dot_product_attention
from cs336_systems.flash_attention_triton import FlashAttentionTriton, get_tile_sizes


COLUMNS = (
    "seq_len", "d", "dtype", "tile_q_k_d",
    "attention_fwd_ms", "attention_bwd_ms", "attention_full_ms",
    "triton_fwd_ms", "triton_bwd_ms", "triton_full_ms",
)


def summarize_error(error):
    message = " ".join(str(error).split())
    detail = f"{type(error).__name__}: {message}"
    return "ERROR: " + (detail[:77] + "..." if len(detail) > 80 else detail)


def measure(q, k, v, dO, backend, mode):
    mask = output = forward = None
    result = "OOM"
    try:
        if backend == "attention":
            mask = (
                torch.arange(q.shape[1], device=q.device)[:, None]
                >= torch.arange(k.shape[1], device=k.device)[None, :]
            )
            forward = lambda: scaled_dot_product_attention(q, k, v, mask)
        else:
            forward = lambda: FlashAttentionTriton.apply(q, k, v, True)

        if mode == "forward":
            forward()
            result = do_bench(forward, grad_to_none=[q, k, v])
        elif mode == "backward":
            output = forward()
            output.backward(dO, retain_graph=True)
            result = do_bench(
                lambda: output.backward(dO, retain_graph=True),
                grad_to_none=[q, k, v],
            )
        else:
            forward().backward(dO)
            result = do_bench(
                lambda: forward().backward(dO), grad_to_none=[q, k, v]
            )
    except torch.cuda.OutOfMemoryError:
        pass
    except Exception as error:
        result = summarize_error(error)
        print(
            f"{backend}/{mode} shape={tuple(q.shape)} dtype={q.dtype}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )

    # The OOM traceback is released after leaving except. Drop the retained
    # backward graph and mask before clearing the allocator's unused blocks.
    del output, forward, mask
    q.grad = k.grad = v.grad = None
    torch.cuda.empty_cache()
    return result


def benchmark_measurements(q, k, v, dO):
    return tuple(
        measure(q, k, v, dO, backend, mode)
        for backend in ("attention", "triton")
        for mode in ("forward", "backward", "full")
    )


def benchmark_configuration(seq_len, d, dtype):
    q = k = v = dO = None
    allocated = False
    allocation_result = "OOM"
    tile = "x".join(str(size) for size in get_tile_sizes(d))
    try:
        q = torch.randn(
            (1, seq_len, d), requires_grad=True, device="cuda", dtype=dtype
        )
        k = torch.randn_like(q, requires_grad=True)
        v = torch.randn_like(q, requires_grad=True)
        dO = torch.randn_like(q)
        allocated = True
    except torch.cuda.OutOfMemoryError:
        pass
    except Exception as error:
        allocation_result = summarize_error(error)
        print(
            f"input allocation seq_len={seq_len} d={d} dtype={dtype}: "
            f"{type(error).__name__}: {error}",
            file=sys.stderr,
        )

    # Shared-input allocation failure applies to all six measurements.
    times = (
        benchmark_measurements(q, k, v, dO)
        if allocated else (allocation_result,) * 6
    )
    del q, k, v, dO
    torch.cuda.empty_cache()
    return (seq_len, d, str(dtype).removeprefix("torch."), tile, *times)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark causal attention (ms).",
        epilog='For the first full scan, set TORCH_LOGS="recompiles" and save stderr.',
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run fp32 (128, 16) and (65536, 16); the latter stresses OOM handling.",
    )
    parser.add_argument("--csv", default="flash_benchmark.csv", help="Output CSV path.")
    args = parser.parse_args()

    configurations = (
        [(128, 16, torch.float32), (65536, 16, torch.float32)]
        if args.quick else product(
            [2**i for i in range(7, 17)],
            [2**i for i in range(4, 8)],
            [torch.bfloat16, torch.float32],
        )
    )
    rows = []
    # Persist each completed configuration even if a later run is interrupted.
    with open(args.csv, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(COLUMNS)
        csv_file.flush()
        for seq_len, d, dtype in configurations:
            row = benchmark_configuration(seq_len, d, dtype)
            writer.writerow(row)
            csv_file.flush()
            rows.append(row)

    formatted_rows = [
        [
            f"{value:.4f}" if isinstance(value, float) else str(value)
            for value in row
        ]
        for row in rows
    ]
    widths = [
        max(12, len(column), *(len(row[i]) for row in formatted_rows))
        for i, column in enumerate(COLUMNS)
    ]
    print(" ".join(f"{column:>{width}}" for column, width in zip(COLUMNS, widths)))
    for cells in formatted_rows:
        print(" ".join(f"{cell:>{width}}" for cell, width in zip(cells, widths)))

    print(f"Saved {len(rows)} configurations to {args.csv}")


if __name__ == "__main__":
    main()
