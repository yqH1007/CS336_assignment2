import torch
import timeit
import statistics
import argparse
import sys
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_basics.profiling import nvtx_range
from cs336_systems.common import get_device,synchronize

vocab_size = 10000

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32
    },
    "10b": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36
    }
}
PRECISION = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
}



def run_step(model,x,targets,optimizer,mode,ctx):
    with nvtx_range("forward"):
        with ctx:
            logits = model(x)
            if mode!="forward":
                loss = cross_entropy(logits,targets)
    if mode != "forward":
        with nvtx_range("backward"):
            loss.backward()
    if mode=="full":
        with nvtx_range("optimizer"):
            optimizer.step()
            optimizer.zero_grad()


parser = argparse.ArgumentParser()
parser.add_argument("--size",type=str,default="small",choices=MODEL_CONFIGS.keys())
parser.add_argument("--warmup",type=int,default=5)
parser.add_argument("--steps",type=int,default=None,help="Measured steps (default: 10; 3 with --memory-profile).")
parser.add_argument("--context-length",type=int,default=512)
parser.add_argument("--batch-size",type=int,default=4)
parser.add_argument("--mode",type=str,default="forward",choices=["forward","forward_backward","full"])
parser.add_argument("--compile",action="store_true")
parser.add_argument("--precision",choices=["fp32","bf16","fp16"],default="fp32")
parser.add_argument("--memory-profile",action="store_true",help="Record CUDA allocation history and peak memory instead of timing.")
parser.add_argument("--memory-profile-dir",type=Path,default=Path("memory_snapshots"),help="Directory for memory snapshots.")


args = parser.parse_args()
if args.steps is None:
    args.steps = 3 if args.memory_profile else 10
if args.steps < 1:
    parser.error("--steps must be at least 1")
if args.warmup < 0:
    parser.error("--warmup must be nonnegative")
device = get_device()
memory_profile = args.memory_profile and device == "cuda"
if args.memory_profile and not memory_profile:
    print(f"--memory-profile skipped: device={device}; CUDA memory recording is unavailable. Running normal timing.")
elif memory_profile:
    print("Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.")
    if args.steps > 3:
        print(f"Recording {args.steps} steps; 2-3 are recommended to limit snapshot size.")

snapshot_path = None
if memory_profile:
    args.memory_profile_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    snapshot_path = args.memory_profile_dir / (
        f"memory_{args.size}_{args.mode}_seq{args.context_length}_batch{args.batch_size}_"
        f"{args.precision}_compile{int(args.compile)}_steps{args.steps}_{timestamp}.pickle"
    )

mode = args.mode
precision = PRECISION[args.precision]
config_label = (
    f"size: {args.size} mode: {mode}, context-length: {args.context_length}, "
    f"batch-size: {args.batch_size}, compile: {args.compile}, precision: {precision}, steps: {args.steps}"
)
model = BasicsTransformerLM(**MODEL_CONFIGS[args.size],vocab_size=vocab_size,context_length=args.context_length).to(device=device)
if args.compile:
    model = torch.compile(model)
x = torch.randint(0,vocab_size,(args.batch_size,args.context_length),device=device)
optimizer = AdamW(model.parameters())

if mode=="forward":
    context = torch.no_grad()
else:
    context = nullcontext()
if args.precision=="fp32":
    ctx = nullcontext()
else:
    ctx = torch.autocast(device_type=device,dtype=precision,enabled=True)
with context:
    with nvtx_range("warmup"):
        for step in range(args.warmup):
            with nvtx_range(f"step_{step + 1}"):
                run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode,ctx=ctx)
        synchronize(device=device)
    times = []
    if memory_profile:
        torch.cuda.reset_peak_memory_stats(device=device)
        torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    measurement_complete = False
    try:
        with nvtx_range("measure"):
            if memory_profile:
                for step in range(args.steps):
                    with nvtx_range(f"step_{step + 1}"):
                        run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode,ctx=ctx)
                        synchronize(device=device)
            else:
                for step in range(args.steps):
                    with nvtx_range(f"step_{step + 1}"):
                        synchronize(device=device)
                        t0 = timeit.default_timer()
                        run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode,ctx=ctx)
                        synchronize(device=device)
                        t1 = timeit.default_timer()
                        times.append(t1-t0)
        measurement_complete = True
    finally:
        if memory_profile:
            try:
                torch.cuda.memory._dump_snapshot(str(snapshot_path))
            except Exception as error:
                # Keep a measurement failure (including OOM) as the primary error.
                print(f"Memory snapshot export failed: {error}", file=sys.stderr)
                if measurement_complete:
                    raise
            else:
                print(f"Memory snapshot: {snapshot_path.resolve()}")
            finally:
                # Export while history is enabled; always stop, even if export fails.
                torch.cuda.memory._record_memory_history(enabled=None)
            peak_bytes = torch.cuda.max_memory_allocated(device=device)
            peak_reserved_bytes = torch.cuda.max_memory_reserved(device=device)
            status = "complete" if measurement_complete else "incomplete"
            print(
                f"{config_label} | peak_allocated_bytes: {peak_bytes}, "
                f"peak_allocated_MiB: {peak_bytes / 2**20:.2f}, "
                f"peak_reserved_bytes: {peak_reserved_bytes}, "
                f"peak_reserved_MiB: {peak_reserved_bytes / 2**20:.2f}, status: {status} | "
                "timing: disabled (memory profiling)"
            )

if not memory_profile:
    mean = statistics.mean(times)*1000
    stdev = statistics.stdev(times)*1000 if len(times) > 1 else None
    timing = f"{mean:.2f} +- {stdev:.2f} ms" if stdev is not None else f"{mean:.2f} ms (stdev: N/A; one step)"
    print(f"{config_label}| {timing}")
