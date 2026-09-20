import torch
import timeit
import statistics
import argparse
from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
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


def run_step(model,x,targets,optimizer,mode):
    logits = model(x)
    if mode!="forward":
        loss = cross_entropy(logits,targets)
        loss.backward()
    if mode=="full":
        optimizer.step()
        optimizer.zero_grad()


parser = argparse.ArgumentParser()
parser.add_argument("--size",type=str,default="small",choices=MODEL_CONFIGS.keys())
parser.add_argument("--warmup",type=int,default=5)
parser.add_argument("--steps",type=int,default=10)
parser.add_argument("--context-length",type=int,default=512)
parser.add_argument("--batch-size",type=int,default=4)
parser.add_argument("--mode",type=str,default="forward",choices=["forward","forward_backward","full"])


args = parser.parse_args()
device = get_device()
mode = args.mode
model = BasicsTransformerLM(**MODEL_CONFIGS[args.size],vocab_size=vocab_size,context_length=args.context_length).to(device=device)
x = torch.randint(0,vocab_size,(args.batch_size,args.context_length),device=device)
optimizer = AdamW(model.parameters())

if mode=="forward":
    context = torch.no_grad()
else:
    context = nullcontext()
with context:
    for _ in range(args.warmup):
        run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode)
    synchronize(device=device)
    times = []
    for _ in range(args.steps):
        synchronize(device=device)
        t0 = timeit.default_timer()
        run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode)
        synchronize(device=device)
        t1 = timeit.default_timer()
        times.append(t1-t0)

mean = statistics.mean(times)*1000
stdev = statistics.stdev(times)*1000

print(f"{args.size} | {mean:.2f} +- {stdev:.2f} ms")

