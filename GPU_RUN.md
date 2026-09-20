# GPU 运行说明（CS336 Assignment 2）

这份文档给**代为在 GPU 机器上执行的 agent**。请完整读完「角色与边界」再开始。

目标：在一次 GPU 会话里，把作业需要的**全部测量数据**采集完，带回本地做分析和写报告。

---

## 角色与边界（先读这一节）

这是 Stanford CS336 的课程作业，仓库根目录的 `CLAUDE.md` 规定了 AI 的使用边界。
你的职责是**运行和观测**，不是实现，也不是回答作业里的分析题。

**你应该做：**

- 按下面的顺序执行命令，把 stdout / stderr 完整保存到 `results/` 下
- 失败时报告**完整报错原文**，指出文件和行号
- 解释报错的含义，指出可疑的位置
- 处理**环境层面**的问题（缺依赖、缓存目录不可写、路径问题、nsys 权限）


---

## 0. 环境准备与自检

工作目录：仓库根目录（含 `pyproject.toml` 的那一层）。

```bash
mkdir -p results
uv sync
```

本地开发机是 macOS，没装 triton；GPU 机器上靠 `uv sync` 把它装上。

```bash
uv run python -c "import torch, triton; print('torch', torch.__version__); print('triton', triton.__version__); print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0))" 2>&1 | tee results/00_env.txt
nvidia-smi -L 2>&1 | tee -a results/00_env.txt
nsys --version 2>&1 | tee -a results/00_env.txt
```

**期望**：CUDA 可用、设备是 H100、torch 和 triton 版本号记录在案。

如果 `nsys` 不存在或没有权限，**先报告**，任务 F 会做不了，但其余任务不受影响。

如果容器里 `$HOME` 不可写，Triton 和 Inductor 的编译缓存会失败：

```bash
export TRITON_CACHE_DIR=/tmp/triton_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/inductor_cache
```

单卡即可，本文档所有任务都不需要多 GPU。

---

## 任务 A：正确性测试（必须最先做，不通过就停）

```bash
uv run pytest tests/ -q 2>&1 | tee results/01_pytest.txt
```

**期望**：`14 passed`。（本地 macOS 上是 10 passed + 4 skipped，4 个 skip 是没有 GPU。）

那 4 个之前被跳过的测试，是 Triton FlashAttention kernel **第一次真正被编译和执行**。
很可能在这里失败。

> **任务 A 没有全部通过就停止，不要继续。**
> 一个结果不正确的 kernel，后面测出来的性能数据没有意义。
> 报告：失败的测试名、参数（`is_causal` 是 True 还是 False）、完整报错、数值误差大小。

---

## 任务 B：模型基准表

`benchmark.py` 的参数：

| 参数 | 说明 |
|---|---|
| `--size` | small / medium / large / xl / 10b |
| `--mode` | forward / forward_backward / full |
| `--context-length` | 序列长度，默认 512 |
| `--batch-size` | 默认 4 |
| `--warmup` / `--steps` | 预热步数 / 计时步数 |
| `--compile` | 开启 `torch.compile` |
| `--precision` | fp32 / bf16 / fp16 |
| `--memory-profile` | 录制显存分配历史（此时计时无效） |

### B1. 五种规模 × 三种模式

```bash
for size in small medium large xl 10b; do
  for mode in forward forward_backward full; do
    uv run python cs336_systems/benchmark.py --size $size --mode $mode \
      --context-length 512 --batch-size 4 --warmup 5 --steps 10 \
      2>&1 | tee -a results/02_benchmark_base.txt || true
  done
done
```

**大模型 OOM 是预期内的**，`|| true` 保证循环继续。OOM 的那几组在报告里记成 OOM 即可。

### B2. 序列长度扫描

```bash
for ctx in 128 256 512 1024; do
  uv run python cs336_systems/benchmark.py --size medium --mode full \
    --context-length $ctx --batch-size 4 --warmup 5 --steps 10 \
    2>&1 | tee -a results/03_benchmark_ctx.txt || true
done
```

### B3. warmup 消融

```bash
for w in 0 1 2 5; do
  uv run python cs336_systems/benchmark.py --size medium --mode full \
    --context-length 512 --warmup $w --steps 10 \
    2>&1 | tee -a results/04_warmup_ablation.txt || true
done
```

**注意**：`--warmup 0` 时第一次计时里混着 cuBLAS 自动调优和分配器冷启动，
标准差会明显偏大。**这正是要观测的现象**，不要因为"数据难看"就重跑。

---

## 任务 C：torch.compile 对比

和 B1 同样的配置，加上 `--compile`。为控制时间只跑三种规模：

```bash
for size in small medium large; do
  for mode in forward forward_backward full; do
    uv run python cs336_systems/benchmark.py --size $size --mode $mode \
      --context-length 512 --batch-size 4 --warmup 5 --steps 10 --compile \
      2>&1 | tee -a results/05_compile.txt || true
  done
done
```

**每一条的第一次运行会卡住几十秒**，那是编译，不是死机。
`--warmup 5` 足够把编译时间吸收掉，`full` 模式下前向和反向会各编译一次。

如果出现 Dynamo 的重编译上限警告，**请报告**。

---

## 任务 D：混合精度

```bash
for prec in bf16 fp16; do
  for mode in forward full; do
    uv run python cs336_systems/benchmark.py --size medium --mode $mode \
      --context-length 512 --warmup 5 --steps 10 --precision $prec \
      2>&1 | tee -a results/06_precision.txt || true
  done
done
```

**说明**：fp16 **没有**使用 loss scaling，这是有意为之。
如果 fp16 跑出 NaN 或者数值异常，那是预期内的观测结果，照实记录，不要去加 scaler。

---

## 任务 E：显存剖析

```bash
for ctx in 128 256 512; do
  for mode in forward full; do
    uv run python cs336_systems/benchmark.py --size large --mode $mode \
      --context-length $ctx --warmup 3 --memory-profile \
      2>&1 | tee -a results/07_memory.txt || true
  done
done
```

再补一组混合精度下的对比：

```bash
uv run python cs336_systems/benchmark.py --size large --mode full \
  --context-length 512 --warmup 3 --memory-profile --precision bf16 \
  2>&1 | tee -a results/07_memory.txt || true
```

**要点**：

- 开了 `--memory-profile` 时脚本**不输出时间**，只输出峰值显存和快照路径。这是设计如此
- 快照 pickle 写在 `memory_snapshots/` 下，文件名带完整配置
- 快照文件可能有几百 MB。**确认磁盘空间**，并在交付时说明总大小
- 分析要在 https://pytorch.org/memory_viz 上做，**由学生本人做**，你只负责把 pickle 带回来

---

## 任务 F：nsys 性能剖析

需要 `nsys` 可用。先跑一条最小的确认能出报告：

```bash
nsys profile -o results/nsys_smoke --force-overwrite true \
  --trace=cuda,nvtx,osrt \
  uv run python cs336_systems/benchmark.py --size small --mode full \
    --context-length 512 --warmup 2 --steps 3 \
  2>&1 | tee results/08_nsys_smoke.txt
```

成功后跑正式的两组：

```bash
for mode in forward full; do
  nsys profile -o results/nsys_medium_$mode --force-overwrite true \
    --trace=cuda,nvtx,osrt \
    uv run python cs336_systems/benchmark.py --size medium --mode $mode \
      --context-length 512 --warmup 2 --steps 3 \
    2>&1 | tee -a results/09_nsys.txt || true
done
```

然后把报告转成文本带回来（学生本地不一定装了 nsys GUI）：

```bash
for rep in results/nsys_medium_*.nsys-rep; do
  nsys stats --report cuda_gpu_kern_sum "$rep" 2>&1 | tee -a results/10_nsys_kernels.txt
  nsys stats --report nvtx_sum "$rep" 2>&1 | tee -a results/10_nsys_kernels.txt
done
```

代码里已经打好的 NVTX 区间：

```
  warmup / measure                     <- 最外层，用来区分预热和正式测量
    step_1, step_2, ...                <- 每一步
      forward / backward / optimizer   <- 一步内部的三段
        attention/qk                   <- attention 内部三段
        attention/softmax
        attention/pv
```

**如果 nsys 因为权限问题失败**（容器里常见），报告完整报错。通常需要
`--cap-add=SYS_ADMIN` 或者宿主机的 `perf_event_paranoid` 设置，这类问题**由学生去协调**，
你不要尝试修改宿主机配置。

---

## 任务 G：FlashAttention 基准扫描（耗时最长，放最后）

### G1. 冒烟测试

```bash
uv run python cs336_systems/flash_benchmark.py --quick 2>&1 | tee results/11_flash_quick.txt
```

只跑两组配置：`(seq=128, d=16, fp32)` 和 `(seq=65536, d=16, fp32)`。

**期望看到两种行为**：

1. 小配置：6 个数字全部正常输出（毫秒）
2. 大配置：至少有一项（多半是对照组 attention 的 backward）记成 `OOM`，
   **脚本不崩溃**，继续跑完并写出 CSV

大配置全部跑出数字也可以接受（H100 有 80GB）。但**整个脚本因 OOM 退出就是 bug**，请报告。

### G2. 完整扫描

```bash
TORCH_LOGS="recompiles" uv run python cs336_systems/flash_benchmark.py \
  --csv results/flash_benchmark.csv \
  2> results/12_recompiles.log | tee results/12_flash_full.txt
```

扫描空间：10 种序列长度 × 4 种维度 × 2 种精度 = **80 组配置**，每组 6 次测量。
预计 **20 到 40 分钟**，大部分时间花在 Triton kernel 的重复编译上
（`N_Q`、`N_K`、`D` 都是 `tl.constexpr`，每组配置都要重新编译）。

脚本每跑完一组就 flush 一次 CSV，中途被打断也不会丢已有结果。

跑完检查 `results/12_recompiles.log`：如果出现**重编译次数达到上限**之类的警告，
说明编译过的 backward 在某个时刻退回了 eager 模式，后面的 backward 数据会整体偏慢。
**这一点必须报告**，否则表格会被误读。

CSV 的列：

```
seq_len, d, dtype, tile_q_k_d,
attention_fwd_ms, attention_bwd_ms, attention_full_ms,
triton_fwd_ms,    triton_bwd_ms,    triton_full_ms
```

单元格里可能出现数字（毫秒）、`OOM`、以 `ERROR:` 开头的字符串。
后两种都是**预期内**的，不要试图消除它们。

---

## 交付清单

全部打包带回：

| 路径 | 内容 |
|---|---|
| `results/00_env.txt` | 版本号、GPU 型号、nsys 可用性 |
| `results/01_pytest.txt` | 正确性测试完整输出 |
| `results/02_benchmark_base.txt` | 五种规模 × 三种模式 |
| `results/03_benchmark_ctx.txt` | 序列长度扫描 |
| `results/04_warmup_ablation.txt` | warmup 消融 |
| `results/05_compile.txt` | torch.compile 对比 |
| `results/06_precision.txt` | 混合精度 |
| `results/07_memory.txt` | 显存峰值 |
| `memory_snapshots/*.pickle` | 显存快照（文件可能很大，说明总大小） |
| `results/08_nsys_smoke.txt`、`09_nsys.txt` | nsys 运行日志 |
| `results/nsys_*.nsys-rep` | nsys 报告 |
| `results/10_nsys_kernels.txt` | kernel 和 NVTX 汇总（文本） |
| `results/11_flash_quick.txt` | FlashAttention 冒烟测试 |
| `results/flash_benchmark.csv` | **FlashAttention 80 行结果表** |
| `results/12_flash_full.txt`、`12_recompiles.log` | 扫描日志和重编译日志 |

另外用文字汇报：

1. 哪些任务完整跑完、哪些失败或跳过，各自原因
2. 所有 OOM 出现在哪些配置上
3. 任何异常现象（NaN、重编译警告、数值不匹配、耗时明显反常）

**不要**在汇报里写 handout 分析题的答案，也不要替学生下结论。
描述观测到的现象即可。

---

## 故障对照表

| 现象 | 含义 | 你该做什么 |
|---|---|---|
| `ModuleNotFoundError: No module named 'triton'` | 依赖没装 | 跑 `uv sync`，仍失败则报告 |
| Triton 报 `dot()` 参数相关的错误 | 代码用了 `tl.dot` 的第三个累加器参数，可能装的 Triton 版本太旧 | **报告 triton 版本号**，不要改代码 |
| 编译缓存权限错误 | `$HOME` 不可写 | 设置第 0 节那两个环境变量后重试 |
| pytest 的 `assert_close` 失败 | kernel 数值不对 | 报告**最大误差、失败的测试名和 `is_causal` 取值** |
| pytest 只在 `is_causal=True` 时失败 | causal 掩码逻辑问题 | 明确说明只有 causal 这一侧失败 |
| 大模型 / 长序列 OOM | 预期内 | 记录后继续 |
| 小配置（seq ≤ 1024）就 OOM | 不正常 | 报告，可能有显存泄漏 |
| flash_benchmark 整体因 OOM 退出 | OOM 处理有 bug | 报告完整 traceback |
| Dynamo 重编译上限警告 | 编译过的 backward 退回 eager | **必须报告**，会影响数据解读 |
| nsys 权限错误 | 容器缺 `SYS_ADMIN` 等能力 | 报告，由学生协调，不要改宿主机配置 |
| fp16 出现 NaN | 没用 loss scaling，预期内 | 照实记录 |

---

## 相关文件位置

| 文件 | 作用 |
|---|---|
| `cs336_systems/benchmark.py` | 模型基准 / compile / 混合精度 / 显存剖析 |
| `cs336_systems/flash_benchmark.py` | FlashAttention 扫描 |
| `cs336_systems/flash_attention_triton.py` | Triton kernel 和 `FlashAttentionTriton` |
| `cs336_systems/flash_attention.py` | PyTorch 版 FlashAttention，以及被 `torch.compile` 编译的共享 backward |
| `cs336-basics/cs336_basics/profiling.py` | `nvtx_range` 包装 |
| `cs336-basics/cs336_basics/model.py` | attention 内部的 NVTX 标记 |
| `tests/` | 正确性测试 |

以上文件**都不要修改**。

---

## 时间预算

| 任务 | 预计 |
|---|---|
| 0 环境 + A 正确性 | 10-30 分钟（kernel 首次上 GPU，可能要调试） |
| B 基准表 | 15-25 分钟 |
| C compile | 15-20 分钟（含编译时间） |
| D 混合精度 | 5-10 分钟 |
| E 显存 | 10 分钟 |
| F nsys | 10-15 分钟 |
| G FlashAttention 扫描 | 25-45 分钟 |

总计约 **1.5 到 2.5 小时**。如果时间不够，按 A -> G -> B -> E -> F -> C -> D 的优先级砍，
A 和 G 是必须完成的（G 是独立计分的交付物）。
