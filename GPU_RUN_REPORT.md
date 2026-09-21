# CS336 Assignment 2 — GPU 运行记录

本文件汇总了一次完整 GPU 测量产生的全部文本产物（日志 + CSV）。
二进制产物（nsys 报告、显存快照 pickle）无法内嵌，路径在文末列出。

## 运行概况

| 项 | 值 |
|---|---|
| 仓库 | https://github.com/yqH1007/CS336_assignment2.git |
| 平台 | Autel Crater, 作业 jpt-a26473-260920-d8099（已释放） |
| 节点 / 区域 | k3s-slave-a800-8 / hw-bj4（华为云-北京四） |
| GPU | NVIDIA A800-SXM4-80GB × 1，驱动 580.173.02 |
| 软件栈 | torch 2.11.0+cu130 / triton 3.6.0 / Python 3.13 / nsys 2026.3.2 |
| 测量窗口 | 2026-09-20 13:52:49 – 14:22:53 UTC（30 分钟） |
| 结果 | 9 个任务全部完成，无失败、无跳过 |

**环境偏差**：`GPU_RUN.md` 预期 H100。`pyproject.toml` 锁定的 `torch~=2.11.0` 只发布 CUDA 13 wheel，而当时所有空闲 H100 都在 560.35.03（CUDA 12.6）驱动节点上，CUDA 前向兼容被拒（错误 803，560 非 LTS 分支）。改用同为 80GB 显存的 A800，OOM 边界与文档假设一致，但绝对耗时不是 H100 的数字。

## 各任务耗时

| 任务 | 起始 (UTC) | 耗时 | 结果 |
|---|---|---|---|
| 0 环境自检 | 13:52:49 | 3 秒 | 通过 |
| A 正确性测试 | 13:52:52 | 1 分 19 秒 | 14 passed |
| G1 flash 冒烟 | 13:54:11 | 24 秒 | 两种预期行为都出现 |
| G2 flash 完整扫描 | 13:54:35 | 3 分 45 秒 | 80/80 写入 CSV |
| B1 规模 × 模式 | 13:58:20 | 13 分 41 秒 | 13/15，10b 两组 OOM |
| B2 序列长度 | 14:12:01 | 1 分 16 秒 | 4/4 |
| B3 warmup 消融 | 14:13:17 | 1 分 10 秒 | 4/4 |
| E 显存剖析 | 14:14:27 | 2 分 03 秒 | 7/7 快照 |
| F nsys | 14:16:30 | 1 分 25 秒 | 冒烟 + 2 组正式 |
| C torch.compile | 14:17:55 | 3 分 59 秒 | 9/9 |
| D 混合精度 | 14:21:54 | 59 秒 | 4/4 |

## 目录

- [FlashAttention 结果表](#flashattention-结果表)
- [环境自检](#环境自检)
- [任务 A — 正确性测试](#任务-a--正确性测试)
- [任务 B1 — 五种规模 × 三种模式](#任务-b1--五种规模--三种模式)
- [任务 B2 — 序列长度扫描](#任务-b2--序列长度扫描)
- [任务 B3 — warmup 消融](#任务-b3--warmup-消融)
- [任务 C — torch.compile 对比](#任务-c--torchcompile-对比)
- [任务 D — 混合精度](#任务-d--混合精度)
- [任务 E — 显存剖析](#任务-e--显存剖析)
- [任务 F — nsys 冒烟](#任务-f--nsys-冒烟)
- [任务 F — nsys 正式采集](#任务-f--nsys-正式采集)
- [任务 F — kernel 与 NVTX 汇总](#任务-f--kernel-与-nvtx-汇总)
- [任务 G1 — FlashAttention 冒烟](#任务-g1--flashattention-冒烟)
- [任务 G2 — FlashAttention 完整扫描日志](#任务-g2--flashattention-完整扫描日志)
- [任务 G2 — Dynamo 重编译日志](#任务-g2--dynamo-重编译日志)
- [无法内嵌的二进制产物](#无法内嵌的二进制产物)
- [复现](#复现)

---

## FlashAttention 结果表

来源 results/flash_benchmark.csv。80 组配置，每组 6 次测量。同一行的 attention_*（对照组）与 triton_*（你的 kernel）才是可比的。单元格可能是毫秒数、OOM 或 ERROR: 开头的字符串，后两种均为预期内。

| seq_len | d | dtype | tile_q_k_d | attention_fwd_ms | attention_bwd_ms | attention_full_ms | triton_fwd_ms | triton_bwd_ms | triton_full_ms |
|---|---|---|---|---|---|---|---|---|---|
| 128 | 16 | bfloat16 | 16x16x16 | 0.2165 | 0.5943 | 1.0387 | 0.0141 | 0.2830 | 0.3240 |
| 128 | 16 | float32 | 16x16x16 | 0.3093 | 0.4372 | 0.8925 | 0.0168 | 0.1958 | 0.4051 |
| 128 | 32 | bfloat16 | 16x16x32 | 0.2019 | 0.3761 | 1.0207 | 0.0129 | 0.3181 | 0.4335 |
| 128 | 32 | float32 | 16x16x32 | 0.2099 | 0.4037 | 0.8796 | 0.0291 | 0.2548 | 0.3482 |
| 128 | 64 | bfloat16 | 16x16x64 | 0.2094 | 0.5031 | 0.8748 | 0.0127 | 0.3425 | 0.5033 |
| 128 | 64 | float32 | 16x16x64 | 0.2521 | 0.3046 | 0.8423 | 0.0261 | 0.2568 | 0.5210 |
| 128 | 128 | bfloat16 | 16x16x128 | 0.2208 | 0.3734 | 0.8435 | 0.0146 | 0.3469 | 0.4845 |
| 128 | 128 | float32 | 16x16x128 | 0.2861 | 0.3724 | 0.7584 | 0.0422 | 0.3126 | 0.3172 |
| 256 | 16 | bfloat16 | 16x16x16 | 0.2640 | 0.4552 | 0.9938 | 0.0163 | 0.2521 | 0.4973 |
| 256 | 16 | float32 | 16x16x16 | 0.3171 | 0.4638 | 0.9903 | 0.0218 | 0.3949 | 0.4118 |
| 256 | 32 | bfloat16 | 16x16x32 | 0.2478 | 0.4295 | 0.8631 | 0.0188 | 0.3723 | 0.3077 |
| 256 | 32 | float32 | 16x16x32 | 0.2190 | 0.3908 | 0.8127 | 0.0293 | 0.3313 | 0.4801 |
| 256 | 64 | bfloat16 | 16x16x64 | 0.2717 | 0.5205 | 0.9714 | 0.0181 | 0.2882 | 0.4981 |
| 256 | 64 | float32 | 16x16x64 | 0.2256 | 0.4805 | 0.9208 | 0.0447 | 0.2075 | 0.4299 |
| 256 | 128 | bfloat16 | 16x16x128 | 0.2754 | 0.3233 | 0.9217 | 0.0204 | 0.1662 | 0.2685 |
| 256 | 128 | float32 | 16x16x128 | 0.2212 | 0.5185 | 1.0016 | 0.0759 | 0.2697 | 0.4635 |
| 512 | 16 | bfloat16 | 16x16x16 | 0.2903 | 0.4301 | 0.9696 | 0.0240 | 0.2590 | 0.3648 |
| 512 | 16 | float32 | 16x16x16 | 0.2334 | 0.3922 | 0.8373 | 0.0321 | 0.2954 | 0.4171 |
| 512 | 32 | bfloat16 | 16x16x32 | 0.2548 | 0.5088 | 0.9069 | 0.0283 | 0.3282 | 0.4391 |
| 512 | 32 | float32 | 16x16x32 | 0.2490 | 0.3947 | 0.8384 | 0.0520 | 0.2926 | 0.4137 |
| 512 | 64 | bfloat16 | 16x16x64 | 0.2278 | 0.4075 | 0.8673 | 0.0289 | 0.3405 | 0.4169 |
| 512 | 64 | float32 | 16x16x64 | 0.2438 | 0.3313 | 1.3801 | 0.0824 | 0.2746 | 0.5424 |
| 512 | 128 | bfloat16 | 16x16x128 | 0.2468 | 0.4636 | 0.9851 | 0.0335 | 0.3479 | 0.4234 |
| 512 | 128 | float32 | 16x16x128 | 0.2110 | 0.4174 | 0.6351 | 0.1428 | 0.1719 | 0.2670 |
| 1024 | 16 | bfloat16 | 16x16x16 | 0.1522 | 0.4807 | 1.1912 | 0.0412 | 0.3831 | 0.3908 |
| 1024 | 16 | float32 | 16x16x16 | 0.2879 | 0.5464 | 1.0299 | 0.0560 | 0.3282 | 0.6156 |
| 1024 | 32 | bfloat16 | 16x16x32 | 0.3106 | 0.5142 | 0.9182 | 0.0490 | 0.4265 | 0.5005 |
| 1024 | 32 | float32 | 16x16x32 | 0.2417 | 0.4106 | 0.8291 | 0.0955 | 0.3214 | 0.5239 |
| 1024 | 64 | bfloat16 | 16x16x64 | 0.2272 | 0.3724 | 0.8781 | 0.0505 | 0.1904 | 0.4325 |
| 1024 | 64 | float32 | 16x16x64 | 0.2171 | 0.4399 | 0.8820 | 0.1576 | 0.3482 | 0.5344 |
| 1024 | 128 | bfloat16 | 16x16x128 | 0.2259 | 0.3469 | 0.7984 | 0.0603 | 0.2541 | 0.4453 |
| 1024 | 128 | float32 | 16x16x128 | 0.2581 | 0.5045 | 0.9613 | 0.2778 | 0.2072 | 0.5235 |
| 2048 | 16 | bfloat16 | 16x16x16 | 0.2183 | 0.4460 | 0.9393 | 0.0756 | 0.2890 | 0.4335 |
| 2048 | 16 | float32 | 16x16x16 | 0.2689 | 0.5524 | 0.7778 | 0.1096 | 0.2198 | 0.3249 |
| 2048 | 32 | bfloat16 | 16x16x32 | 0.1615 | 0.2746 | 0.8845 | 0.0914 | 0.3752 | 0.2985 |
| 2048 | 32 | float32 | 16x16x32 | 0.2876 | 0.4894 | 1.0442 | 0.2013 | 0.2661 | 0.4353 |
| 2048 | 64 | bfloat16 | 16x16x64 | 0.2351 | 0.3693 | 0.8967 | 0.0966 | 0.3164 | 0.4230 |
| 2048 | 64 | float32 | 16x16x64 | 0.2389 | 0.4925 | 1.0991 | 0.3401 | 0.3129 | 0.6412 |
| 2048 | 128 | bfloat16 | 16x16x128 | 0.2452 | 0.4570 | 0.9425 | 0.1162 | 0.2888 | 0.5162 |
| 2048 | 128 | float32 | 16x16x128 | 0.3188 | 0.6491 | 1.0602 | 0.6142 | 0.4914 | 1.1016 |
| 4096 | 16 | bfloat16 | 16x16x16 | 0.4635 | 0.9877 | 1.4286 | 0.1602 | 0.4113 | 0.5609 |
| 4096 | 16 | float32 | 16x16x16 | 0.7109 | 1.5466 | 2.2483 | 0.2612 | 0.6212 | 0.8766 |
| 4096 | 32 | bfloat16 | 16x16x32 | 0.4628 | 0.9567 | 1.3971 | 0.1877 | 0.4440 | 0.5944 |
| 4096 | 32 | float32 | 16x16x32 | 0.7379 | 1.5690 | 2.2932 | 0.5490 | 0.6738 | 1.2182 |
| 4096 | 64 | bfloat16 | 16x16x64 | 0.4711 | 0.9698 | 1.4202 | 0.2034 | 0.4211 | 0.6175 |
| 4096 | 64 | float32 | 16x16x64 | 0.8449 | 1.8081 | 2.6426 | 0.9652 | 0.9855 | 1.9450 |
| 4096 | 128 | bfloat16 | 16x16x128 | 0.4885 | 0.9994 | 1.4705 | 0.2539 | 0.4989 | 0.7249 |
| 4096 | 128 | float32 | 16x16x128 | 1.0732 | 2.3311 | 3.4003 | 1.7934 | 1.6263 | 3.4146 |
| 8192 | 16 | bfloat16 | 16x16x16 | 1.7011 | 3.4628 | 5.1261 | 0.3873 | 1.4005 | 1.7839 |
| 8192 | 16 | float32 | 16x16x16 | 2.5704 | 5.7622 | 8.3284 | 0.7402 | 2.2704 | 3.0076 |
| 8192 | 32 | bfloat16 | 16x16x32 | 1.7018 | 3.4297 | 5.0985 | 0.4374 | 1.4073 | 1.8408 |
| 8192 | 32 | float32 | 16x16x32 | 2.6765 | 5.7384 | 8.4040 | 1.7615 | 2.4807 | 4.2370 |
| 8192 | 64 | bfloat16 | 16x16x64 | 1.7432 | 3.4620 | 5.1771 | 0.4924 | 1.5073 | 1.9954 |
| 8192 | 64 | float32 | 16x16x64 | 3.1217 | 6.8581 | 9.9762 | 3.1337 | 3.7750 | 6.8991 |
| 8192 | 128 | bfloat16 | 16x16x128 | 1.7915 | 3.5352 | 5.2970 | 0.6436 | 1.6328 | 2.2746 |
| 8192 | 128 | float32 | 16x16x128 | 4.0635 | 8.6363 | 12.6948 | 6.3309 | 6.0667 | 12.3940 |
| 16384 | 16 | bfloat16 | 16x16x16 | 6.3497 | 13.1706 | 19.5190 | 1.0848 | 5.5322 | 6.6014 |
| 16384 | 16 | float32 | 16x16x16 | 9.8312 | 22.0792 | 31.9093 | 2.5938 | 9.1310 | 11.6073 |
| 16384 | 32 | bfloat16 | 16x16x32 | 6.4236 | 13.1354 | 19.4717 | 1.3336 | 5.4861 | 6.8072 |
| 16384 | 32 | float32 | 16x16x32 | 10.2126 | 22.6460 | 32.8657 | 6.3802 | 9.8947 | 16.1655 |
| 16384 | 64 | bfloat16 | 16x16x64 | 6.5164 | 13.2429 | 19.7080 | 1.5426 | 5.7304 | 7.2591 |
| 16384 | 64 | float32 | 16x16x64 | 12.2667 | 26.4639 | 38.7724 | 11.4756 | 14.5551 | 26.0061 |
| 16384 | 128 | bfloat16 | 16x16x128 | 6.6872 | 13.4524 | 20.1210 | 2.0913 | 6.1296 | 8.2386 |
| 16384 | 128 | float32 | 16x16x128 | 15.9361 | 33.8924 | 49.8077 | 22.0262 | 23.8589 | 45.8648 |
| 32768 | 16 | bfloat16 | 16x16x16 | 24.9868 | 52.8052 | 77.7951 | 3.6994 | 21.5686 | 25.2466 |
| 32768 | 16 | float32 | 16x16x16 | 39.4783 | 87.9300 | 127.4157 | 8.9892 | 36.6850 | 45.0999 |
| 32768 | 32 | bfloat16 | 16x16x32 | 25.5090 | 52.0594 | 77.0635 | 4.4509 | 21.6162 | 26.0577 |
| 32768 | 32 | float32 | 16x16x32 | 40.9567 | 89.7729 | 130.7041 | 22.5449 | 40.1644 | 61.8350 |
| 32768 | 64 | bfloat16 | 16x16x64 | 25.9389 | 52.5728 | 78.0089 | 5.1813 | 22.5228 | 27.7295 |
| 32768 | 64 | float32 | 16x16x64 | 47.9492 | 104.0436 | 152.2513 | 40.4639 | 57.6551 | 97.6355 |
| 32768 | 128 | bfloat16 | 16x16x128 | 26.5820 | 53.3141 | 79.4739 | 7.1043 | 24.4590 | 31.4669 |
| 32768 | 128 | float32 | 16x16x128 | 62.0894 | 132.4214 | 194.5049 | 78.6945 | 93.2642 | 171.9147 |
| 65536 | 16 | bfloat16 | 16x16x16 | 99.8666 | 207.9901 | 307.8762 | 13.1775 | 92.3919 | 107.1703 |
| 65536 | 16 | float32 | 16x16x16 | OOM | OOM | OOM | 33.7678 | OOM | OOM |
| 65536 | 32 | bfloat16 | 16x16x32 | 101.8324 | 208.2048 | 308.1200 | 16.2345 | 94.0848 | 110.3473 |
| 65536 | 32 | float32 | 16x16x32 | OOM | OOM | OOM | 86.7124 | OOM | OOM |
| 65536 | 64 | bfloat16 | 16x16x64 | 103.4907 | 209.8878 | 311.3236 | 19.0281 | 97.2985 | 114.6689 |
| 65536 | 64 | float32 | 16x16x64 | OOM | OOM | OOM | 155.1774 | OOM | OOM |
| 65536 | 128 | bfloat16 | 16x16x128 | 106.2378 | 214.9245 | 321.3840 | 26.3975 | 106.3672 | 133.3189 |
| 65536 | 128 | float32 | 16x16x128 | OOM | OOM | OOM | 296.7409 | OOM | OOM |

共 80 行，其中 4 行含 OOM（全部为 seq_len=65536 + float32），0 行含 `ERROR:`。

<details>
<summary>展开原始 CSV（完整精度）</summary>

```csv
seq_len,d,dtype,tile_q_k_d,attention_fwd_ms,attention_bwd_ms,attention_full_ms,triton_fwd_ms,triton_bwd_ms,triton_full_ms
128,16,bfloat16,16x16x16,0.21654241548234096,0.5943130810518522,1.038725563581439,0.014083636368249918,0.28301679132840574,0.3239746779203415
128,16,float32,16x16x16,0.3093216747565325,0.43718744539297544,0.8924975080310174,0.01676297432725103,0.19577232086284652,0.4050557121567662
128,32,bfloat16,16x16x32,0.2019374890816073,0.3760705777340465,1.0206812031567096,0.012878092183525687,0.3180537661453625,0.4334757886826992
128,32,float32,16x16x32,0.20987768818934757,0.4036567761532722,0.8796042573602894,0.029107992739635696,0.25484660937302356,0.3482059573239468
128,64,bfloat16,16x16x64,0.20938732999078394,0.5030711896811859,0.8748448944803494,0.012703290029108619,0.34248358695320386,0.5033061939530667
128,64,float32,16x16x64,0.2521426671051553,0.3045724434985055,0.8423488860006456,0.02606711824015733,0.2567874516536987,0.5209848292428871
128,128,bfloat16,16x16x128,0.22083290652284082,0.37339590454664756,0.8434502669234774,0.014577015844129381,0.34694594739734447,0.484503541461059
128,128,float32,16x16x128,0.28607793519399033,0.3723504764693124,0.7583968904283311,0.04218561781470099,0.31257715070847986,0.3171644609058853
256,16,bfloat16,16x16x16,0.26395450257012065,0.45516482496079597,0.9938388415004896,0.01633608435658715,0.25206957576852856,0.4973450025100879
256,16,float32,16x16x16,0.3171141811876328,0.46376240635867666,0.9902695673930494,0.021762688993280675,0.39489222869155854,0.4117801769272141
256,32,bfloat16,16x16x32,0.2477963106136888,0.4294616868180677,0.8630970405495685,0.018763034611626376,0.37231823473464787,0.307693599909544
256,32,float32,16x16x32,0.21903829960431453,0.3907528903197359,0.8126857830927923,0.029334896128288324,0.3312640003859997,0.48010322575195596
256,64,bfloat16,16x16x64,0.2717495306774422,0.5204844081401825,0.9713604214944338,0.01810693348455506,0.28816820473679694,0.49808014716420856
256,64,float32,16x16x64,0.22558543906010017,0.4804976115855136,0.9208130868998441,0.04471361855321307,0.20749275974738293,0.42990460920901524
256,128,bfloat16,16x16x128,0.2753563811115566,0.3233362418232542,0.9217118651438982,0.020423545115968077,0.16616094089021868,0.2685272923085542
256,128,float32,16x16x128,0.2212437145900183,0.5185477213804112,1.0016468823418136,0.07588577349872692,0.26966185111608076,0.4634997427736947
512,16,bfloat16,16x16x16,0.29028207385613597,0.43014580592131,0.96959826833493,0.023987116727542567,0.25904393550313887,0.3648131590023219
512,16,float32,16x16x16,0.23344407021476513,0.39221667738284094,0.8372759968042374,0.032141849840347,0.2954259836230396,0.41714475047774613
512,32,bfloat16,16x16x32,0.25484257952763084,0.5087675034075745,0.906924647647281,0.028311332100700887,0.3281764772845738,0.4391259996650311
512,32,float32,16x16x32,0.24897673120958552,0.39465520697191725,0.8383562336949741,0.052015313359033394,0.29256636095813,0.4136745831159156
512,64,bfloat16,16x16x64,0.22776029913503432,0.40751429491264873,0.8672581810455817,0.02893390204259486,0.340518448446224,0.4168548123671277
512,64,float32,16x16x64,0.243780822514399,0.33126470381087,1.3801447601545425,0.08241069570183754,0.27460882950712134,0.5423831584636104
512,128,bfloat16,16x16x128,0.2468134360691914,0.4636197269541546,0.9850928847116368,0.03347807773930389,0.34791925382391314,0.42344662802990035
512,128,float32,16x16x128,0.21104042870657785,0.4173602102625318,0.635079600661993,0.14284466450483027,0.17188822394692815,0.26704876617245055
1024,16,bfloat16,16x16x16,0.15219074131900004,0.48065798846451013,1.1911874095924566,0.04124478826530441,0.38312139942845463,0.39076385299364724
1024,16,float32,16x16x16,0.2878757048122309,0.5464487697642583,1.029941084839049,0.056046319413272774,0.32815608482823594,0.6155825712878233
1024,32,bfloat16,16x16x32,0.3106431251625278,0.5141969072818756,0.9181828381668808,0.04900486014502228,0.42650494324328075,0.5005461320612166
1024,32,float32,16x16x32,0.24169609951714613,0.4106447273357348,0.8290742196970515,0.09551681334514754,0.32136871407527734,0.5239275939728257
1024,64,bfloat16,16x16x64,0.22723797578589025,0.37244335772426984,0.878138130903244,0.050526403463231825,0.1904495483444583,0.4325118178332394
1024,64,float32,16x16x64,0.21713664722924977,0.43991154529214876,0.882027763654204,0.15760308409850282,0.34820410514368993,0.5344054327820832
1024,128,bfloat16,16x16x128,0.22592057630016998,0.3468764787912369,0.7983772728754126,0.0602738996584938,0.25411304917003286,0.4453023578402257
1024,128,float32,16x16x128,0.2580863030119376,0.5044602191809452,0.9612588101142162,0.2778413028420297,0.20722115191362672,0.523519069535061
2048,16,bfloat16,16x16x16,0.21834293968347182,0.4460431646100051,0.9392778987836357,0.07555557191052151,0.28898702692743894,0.4335056693068048
2048,16,float32,16x16x16,0.26894071203184455,0.5524183587461221,0.7778329445469764,0.10961738962763297,0.219817298257126,0.3248776492631831
2048,32,bfloat16,16x16x32,0.16150039016473583,0.2745660011967023,0.8845034041511479,0.09136275896728734,0.37516642407034384,0.2985382550037824
2048,32,float32,16x16x32,0.2875721210783178,0.4894494968075906,1.0442487529988558,0.20128067838218944,0.26610208491624243,0.4352981147338759
2048,64,bfloat16,16x16x64,0.23508140310574482,0.36927782505461315,0.8966646203627953,0.09664649482732429,0.31637836510026957,0.4230405802076513
2048,64,float32,16x16x64,0.2389137884659797,0.49248670930823973,1.099143225339151,0.34011313093431067,0.31288699088273225,0.6412169496611794
2048,128,bfloat16,16x16x128,0.2452202456963571,0.45702361342418624,0.9424635324164898,0.11621970432261898,0.28876583957219426,0.5162241458892822
2048,128,float32,16x16x128,0.3188329584832884,0.6491446393728256,1.0602073015078255,0.6142225032672286,0.49144554235576804,1.101604051227811
4096,16,bfloat16,16x16x16,0.4634639989282634,0.987743999470364,1.4286400001557147,0.1602432452062695,0.4113309759964315,0.5608706469719227
4096,16,float32,16x16x16,0.7108667590494814,1.5465986194281742,2.248251321839123,0.26119294964661033,0.6212191208591306,0.8765723736015791
4096,32,bfloat16,16x16x32,0.4627546294503016,0.9567266986813656,1.397054978779384,0.18771610521760426,0.44395217763053046,0.5944419462703964
4096,32,float32,16x16x32,0.7378708777720469,1.5689964232773617,2.2932097713152566,0.5489624428252379,0.6738168268368162,1.2181723957330408
4096,64,bfloat16,16x16x64,0.47107551156020744,0.9697794164164683,1.4202374181439799,0.20342608214649435,0.4211018484586861,0.61753053878381
4096,64,float32,16x16x64,0.8449314517133376,1.8080539609871658,2.642638635635376,0.9651594665315416,0.9855127597138995,1.9449729462887377
4096,128,bfloat16,16x16x128,0.4884604329516174,0.999448098154629,1.4704891775475173,0.2539467946404502,0.4988808159235936,0.7248730923000135
4096,128,float32,16x16x128,1.073197271765732,2.331145405769348,3.400330658312197,1.7934067153930664,1.62633841442612,3.41456800699234
8192,16,bfloat16,16x16x16,1.7011091281782906,3.4628302344569453,5.126096036699083,0.3873210950901634,1.4004746576150258,1.7839173302054405
8192,16,float32,16x16x16,2.5703875488705106,5.7621999979019165,8.328439279036088,0.7402166354972705,2.2704418439131517,3.00759612280747
8192,32,bfloat16,16x16x32,1.701826421719677,3.4296912528850414,5.098503059811062,0.4373522395502307,1.4073075427383672,1.8408466136973838
8192,32,float32,16x16x32,2.676533031463623,5.738433986902237,8.404037822376598,1.7614504557389479,2.480669332875146,4.237044442783702
8192,64,bfloat16,16x16x64,1.7431673820202167,3.4619733404230186,5.177125347985162,0.49238142231166765,1.5073004919185973,1.9954170455103335
8192,64,float32,16x16x64,3.1216608047485352,6.858098302568708,9.976167149013943,3.133693830172221,3.7750105476379394,6.899099486214774
8192,128,bfloat16,16x16x128,1.7914660350949156,3.5352047284444175,5.296995560328166,0.6435736811930134,1.6327953425320711,2.2745919942855837
8192,128,float32,16x16x128,4.063522774240245,8.636334592645818,12.694752011980329,6.3309397061665855,6.066749858856201,12.393961088997978
16384,16,bfloat16,16x16x16,6.349702390034993,13.170573779514857,19.5189697265625,1.0847570624532579,5.532193772933063,6.601423944745745
16384,16,float32,16x16x16,9.831207063463,22.07918405532837,31.909311930338543,2.5938142206933765,9.130998516082764,11.607311964035034
16384,32,bfloat16,16x16x32,6.423586334500994,13.135369300842285,19.471680068969725,1.3335866234195766,5.486142158508301,6.807179416928973
16384,32,float32,16x16x32,10.212622218661839,22.646039962768555,32.8657283782959,6.380177084604899,9.894710445404053,16.165514945983887
16384,64,bfloat16,16x16x64,6.516404594693865,13.24291215624128,19.70796012878418,1.5425681360697343,5.730449944734573,7.259103958423321
16384,64,float32,16x16x64,12.266748070716858,26.463850657145183,38.77241516113281,11.475624084472656,14.55507198969523,26.00606918334961
16384,128,bfloat16,16x16x128,6.687225137438093,13.452425003051758,20.120984077453613,2.091308649196181,6.129636255900065,8.238647200844504
16384,128,float32,16x16x128,15.93614387512207,33.89240074157715,49.80767822265625,22.02623176574707,23.858856201171875,45.864784240722656
32768,16,bfloat16,16x16x16,24.98681640625,52.805152893066406,77.79507446289062,3.699444456100464,21.568639755249023,25.246592203776043
32768,16,float32,16x16x16,39.47825622558594,87.93004608154297,127.41574096679688,8.989199924468995,36.68499183654785,45.099905014038086
32768,32,bfloat16,16x16x32,25.509024302164715,52.059391021728516,77.06352233886719,4.450858683813186,21.61616802215576,26.057685216267902
32768,32,float32,16x16x32,40.956735610961914,89.77289581298828,130.70407104492188,22.54487180709839,40.1644172668457,61.83500671386719
32768,64,bfloat16,16x16x64,25.93890126546224,52.57283020019531,78.00892639160156,5.181281725565593,22.52279233932495,27.729504267374676
32768,64,float32,16x16x64,47.94915199279785,104.04364776611328,152.2512664794922,40.46388816833496,57.65510559082031,97.6355209350586
32768,128,bfloat16,16x16x128,26.582037607828777,53.314144134521484,79.47388458251953,7.104285533611591,24.45901584625244,31.466911951700848
32768,128,float32,16x16x128,62.089439392089844,132.42137145996094,194.50489807128906,78.69446563720703,93.26419067382812,171.9147491455078
65536,16,bfloat16,16x16x16,99.86662292480469,207.9901123046875,307.87615966796875,13.177462714059013,92.39190673828125,107.17033386230469
65536,16,float32,16x16x16,OOM,OOM,OOM,33.76780891418457,OOM,OOM
65536,32,bfloat16,16x16x32,101.83235168457031,208.20477294921875,308.1199645996094,16.234533309936523,94.08483123779297,110.34729766845703
65536,32,float32,16x16x32,OOM,OOM,OOM,86.71241760253906,OOM,OOM
65536,64,bfloat16,16x16x64,103.49072265625,209.8877716064453,311.3235778808594,19.02805099487305,97.29853057861328,114.66893005371094
65536,64,float32,16x16x64,OOM,OOM,OOM,155.17738342285156,OOM,OOM
65536,128,bfloat16,16x16x128,106.23779296875,214.9245147705078,321.38397216796875,26.39746157328288,106.3671646118164,133.31887817382812
65536,128,float32,16x16x128,OOM,OOM,OOM,296.7408752441406,OOM,OOM
```

</details>

---

## 环境自检

来源 results/00_env.txt — torch / triton 版本、CUDA 可用性、GPU 型号、nsys 版本

```
torch 2.11.0+cu130
triton 3.6.0
cuda True
NVIDIA A800-SXM4-80GB
GPU 0: NVIDIA A800-SXM4-80GB (UUID: GPU-94da3560-9d64-1e21-4009-21c37101b277)
NVIDIA Nsight Systems version 2026.3.2.476-263238834031v0
```

---

## 任务 A — 正确性测试

来源 results/01_pytest.txt — uv run pytest tests/ -q 完整输出

```
tests/test_attention.py::test_flash_forward_pass_pytorch PASSED
tests/test_attention.py::test_flash_forward_pass_triton[False] PASSED
tests/test_attention.py::test_flash_forward_pass_triton[True] PASSED
tests/test_attention.py::test_flash_backward_pytorch PASSED
tests/test_attention.py::test_flash_backward_triton[False] PASSED
tests/test_attention.py::test_flash_backward_triton[True] PASSED
tests/test_ddp.py::test_DistributedDataParallel[ToyModel] PASSED
tests/test_ddp.py::test_DistributedDataParallel[ToyModelWithTiedWeights] PASSED
tests/test_fsdp.py::test_fsdp_correctness[fp32] <sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
<sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
PASSED
tests/test_fsdp.py::test_fsdp_correctness[fp16] <sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
<sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
PASSED
tests/test_fsdp.py::test_fsdp_gradient_sync[fp32] <sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
<sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
PASSED
tests/test_fsdp.py::test_fsdp_gradient_sync[fp16] <sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
<sys>:0: UserWarning: Full backward hook is firing when gradients are computed with respect to module outputs since no inputs require gradients. See https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook for more details.
PASSED
tests/test_sharded_optimizer.py::test_sharded_optimizer[ToyModel] PASSED
tests/test_sharded_optimizer.py::test_sharded_optimizer[ToyModelWithTiedWeights] PASSED

======================== 14 passed in 74.28s (0:01:14) =========================
pytest exit code: 0
```

---

## 任务 B1 — 五种规模 × 三种模式

来源 results/02_benchmark_base.txt — ctx=512, batch=4, warmup=5, steps=10

```
size: small mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 43.59 +- 0.30 ms
size: small mode: forward_backward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 133.95 +- 0.31 ms
size: small mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 146.79 +- 3.02 ms
size: medium mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 131.06 +- 0.30 ms
size: medium mode: forward_backward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 403.53 +- 0.30 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 440.69 +- 0.81 ms
size: large mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 274.23 +- 0.21 ms
size: large mode: forward_backward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 861.14 +- 0.31 ms
size: large mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 946.06 +- 0.26 ms
size: xl mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 834.03 +- 0.13 ms
size: xl mode: forward_backward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 2562.49 +- 0.62 ms
size: xl mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 2846.78 +- 0.19 ms
size: 10b mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 3021.63 +- 0.31 ms
Traceback (most recent call last):
  File "/home/a26473/projects/cs336_assignment2/cs336_systems/benchmark.py", line 134, in <module>
    run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode,ctx=ctx)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/cs336_systems/benchmark.py", line 60, in run_step
    logits = model(x)
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 254, in forward
    x = layer(x)
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 383, in forward
    x_attn = self.attn(self.ln1(x))
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 524, in forward
    attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 435, in scaled_dot_product_attention
    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/nn_utils.py", line 7, in softmax
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 144.00 MiB. GPU 0 has a total capacity of 79.25 GiB of which 6.88 MiB is free. Process 1322713 has 79.23 GiB memory in use. Of the allocated memory 78.42 GiB is allocated by PyTorch, and 336.78 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
Traceback (most recent call last):
  File "/home/a26473/projects/cs336_assignment2/cs336_systems/benchmark.py", line 134, in <module>
    run_step(model=model,x=x,targets=x,optimizer=optimizer,mode=mode,ctx=ctx)
    ~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/cs336_systems/benchmark.py", line 60, in run_step
    logits = model(x)
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 254, in forward
    x = layer(x)
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 383, in forward
    x_attn = self.attn(self.ln1(x))
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1779, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^^^
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/nn/modules/module.py", line 1790, in _call_impl
    return forward_call(*args, **kwargs)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 524, in forward
    attn_output = scaled_dot_product_attention(K=K, Q=Q, V=V, mask=causal_mask)
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/model.py", line 435, in scaled_dot_product_attention
    attention_weights = softmax(attention_scores, dim=-1)  # Softmax over the key dimension
  File "/home/a26473/projects/cs336_assignment2/cs336-basics/cs336_basics/nn_utils.py", line 7, in softmax
    return exponentiated_rescaled_input / torch.sum(exponentiated_rescaled_input, dim=dim, keepdim=True)
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 144.00 MiB. GPU 0 has a total capacity of 79.25 GiB of which 6.88 MiB is free. Process 1326888 has 79.23 GiB memory in use. Of the allocated memory 78.42 GiB is allocated by PyTorch, and 336.78 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://docs.pytorch.org/docs/stable/notes/cuda.html#optimizing-memory-usage-with-pytorch-cuda-alloc-conf)
```

---

## 任务 B2 — 序列长度扫描

来源 results/03_benchmark_ctx.txt — medium / full, ctx ∈ {128, 256, 512, 1024}

```
size: medium mode: full, context-length: 128, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 155.68 +- 8.46 ms
size: medium mode: full, context-length: 256, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 232.66 +- 1.65 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 440.04 +- 0.66 ms
size: medium mode: full, context-length: 1024, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 948.37 +- 1.92 ms
```

---

## 任务 B3 — warmup 消融

来源 results/04_warmup_ablation.txt — medium / full, warmup ∈ {0, 1, 2, 5}

```
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 477.76 +- 117.67 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 439.82 +- 0.74 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 439.96 +- 1.22 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 10| 439.58 +- 0.49 ms
```

---

## 任务 C — torch.compile 对比

来源 results/05_compile.txt — small / medium / large × 三种模式, 带 --compile

```
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: small mode: forward, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 37.73 +- 0.75 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: small mode: forward_backward, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 115.10 +- 1.05 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: small mode: full, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 126.23 +- 0.58 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: medium mode: forward, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 116.66 +- 0.20 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: medium mode: forward_backward, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 354.05 +- 0.28 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: medium mode: full, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 390.37 +- 0.43 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: large mode: forward, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 248.45 +- 0.18 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: large mode: forward_backward, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 771.57 +- 0.54 ms
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_dynamo/variables/functions.py:2202: UserWarning: Dynamo detected a call to a `functools.lru_cache`-wrapped function at 'einops.py:938'. Dynamo ignores the cache wrapper and directly traces the wrapped function. Silent incorrectness is only a *potential* risk, not something we have observed. Enable TORCH_LOGS=+dynamo for a DEBUG stack trace.

This call originates from:
  File "/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/einops/einops.py", line 938, in einsum
    pattern = _compactify_pattern_for_einsum(pattern)

  torch._dynamo.utils.warn_once(msg)
size: large mode: full, context-length: 512, batch-size: 4, compile: True, precision: torch.float32, steps: 10| 857.47 +- 0.49 ms
```

---

## 任务 D — 混合精度

来源 results/06_precision.txt — medium, bf16 / fp16, forward / full

```
size: medium mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.bfloat16, steps: 10| 44.19 +- 5.92 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.bfloat16, steps: 10| 174.98 +- 16.83 ms
size: medium mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float16, steps: 10| 37.96 +- 0.38 ms
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float16, steps: 10| 175.27 +- 13.35 ms
```

---

## 任务 E — 显存剖析

来源 results/07_memory.txt — large, ctx ∈ {128,256,512}, 含一组 bf16 对照

```
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_forward_seq128_batch4_fp32_compile0_steps3_20260920-141429-860812.pickle
size: large mode: forward, context-length: 128, batch-size: 4, compile: False, precision: torch.float32, steps: 3 | peak_allocated_bytes: 4044014592, peak_allocated_MiB: 3856.67, peak_reserved_bytes: 4120903680, peak_reserved_MiB: 3930.00, status: complete | timing: disabled (memory profiling)
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_full_seq128_batch4_fp32_compile0_steps3_20260920-141446-801394.pickle
size: large mode: full, context-length: 128, batch-size: 4, compile: False, precision: torch.float32, steps: 3 | peak_allocated_bytes: 16252936704, peak_allocated_MiB: 15500.01, peak_reserved_bytes: 17980981248, peak_reserved_MiB: 17148.00, status: complete | timing: disabled (memory profiling)
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_forward_seq256_batch4_fp32_compile0_steps3_20260920-141504-516545.pickle
size: large mode: forward, context-length: 256, batch-size: 4, compile: False, precision: torch.float32, steps: 3 | peak_allocated_bytes: 4114979840, peak_allocated_MiB: 3924.35, peak_reserved_bytes: 4246732800, peak_reserved_MiB: 4050.00, status: complete | timing: disabled (memory profiling)
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_full_seq256_batch4_fp32_compile0_steps3_20260920-141518-966670.pickle
size: large mode: full, context-length: 256, batch-size: 4, compile: False, precision: torch.float32, steps: 3 | peak_allocated_bytes: 19418225664, peak_allocated_MiB: 18518.66, peak_reserved_bytes: 21218983936, peak_reserved_MiB: 20236.00, status: complete | timing: disabled (memory profiling)
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_forward_seq512_batch4_fp32_compile0_steps3_20260920-141536-228823.pickle
size: large mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 3 | peak_allocated_bytes: 4398449664, peak_allocated_MiB: 4194.69, peak_reserved_bytes: 4571791360, peak_reserved_MiB: 4360.00, status: complete | timing: disabled (memory profiling)
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_full_seq512_batch4_fp32_compile0_steps3_20260920-141551-403092.pickle
size: large mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 3 | peak_allocated_bytes: 29514701824, peak_allocated_MiB: 28147.41, peak_reserved_bytes: 31306285056, peak_reserved_MiB: 29856.00, status: complete | timing: disabled (memory profiling)
Memory profiling enabled: timing is disabled; run separately without --memory-profile for latency.
Memory snapshot: /home/a26473/projects/cs336_assignment2/memory_snapshots/memory_large_full_seq512_batch4_bf16_compile0_steps3_20260920-141611-623975.pickle
size: large mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.bfloat16, steps: 3 | peak_allocated_bytes: 25741731840, peak_allocated_MiB: 24549.23, peak_reserved_bytes: 26604470272, peak_reserved_MiB: 25372.00, status: complete | timing: disabled (memory profiling)
```

---

## 任务 F — nsys 冒烟

来源 results/08_nsys_smoke.txt — small / full, 确认 nsys 能出报告

```
size: small mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 3| 158.29 +- 8.22 ms
Collecting data...
Generating '/tmp/nsys-a26473/nsys-report-4356.qdstrm'

[1/1] [0%                          ] nsys_smoke.nsys-rep
[1/1] [0%                          ] nsys_smoke.nsys-rep
[1/1] [1%                          ] nsys_smoke.nsys-rep
[1/1] [2%                          ] nsys_smoke.nsys-rep
[1/1] [5%                          ] nsys_smoke.nsys-rep
[1/1] [6%                          ] nsys_smoke.nsys-rep
[1/1] [7%                          ] nsys_smoke.nsys-rep
[1/1] [10%                         ] nsys_smoke.nsys-rep
[1/1] [11%                         ] nsys_smoke.nsys-rep
[1/1] [13%                         ] nsys_smoke.nsys-rep
[1/1] [14%                         ] nsys_smoke.nsys-rep
[1/1] [=16%                        ] nsys_smoke.nsys-rep
[1/1] [==19%                       ] nsys_smoke.nsys-rep
[1/1] [==20%                       ] nsys_smoke.nsys-rep
[1/1] [=====31%                    ] nsys_smoke.nsys-rep
[1/1] [=======================93%  ] nsys_smoke.nsys-rep
[1/1] [=======================96%  ] nsys_smoke.nsys-rep
[1/1] [========================98% ] nsys_smoke.nsys-rep
[1/1] [========================100%] nsys_smoke.nsys-rep
[1/1] [========================100%] nsys_smoke.nsys-rep
Generated:
        /home/a26473/projects/cs336_assignment2/results/nsys_smoke.nsys-rep
```

---

## 任务 F — nsys 正式采集

来源 results/09_nsys.txt — medium, forward / full

```
size: medium mode: forward, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 3| 132.67 +- 0.49 ms
Collecting data...
Generating '/tmp/nsys-a26473/nsys-report-1dba.qdstrm'

[1/1] [0%                          ] nsys_medium_forward.nsys-rep
[1/1] [0%                          ] nsys_medium_forward.nsys-rep
[1/1] [1%                          ] nsys_medium_forward.nsys-rep
[1/1] [2%                          ] nsys_medium_forward.nsys-rep
[1/1] [5%                          ] nsys_medium_forward.nsys-rep
[1/1] [6%                          ] nsys_medium_forward.nsys-rep
[1/1] [9%                          ] nsys_medium_forward.nsys-rep
[1/1] [11%                         ] nsys_medium_forward.nsys-rep
[1/1] [14%                         ] nsys_medium_forward.nsys-rep
[1/1] [=16%                        ] nsys_medium_forward.nsys-rep
[1/1] [==19%                       ] nsys_medium_forward.nsys-rep
[1/1] [==20%                       ] nsys_medium_forward.nsys-rep
[1/1] [====25%                     ] nsys_medium_forward.nsys-rep
[1/1] [=======================93%  ] nsys_medium_forward.nsys-rep
[1/1] [=======================96%  ] nsys_medium_forward.nsys-rep
[1/1] [========================98% ] nsys_medium_forward.nsys-rep
[1/1] [========================100%] nsys_medium_forward.nsys-rep
[1/1] [========================100%] nsys_medium_forward.nsys-rep
Generated:
        /home/a26473/projects/cs336_assignment2/results/nsys_medium_forward.nsys-rep
size: medium mode: full, context-length: 512, batch-size: 4, compile: False, precision: torch.float32, steps: 3| 454.24 +- 10.54 ms
Collecting data...
Generating '/tmp/nsys-a26473/nsys-report-bd54.qdstrm'

[1/1] [0%                          ] nsys_medium_full.nsys-rep
[1/1] [0%                          ] nsys_medium_full.nsys-rep
[1/1] [1%                          ] nsys_medium_full.nsys-rep
[1/1] [2%                          ] nsys_medium_full.nsys-rep
[1/1] [5%                          ] nsys_medium_full.nsys-rep
[1/1] [6%                          ] nsys_medium_full.nsys-rep
[1/1] [8%                          ] nsys_medium_full.nsys-rep
[1/1] [9%                          ] nsys_medium_full.nsys-rep
[1/1] [10%                         ] nsys_medium_full.nsys-rep
[1/1] [12%                         ] nsys_medium_full.nsys-rep
[1/1] [14%                         ] nsys_medium_full.nsys-rep
[1/1] [=15%                        ] nsys_medium_full.nsys-rep
[1/1] [=16%                        ] nsys_medium_full.nsys-rep
[1/1] [=17%                        ] nsys_medium_full.nsys-rep
[1/1] [==18%                       ] nsys_medium_full.nsys-rep
[1/1] [==19%                       ] nsys_medium_full.nsys-rep
[1/1] [==20%                       ] nsys_medium_full.nsys-rep
[1/1] [======35%                   ] nsys_medium_full.nsys-rep
[1/1] [=======36%                  ] nsys_medium_full.nsys-rep
[1/1] [=======================93%  ] nsys_medium_full.nsys-rep
[1/1] [=======================96%  ] nsys_medium_full.nsys-rep
[1/1] [========================98% ] nsys_medium_full.nsys-rep
[1/1] [========================100%] nsys_medium_full.nsys-rep
[1/1] [========================100%] nsys_medium_full.nsys-rep
Generated:
        /home/a26473/projects/cs336_assignment2/results/nsys_medium_full.nsys-rep
```

---

## 任务 F — kernel 与 NVTX 汇总

来源 results/10_nsys_kernels.txt — cuda_gpu_kern_sum + nvtx_sum, 两份 rep

```
Generating SQLite file results/nsys_medium_forward.sqlite from results/nsys_medium_forward.nsys-rep
Processing [results/nsys_medium_forward.sqlite] with [/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     38.7        264249994        720   367013.9   249790.0    139871   1158041     286038.4  ampere_sgemm_128x128_tn                                                                             
     36.5        249314712        240  1038811.3   980105.5    976986   1208792      99596.3  ampere_sgemm_128x32_tn                                                                              
      5.5         37417347        120   311811.2   293598.0    292990    362846      30536.4  ampere_sgemm_128x128_nn                                                                             
      2.0         13935741        240    58065.6    58191.5     55488     60832       2100.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      2.0         13858602       1450     9557.7     9856.0      7328     13408       1637.0  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.9         12744529        120   106204.4   101615.0    101215    119327       7759.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.8         11976794        120    99806.6    95423.5     95103    112288       7392.1  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.8         11967026          5  2393405.2  2285617.0   2285169   2825006     241272.3  ampere_sgemm_128x64_tn                                                                              
      1.6         10607038        120    88392.0    85503.5     85056     96864       4938.6  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.4          9243578        120    77029.8    76991.5     75935     78271        383.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, floa…
      1.3          8938248        120    74485.4    74431.5     73663     75808        431.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIterat…
      1.2          8103784        120    67531.5    65504.0     64352     73983       3570.4  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MaxOps<flo…
      1.0          6684598        120    55705.0    55583.5     53120     58303        890.5  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapp…
      0.8          5354977        720     7437.5     6016.0      4672     14815       3054.9  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::arr…
      0.7          4805884        240    20024.5    19008.0     18464     23807       1982.5  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>,…
      0.6          4255882        120    35465.7    35295.5     34687     36768        487.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_kernel_cuda(at::TensorIt…
      0.5          3719854        240    15499.4    16127.5     13184     18912       1737.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.3          1941202        245     7923.3     7520.0      7231      9632        816.6  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<fl…
      0.3          1808592        245     7382.0     7007.0      6592      9248        745.9  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scala…
      0.1           560702        120     4672.5     4416.0      4320      5504        468.1  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.1           540896        245     2207.7     2080.0      2016      2592        223.8  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, st…
      0.1           529404        245     2160.8     2048.0      2015      2560        219.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIter…
      0.0           247709        120     2064.2     1984.0      1856      2528        180.1  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<c…
      0.0           225757        120     1881.3     1760.0      1759      2208        189.0  void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &,…
      0.0            65982          5    13196.4    12415.0     12352     15072       1215.3  void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon…
      0.0             3359          1     3359.0     3359.0      3359      3359          0.0  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<unsigned int, (int)4, void …


NOTICE: Existing SQLite export found: results/nsys_medium_forward.sqlite
        It is assumed file was previously exported from: results/nsys_medium_forward.nsys-rep
        Consider using --force-export=true if needed.

Processing [results/nsys_medium_forward.sqlite] with [/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances   Avg (ns)     Med (ns)    Min (ns)   Max (ns)   StdDev (ns)   Style         Range       
 --------  ---------------  ---------  -----------  -----------  ---------  ---------  -----------  -------  ------------------
     24.3        679707706          5  135941541.2   60486563.0   43449553  377178721  141438646.8  PushPop  :forward          
     22.4        626311575          1  626311575.0  626311575.0  626311575  626311575          0.0  PushPop  :warmup           
     18.2        509576958          2  254788479.0  254788479.0  132378819  377198139  173113401.3  PushPop  :step_1           
     14.2        398626796          1  398626796.0  398626796.0  398626796  398626796          0.0  PushPop  :measure          
     10.1        282088533          2  141044266.5  141044266.5  133329981  148758552   10909647.2  PushPop  :step_2           
      4.7        132675297          1  132675297.0  132675297.0  132675297  132675297          0.0  PushPop  :step_3           
      3.4         96007127        120     800059.4     211908.5     135499   71092557    6471077.2  PushPop  :attention/softmax
      1.6         44892080        120     374100.7     125829.5      85224   15860988    1461074.2  PushPop  :attention/pv     
      1.1         31570744        120     263089.5     132654.0      91134   14247960    1288467.7  PushPop  :attention/qk     

Generating SQLite file results/nsys_medium_full.sqlite from results/nsys_medium_full.nsys-rep
Processing [results/nsys_medium_full.sqlite] with [/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/reports/cuda_gpu_kern_sum.py]... 

 ** CUDA GPU Kernel Summary (cuda_gpu_kern_sum):

 Time (%)  Total Time (ns)  Instances  Avg (ns)   Med (ns)   Min (ns)  Max (ns)  StdDev (ns)                                                  Name                                                
 --------  ---------------  ---------  ---------  ---------  --------  --------  -----------  ----------------------------------------------------------------------------------------------------
     15.7        343949793        720   477708.0   244606.0    244031    953626     329965.0  ampere_sgemm_128x64_nn                                                                              
     15.5        340050393        360   944584.4   948186.0    935162    957689       5948.6  ampere_sgemm_64x32_sliced1x4_nt                                                                     
     12.6        276492110        840   329157.3   249695.0    139807   1158296     270380.9  ampere_sgemm_128x128_tn                                                                             
     11.2        244755430        240  1019814.3   980057.0    977465   1210424      86612.8  ampere_sgemm_128x32_tn                                                                              
      5.3        116756438        480   243242.6   243166.0    241950    246142        584.5  ampere_sgemm_128x64_nt                                                                              
      5.1        111593665        120   929947.2   929417.5    928505    939034       1880.3  ampere_sgemm_128x32_nn                                                                              
      4.2         92874979       6675    13913.9     5408.0      1440    144671      19671.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, std::arr…
      3.6         79719500       1940    41092.5    16512.0      1952    120607      36801.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      3.3         71911873        240   299632.8   293246.0    292798    363742      19997.7  ampere_sgemm_128x128_nn                                                                             
      3.2         69534643        240   289727.7   289790.0    288990    292894        532.0  ampere_sgemm_128x128_nt                                                                             
      3.0         65942466       7060     9340.3     6304.0      1408     54432       8358.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, floa…
      2.2         49253950        480   102612.4   103904.0     92224    121087       5473.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.9         41413454       1805    22943.7    15647.0      9376    133887      17886.9  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.4         30348549       2905    10447.0    10047.0      6336    103520       4620.9  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.1         23789253        620    38369.8    55456.0      9120     70751      23251.9  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::func_wrapp…
      1.1         23301158        240    97088.2    95567.5     94815    112896       4841.5  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      1.0         21707760        250    86831.0    85471.5     83647    104256       4835.4  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.9         20034745        260    77056.7    75855.5     73311    100288       5740.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::neg_kernel_cuda(at::TensorIterat…
      0.8         18512551        240    77135.6    77007.5     75040     81120       1112.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, floa…
      0.8         17246260       2570     6710.6     3872.0      1216     46495       7727.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, std::array<c…
      0.6         13662242       1100    12420.2     4704.0      1727     79232      13142.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, floa…
      0.6         12464638       1340     9302.0     4384.0      1408     50239       8606.0  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scala…
      0.5         11672724          5  2334544.8  2334576.0   2334353   2334705        133.8  ampere_sgemm_128x32_sliced1x4_nn                                                                    
      0.5         11528469          5  2305693.8  2305746.0   2305105   2306065        371.4  void cutlass::Kernel2<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params)                       
      0.5         11426643          5  2285328.6  2285489.0   2284816   2285809        428.1  ampere_sgemm_128x64_tn                                                                              
      0.5         10995426       1095    10041.5     4256.0      1664     52512       9820.0  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sqrt_kernel_cuda(at::TensorItera…
      0.4          9361443        125    74891.5    74207.0     73407     90912       3244.5  void at::native::vectorized_elementwise_kernel<(int)4, at::native::exp_kernel_cuda(at::TensorIterat…
      0.4          8413834        125    67310.7    65632.0     64704     75136       3365.3  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MaxOps<flo…
      0.4          8060015       1340     6014.9     4032.0      1408     51903       6043.2  void at::native::vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, st…
      0.3          7306349        360    20295.4    19136.0     18464     24640       1693.5  void at::native::<unnamed>::CatArrayBatchedCopy<at::native::<unnamed>::OpaqueType<(unsigned int)4>,…
      0.3          7272045        120    60600.4    60544.0     57695     62304        594.6  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_backward_kernel_cuda(at:…
      0.2          4250690        120    35422.4    35296.0     34752     37568        488.4  void at::native::vectorized_elementwise_kernel<(int)4, at::native::sigmoid_kernel_cuda(at::TensorIt…
      0.1          3103086        245    12665.7    12672.0     11776     13183        153.5  void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapp…
      0.1          2193872        250     8775.5     8896.0      2656      9088        863.0  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.1          2096048        240     8733.5     8624.0      8128      9408        396.2  void at::native::elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::n…
      0.1          1926130        250     7704.5     7488.0      3872      9632        882.7  void at::native::reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<fl…
      0.1          1718713        125    13749.7    13792.0     10336     14752        516.8  void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)8, void at::native::_cuda_scatte…
      0.0           964664        120     8038.9     8064.0      7296      8608        254.7  void at::native::<unnamed>::CatArrayBatchedCopy_vectorized<at::native::<unnamed>::OpaqueType<(unsig…
      0.0           553950        120     4616.3     4416.0      4320      5600        440.2  void at::native::elementwise_kernel<(int)128, (int)4, void at::native::gpu_kernel_impl_nocast<at::n…
      0.0           524447        245     2140.6     2144.0      1856      2464        139.0  void at::native::vectorized_elementwise_kernel<(int)4, void at::native::<unnamed>::pow_tensor_scala…
      0.0           518173        245     2115.0     2016.0      1952      2528        190.3  void at::native::vectorized_elementwise_kernel<(int)4, at::native::rsqrt_kernel_cuda(at::TensorIter…
      0.0           234078        125     1872.6     1792.0      1759      2240        169.5  void <unnamed>::elementwise_kernel_with_index<int, at::native::arange_cuda_out(const c10::Scalar &,…
      0.0            82112          5    16422.4    16544.0     15936     16640        283.9  void <unnamed>::indexing_backward_kernel<float, (int)4>(const long *, const long *, const T1 *, T1 …
      0.0            65216          5    13043.2    12608.0     12544     14656        908.7  void at::native::vectorized_gather_kernel<(int)16, long>(char *, char *, T2 *, int, long, long, lon…
      0.0            64768          5    12953.6    13088.0     11360     14400       1152.5  void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)8, void at::native::_cuda_scatte…
      0.0            57472          5    11494.4    11840.0     10144     12256        824.4  void at::native::_scatter_gather_elementwise_kernel<(int)128, (int)8, void at::native::_cuda_scatte…
      0.0            45599          5     9119.8     9120.0      8991      9216         99.0  void at_cuda_detail::cub::detail::radix_sort::DeviceRadixSortSingleTileKernel<at_cuda_detail::cub::…
      0.0            18304          5     3660.8     3680.0      3488      3744        100.2  void at::native::vectorized_elementwise_kernel<(int)2, at::native::BUnaryFunctor<long, long, long, …
      0.0            15072          5     3014.4     3008.0      2976      3040         26.8  void at::native::vectorized_elementwise_kernel<(int)2, at::native::BUnaryFunctor<long, long, long, …
      0.0            13504          5     2700.8     2688.0      2592      2784         73.7  void at::native::vectorized_elementwise_kernel<(int)4, at::native::log_kernel_cuda(at::TensorIterat…
      0.0            11584          5     2316.8     2272.0      2240      2496        107.6  void at::native::vectorized_elementwise_kernel<(int)2, at::native::AUnaryFunctor<long, long, long, …
      0.0             3424          1     3424.0     3424.0      3424      3424          0.0  void at::native::<unnamed>::distribution_elementwise_grid_stride_kernel<unsigned int, (int)4, void …


NOTICE: Existing SQLite export found: results/nsys_medium_full.sqlite
        It is assumed file was previously exported from: results/nsys_medium_full.nsys-rep
        Consider using --force-export=true if needed.

Processing [results/nsys_medium_full.sqlite] with [/opt/nvidia/nsight-systems/2026.3.2/target-linux-x64/reports/nvtx_sum.py]... 

 ** NVTX Range Summary (nvtx_sum):

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)   Style             Range          
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  -------  -------------------------
     16.8       1440559866          1  1440559866.0  1440559866.0  1440559866  1440559866          0.0  PushPop  :warmup                  
     16.8       1440383366          2   720191683.0   720191683.0   444336078   996047288  390118737.8  PushPop  :step_1                  
     16.1       1387705025          5   277541005.0   264595537.0   216925362   387925171   64923311.1  PushPop  :backward                
     15.9       1362911371          1  1362911371.0  1362911371.0  1362911371  1362911371          0.0  PushPop  :measure                 
     10.6        908091674          2   454045837.0   454045837.0   442772794   465318880   15942490.3  PushPop  :step_2                  
      8.9        765314595          5   153062919.0    62321623.0    42890877   520223740  206244292.0  PushPop  :forward                 
      7.4        637764745          5   127552949.0   131131949.0    86565522   151277236   24400663.2  PushPop  :optimizer               
      5.3        453181695          1   453181695.0   453181695.0   453181695   453181695          0.0  PushPop  :step_3                  
      1.4        119004588        120      991704.9      181196.5      144251    76009924    6913370.0  PushPop  :attention/softmax       
      0.6         48007195        120      400060.0      137470.5      102434    12641511    1166914.3  PushPop  :attention/qk            
      0.4         32773935        120      273116.1      127756.5       95518    13156313    1201108.5  PushPop  :attention/pv            
      0.0           282837          5       56567.4       45939.0       42559      102931      25973.1  PushPop  CCCL:cub::DeviceRadixSort
```

---

## 任务 G1 — FlashAttention 冒烟

来源 results/11_flash_quick.txt — 两组配置: (128,16,fp32) 和 (65536,16,fp32)

```
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/autograd/graph.py:869: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:335.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
     seq_len            d        dtype   tile_q_k_d attention_fwd_ms attention_bwd_ms attention_full_ms triton_fwd_ms triton_bwd_ms triton_full_ms
         128           16      float32     16x16x16           0.2724           0.5355            1.1032        0.0267        0.2115         0.4922
       65536           16      float32     16x16x16              OOM              OOM               OOM       33.7774      120.2999       153.4395
Saved 2 configurations to flash_benchmark.csv
```

---

## 任务 G2 — FlashAttention 完整扫描日志

来源 results/12_flash_full.txt — 10 序列长度 × 4 维度 × 2 精度 = 80 组

```
     seq_len            d        dtype   tile_q_k_d attention_fwd_ms attention_bwd_ms attention_full_ms triton_fwd_ms triton_bwd_ms triton_full_ms
         128           16     bfloat16     16x16x16           0.2165           0.5943            1.0387        0.0141        0.2830         0.3240
         128           16      float32     16x16x16           0.3093           0.4372            0.8925        0.0168        0.1958         0.4051
         128           32     bfloat16     16x16x32           0.2019           0.3761            1.0207        0.0129        0.3181         0.4335
         128           32      float32     16x16x32           0.2099           0.4037            0.8796        0.0291        0.2548         0.3482
         128           64     bfloat16     16x16x64           0.2094           0.5031            0.8748        0.0127        0.3425         0.5033
         128           64      float32     16x16x64           0.2521           0.3046            0.8423        0.0261        0.2568         0.5210
         128          128     bfloat16    16x16x128           0.2208           0.3734            0.8435        0.0146        0.3469         0.4845
         128          128      float32    16x16x128           0.2861           0.3724            0.7584        0.0422        0.3126         0.3172
         256           16     bfloat16     16x16x16           0.2640           0.4552            0.9938        0.0163        0.2521         0.4973
         256           16      float32     16x16x16           0.3171           0.4638            0.9903        0.0218        0.3949         0.4118
         256           32     bfloat16     16x16x32           0.2478           0.4295            0.8631        0.0188        0.3723         0.3077
         256           32      float32     16x16x32           0.2190           0.3908            0.8127        0.0293        0.3313         0.4801
         256           64     bfloat16     16x16x64           0.2717           0.5205            0.9714        0.0181        0.2882         0.4981
         256           64      float32     16x16x64           0.2256           0.4805            0.9208        0.0447        0.2075         0.4299
         256          128     bfloat16    16x16x128           0.2754           0.3233            0.9217        0.0204        0.1662         0.2685
         256          128      float32    16x16x128           0.2212           0.5185            1.0016        0.0759        0.2697         0.4635
         512           16     bfloat16     16x16x16           0.2903           0.4301            0.9696        0.0240        0.2590         0.3648
         512           16      float32     16x16x16           0.2334           0.3922            0.8373        0.0321        0.2954         0.4171
         512           32     bfloat16     16x16x32           0.2548           0.5088            0.9069        0.0283        0.3282         0.4391
         512           32      float32     16x16x32           0.2490           0.3947            0.8384        0.0520        0.2926         0.4137
         512           64     bfloat16     16x16x64           0.2278           0.4075            0.8673        0.0289        0.3405         0.4169
         512           64      float32     16x16x64           0.2438           0.3313            1.3801        0.0824        0.2746         0.5424
         512          128     bfloat16    16x16x128           0.2468           0.4636            0.9851        0.0335        0.3479         0.4234
         512          128      float32    16x16x128           0.2110           0.4174            0.6351        0.1428        0.1719         0.2670
        1024           16     bfloat16     16x16x16           0.1522           0.4807            1.1912        0.0412        0.3831         0.3908
        1024           16      float32     16x16x16           0.2879           0.5464            1.0299        0.0560        0.3282         0.6156
        1024           32     bfloat16     16x16x32           0.3106           0.5142            0.9182        0.0490        0.4265         0.5005
        1024           32      float32     16x16x32           0.2417           0.4106            0.8291        0.0955        0.3214         0.5239
        1024           64     bfloat16     16x16x64           0.2272           0.3724            0.8781        0.0505        0.1904         0.4325
        1024           64      float32     16x16x64           0.2171           0.4399            0.8820        0.1576        0.3482         0.5344
        1024          128     bfloat16    16x16x128           0.2259           0.3469            0.7984        0.0603        0.2541         0.4453
        1024          128      float32    16x16x128           0.2581           0.5045            0.9613        0.2778        0.2072         0.5235
        2048           16     bfloat16     16x16x16           0.2183           0.4460            0.9393        0.0756        0.2890         0.4335
        2048           16      float32     16x16x16           0.2689           0.5524            0.7778        0.1096        0.2198         0.3249
        2048           32     bfloat16     16x16x32           0.1615           0.2746            0.8845        0.0914        0.3752         0.2985
        2048           32      float32     16x16x32           0.2876           0.4894            1.0442        0.2013        0.2661         0.4353
        2048           64     bfloat16     16x16x64           0.2351           0.3693            0.8967        0.0966        0.3164         0.4230
        2048           64      float32     16x16x64           0.2389           0.4925            1.0991        0.3401        0.3129         0.6412
        2048          128     bfloat16    16x16x128           0.2452           0.4570            0.9425        0.1162        0.2888         0.5162
        2048          128      float32    16x16x128           0.3188           0.6491            1.0602        0.6142        0.4914         1.1016
        4096           16     bfloat16     16x16x16           0.4635           0.9877            1.4286        0.1602        0.4113         0.5609
        4096           16      float32     16x16x16           0.7109           1.5466            2.2483        0.2612        0.6212         0.8766
        4096           32     bfloat16     16x16x32           0.4628           0.9567            1.3971        0.1877        0.4440         0.5944
        4096           32      float32     16x16x32           0.7379           1.5690            2.2932        0.5490        0.6738         1.2182
        4096           64     bfloat16     16x16x64           0.4711           0.9698            1.4202        0.2034        0.4211         0.6175
        4096           64      float32     16x16x64           0.8449           1.8081            2.6426        0.9652        0.9855         1.9450
        4096          128     bfloat16    16x16x128           0.4885           0.9994            1.4705        0.2539        0.4989         0.7249
        4096          128      float32    16x16x128           1.0732           2.3311            3.4003        1.7934        1.6263         3.4146
        8192           16     bfloat16     16x16x16           1.7011           3.4628            5.1261        0.3873        1.4005         1.7839
        8192           16      float32     16x16x16           2.5704           5.7622            8.3284        0.7402        2.2704         3.0076
        8192           32     bfloat16     16x16x32           1.7018           3.4297            5.0985        0.4374        1.4073         1.8408
        8192           32      float32     16x16x32           2.6765           5.7384            8.4040        1.7615        2.4807         4.2370
        8192           64     bfloat16     16x16x64           1.7432           3.4620            5.1771        0.4924        1.5073         1.9954
        8192           64      float32     16x16x64           3.1217           6.8581            9.9762        3.1337        3.7750         6.8991
        8192          128     bfloat16    16x16x128           1.7915           3.5352            5.2970        0.6436        1.6328         2.2746
        8192          128      float32    16x16x128           4.0635           8.6363           12.6948        6.3309        6.0667        12.3940
       16384           16     bfloat16     16x16x16           6.3497          13.1706           19.5190        1.0848        5.5322         6.6014
       16384           16      float32     16x16x16           9.8312          22.0792           31.9093        2.5938        9.1310        11.6073
       16384           32     bfloat16     16x16x32           6.4236          13.1354           19.4717        1.3336        5.4861         6.8072
       16384           32      float32     16x16x32          10.2126          22.6460           32.8657        6.3802        9.8947        16.1655
       16384           64     bfloat16     16x16x64           6.5164          13.2429           19.7080        1.5426        5.7304         7.2591
       16384           64      float32     16x16x64          12.2667          26.4639           38.7724       11.4756       14.5551        26.0061
       16384          128     bfloat16    16x16x128           6.6872          13.4524           20.1210        2.0913        6.1296         8.2386
       16384          128      float32    16x16x128          15.9361          33.8924           49.8077       22.0262       23.8589        45.8648
       32768           16     bfloat16     16x16x16          24.9868          52.8052           77.7951        3.6994       21.5686        25.2466
       32768           16      float32     16x16x16          39.4783          87.9300          127.4157        8.9892       36.6850        45.0999
       32768           32     bfloat16     16x16x32          25.5090          52.0594           77.0635        4.4509       21.6162        26.0577
       32768           32      float32     16x16x32          40.9567          89.7729          130.7041       22.5449       40.1644        61.8350
       32768           64     bfloat16     16x16x64          25.9389          52.5728           78.0089        5.1813       22.5228        27.7295
       32768           64      float32     16x16x64          47.9492         104.0436          152.2513       40.4639       57.6551        97.6355
       32768          128     bfloat16    16x16x128          26.5820          53.3141           79.4739        7.1043       24.4590        31.4669
       32768          128      float32    16x16x128          62.0894         132.4214          194.5049       78.6945       93.2642       171.9147
       65536           16     bfloat16     16x16x16          99.8666         207.9901          307.8762       13.1775       92.3919       107.1703
       65536           16      float32     16x16x16              OOM              OOM               OOM       33.7678           OOM            OOM
       65536           32     bfloat16     16x16x32         101.8324         208.2048          308.1200       16.2345       94.0848       110.3473
       65536           32      float32     16x16x32              OOM              OOM               OOM       86.7124           OOM            OOM
       65536           64     bfloat16     16x16x64         103.4907         209.8878          311.3236       19.0281       97.2985       114.6689
       65536           64      float32     16x16x64              OOM              OOM               OOM      155.1774           OOM            OOM
       65536          128     bfloat16    16x16x128         106.2378         214.9245          321.3840       26.3975      106.3672       133.3189
       65536          128      float32    16x16x128              OOM              OOM               OOM      296.7409           OOM            OOM
Saved 80 configurations to results/flash_benchmark.csv
```

---

## 任务 G2 — Dynamo 重编译日志

来源 results/12_recompiles.log — TORCH_LOGS=recompiles 的 stderr

```
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/autograd/graph.py:869: UserWarning: Attempting to run cuBLAS, but there was no current CUDA context! Attempting to set the primary context... (Triggered internally at /pytorch/aten/src/ATen/cuda/CublasHandlePool.cpp:335.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
V0920 13:54:44.535000 3904 torch/_dynamo/guards.py:4760] [0/1] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:54:44.535000 3904 torch/_dynamo/guards.py:4760] [0/1] [__recompiles]     triggered by the following guard failure(s):
V0920 13:54:44.535000 3904 torch/_dynamo/guards.py:4760] [0/1] [__recompiles]     - 0/0: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:54:46.150000 3904 torch/_dynamo/guards.py:4760] [0/2] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:54:46.150000 3904 torch/_dynamo/guards.py:4760] [0/2] [__recompiles]     triggered by the following guard failure(s):
V0920 13:54:46.150000 3904 torch/_dynamo/guards.py:4760] [0/2] [__recompiles]     - 0/1: tensor 'O' dtype mismatch. expected Float, actual BFloat16
V0920 13:54:46.150000 3904 torch/_dynamo/guards.py:4760] [0/2] [__recompiles]     - 0/0: tensor 'O' size mismatch at index 2. expected 16, actual 32
V0920 13:54:49.149000 3904 torch/_dynamo/guards.py:4760] [0/3] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:54:49.149000 3904 torch/_dynamo/guards.py:4760] [0/3] [__recompiles]     triggered by the following guard failure(s):
V0920 13:54:49.149000 3904 torch/_dynamo/guards.py:4760] [0/3] [__recompiles]     - 0/2: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:54:49.149000 3904 torch/_dynamo/guards.py:4760] [0/3] [__recompiles]     - 0/1: tensor 'O' size mismatch at index 2. expected 16, actual 32
V0920 13:54:49.149000 3904 torch/_dynamo/guards.py:4760] [0/3] [__recompiles]     - 0/0: tensor 'O' dtype mismatch. expected BFloat16, actual Float
/home/a26473/projects/cs336_assignment2/.venv/lib/python3.13/site-packages/torch/_inductor/compile_fx.py:322: UserWarning: TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled. Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
  warnings.warn(
V0920 13:54:56.179000 3904 torch/_dynamo/guards.py:4760] [0/4] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:54:56.179000 3904 torch/_dynamo/guards.py:4760] [0/4] [__recompiles]     triggered by the following guard failure(s):
V0920 13:54:56.179000 3904 torch/_dynamo/guards.py:4760] [0/4] [__recompiles]     - 0/3: tensor 'O' dtype mismatch. expected Float, actual BFloat16
V0920 13:54:56.179000 3904 torch/_dynamo/guards.py:4760] [0/4] [__recompiles]     - 0/2: tensor 'O' size mismatch at index 1. expected 128, actual 256
V0920 13:54:56.179000 3904 torch/_dynamo/guards.py:4760] [0/4] [__recompiles]     - 0/1: tensor 'O' dtype mismatch. expected Float, actual BFloat16
V0920 13:54:56.179000 3904 torch/_dynamo/guards.py:4760] [0/4] [__recompiles]     - 0/0: tensor 'O' size mismatch at index 1. expected 128, actual 256
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles]     triggered by the following guard failure(s):
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles]     - 0/4: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles]     - 0/3: tensor 'O' size mismatch at index 1. expected 128, actual 256
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles]     - 0/2: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles]     - 0/1: tensor 'O' size mismatch at index 1. expected 128, actual 256
V0920 13:55:00.706000 3904 torch/_dynamo/guards.py:4760] [0/5] [__recompiles]     - 0/0: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     triggered by the following guard failure(s):
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     - 0/5: tensor 'O' dtype mismatch. expected Float, actual BFloat16
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     - 0/4: O.size()[1]*k.size()[1] <= 2147483647  # (_inductor/codegen/simd.py:1877 in can_use_32bit_indexing)
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     - 0/3: tensor 'O' dtype mismatch. expected Float, actual BFloat16
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     - 0/2: tensor 'O' size mismatch at index 1. expected 128, actual 65536
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     - 0/1: tensor 'O' dtype mismatch. expected Float, actual BFloat16
V0920 13:56:54.779000 3904 torch/_dynamo/guards.py:4760] [0/6] [__recompiles]     - 0/0: tensor 'O' size mismatch at index 1. expected 128, actual 65536
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles] Recompiling function _backward in /home/a26473/projects/cs336_assignment2/cs336_systems/flash_attention.py:4
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     triggered by the following guard failure(s):
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/6: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/5: O.size()[1]*k.size()[1] <= 2147483647  # (_inductor/codegen/simd.py:1877 in can_use_32bit_indexing)
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/4: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/3: tensor 'O' size mismatch at index 1. expected 128, actual 65536
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/2: tensor 'O' dtype mismatch. expected BFloat16, actual Float
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/1: tensor 'O' size mismatch at index 1. expected 128, actual 65536
V0920 13:57:06.082000 3904 torch/_dynamo/guards.py:4760] [0/7] [__recompiles]     - 0/0: tensor 'O' dtype mismatch. expected BFloat16, actual Float
```

---

## 无法内嵌的二进制产物

| 路径 | 大小 | 说明 |
|---|---|---|
| `results/nsys_medium_forward.nsys-rep` | 5.5 MB | nsys 报告，用 Nsight Systems GUI 打开 |
| `results/nsys_medium_full.nsys-rep` | 8.1 MB | nsys 报告，用 Nsight Systems GUI 打开 |
| `results/nsys_smoke.nsys-rep` | 5.7 MB | nsys 报告，用 Nsight Systems GUI 打开 |
| `memory_snapshots/` | 73.0 MB | 7 个显存快照 pickle |

显存快照需在 <https://pytorch.org/memory_viz> 上分析。文件名编码了完整配置：

```
memory_large_forward_seq128_batch4_fp32_compile0_steps3_20260920-141429-860812.pickle
memory_large_forward_seq256_batch4_fp32_compile0_steps3_20260920-141504-516545.pickle
memory_large_forward_seq512_batch4_fp32_compile0_steps3_20260920-141536-228823.pickle
memory_large_full_seq128_batch4_fp32_compile0_steps3_20260920-141446-801394.pickle
memory_large_full_seq256_batch4_fp32_compile0_steps3_20260920-141518-966670.pickle
memory_large_full_seq512_batch4_bf16_compile0_steps3_20260920-141611-623975.pickle
memory_large_full_seq512_batch4_fp32_compile0_steps3_20260920-141551-403092.pickle
```

---

## 复现

```bash
bash setup.sh      # 装 uv + nsys，克隆仓库，uv sync
bash run_all.sh    # 按 A -> G -> B -> E -> F -> C -> D 顺序跑全部任务
bash run_all.sh G D  # 也可只跑指定任务
```

需要一台显存 80GB、驱动 ≥ 580（CUDA 13）的机器。脚本在 `/mnt/d/test/cs336_run/`。

---

<sub>本文件由 make_report.py 于 2026-09-21 11:08 生成，格式于事后修复。</sub>
