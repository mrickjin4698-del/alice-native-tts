# R1.3 首包延迟评估

测量日期：2026-10-26。基于 R1.2 的 CUDA GPT、GPU 常驻 K/V Cache 与 CUDA EP VITS。模型、参考音频均为本机资源，没有加入仓库。

## 结论

- 中文长句冷路径 Frontend 中位数为 137.3 ms，其中 G2PW 约 86.2 ms、BERT 约 50.4 ms；其他前端步骤合计不到 1 ms。英文 Frontend 低于 0.1 ms。
- 对相同归一化文本缓存 Frontend 输出后，缓存命中的测量样本 Frontend 降至 0.02–0.04 ms。中文长句 TTFA P50 从 346.8 ms 降至 203.5 ms，完整耗时 P50 从 1,425.7 ms 降至 1,265.0 ms。缓存只改善同一模型实例中重复出现的精确文本，冷文本仍需要完整 G2PW/BERT 推理。
- 首包阈值 12/16/20 均降低四个语料的 TTFA，P50 降幅为 5.8%–50.1%。完整合成时间没有稳定下降；边界跳变在部分语料上明显高于阈值 24。没有完成人工试听，因此不满足切换默认值的质量门槛，默认仍为 24。
- G2PW batch 8 只在中文长句上试验。G2PW P50 减少约 3.5 ms，但 TTFA 和 P95 没有改善，默认仍为 4。
- CPU E2E 回归通过。DirectML 在当前 ORT 1.30 runtime 下因 DirectML EP 不可用而 BLOCKED。

## 运行时和方法

| 项目 | 实测配置 |
|---|---|
| 系统 / GPU | Windows x64 / RTX 5070 Ti，SM 12.0，16 GiB |
| CPU / 驱动 | Intel Core Ultra 9 285K / 610.74 |
| ORT / CUDA / cuDNN | ONNX Runtime 1.30.0 / CUDA 13.0 / cuDNN 9.12 |
| Rust ORT wrapper | `ort` 2.0.0-rc.12，动态加载 |
| 模型 | 本机 ABI-1 GPT-SoVITS v2Pro 流式包 |
| Backend | CUDA GPT + CUDA EP VITS，显式允许已审计的 CPU helper nodes |
| Sampling | seed 1234；模型配置与权重保持不变 |
| Chunk policy | 首 Chunk 12/16/20/24；其后固定 24 |
| 样本数 | 每个文本/阈值 30 次预热、20 次计时 |

CUDA VITS Session 仍包含 570 个 ORT CPU 辅助/形状节点；主要 Gemm/MatMul/Conv/Attention 计算节点在 CUDA EP。GPT Step 的 CPU fallback 关闭，K/V Cache 仍由 CUDA 设备缓冲区乒乓更新。详细节点分配见 [R1.2 报告](performance-r1.2.md)。热态测试未启用 ORT operator trace；传输节点数据来自单独的诊断 trace，避免 trace 开销污染延迟。

TTFA 从请求进入到第一次音频 callback 开始计时；RTF 为完整合成墙钟时间除以生成音频时长。报告中的 P50/P95 使用 20 个计时样本的排序分位数。

## 首包阈值矩阵

每格为 P50 / P95。单位：TTFA、完整耗时为 ms；RTF 无单位。测试句为 `你好。`、`今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。`、`Hello Alice, welcome back.` 和 `The quick brown fox jumps over the lazy dog while Alice reads a book and welcomes everyone home.`。

| 语料 | 首包 token | TTFA | 完整耗时 | RTF |
|---|---:|---:|---:|---:|
| zh_short | 12 | 174.8 / 179.5 | 539.3 / 550.4 | 0.265 / 0.271 |
| zh_short | 16 | 191.2 / 199.7 | 529.6 / 543.8 | 0.261 / 0.267 |
| zh_short | 20 | 217.1 / 229.4 | 530.7 / 541.4 | 0.260 / 0.265 |
| zh_short | 24 | 238.7 / 247.7 | 529.3 / 542.0 | 0.260 / 0.266 |
| zh_long | 12 | 277.5 / 295.1 | 1,460.8 / 1,487.0 | 0.220 / 0.223 |
| zh_long | 16 | 305.0 / 329.6 | 1,465.6 / 1,616.9 | 0.220 / 0.243 |
| zh_long | 20 | 326.8 / 336.5 | 1,461.3 / 1,497.6 | 0.219 / 0.225 |
| zh_long | 24 | 346.8 / 356.7 | 1,425.7 / 1,448.8 | 0.214 / 0.218 |
| en_short | 12 | 102.6 / 105.3 | 560.0 / 574.7 | 0.219 / 0.225 |
| en_short | 16 | 161.0 / 166.6 | 522.8 / 537.9 | 0.205 / 0.211 |
| en_short | 20 | 182.2 / 184.8 | 559.5 / 578.8 | 0.219 / 0.226 |
| en_short | 24 | 205.7 / 209.6 | 564.1 / 580.7 | 0.220 / 0.227 |
| en_long | 12 | 138.9 / 142.7 | 1,188.0 / 1,221.0 | 0.195 / 0.200 |
| en_long | 16 | 160.6 / 164.2 | 1,198.0 / 1,210.2 | 0.196 / 0.198 |
| en_long | 20 | 187.4 / 191.4 | 1,207.7 / 1,224.3 | 0.198 / 0.201 |
| en_long | 24 | 203.7 / 241.6 | 1,202.6 / 1,374.5 | 0.197 / 0.225 |

阈值 12 在四个语料上都给出最低 TTFA，但在中文长句完整合成 P50 比阈值 24 慢 35.1 ms；各语料完整耗时并未一致改善。阈值 16 的 `zh_long`、`en_long` P95 出现明显高尾延迟。

## 首包时间线与冷 Frontend 瓶颈

Chunk 24 冷路径 P50/P95 阶段耗时：

| 语料 | Frontend | GPT Prefill | GPT Step 首 token | 首次 VITS | 首次 callback / TTFA |
|---|---:|---:|---:|---:|---:|
| zh_short | 28.8 / 30.0 | 6.7 / 10.7 | 5.6 / 7.7 | 69.6 / 73.2 | 238.7 / 247.7 |
| zh_long | 137.3 / 140.3 | 6.9 / 7.5 | 5.6 / 7.9 | 69.3 / 74.4 | 346.8 / 356.7 |
| en_short | 0.03 / 0.03 | 6.7 / 7.2 | 5.6 / 6.2 | 68.9 / 71.5 | 205.7 / 209.6 |
| en_long | 0.07 / 0.08 | 7.0 / 7.9 | 5.5 / 6.6 | 68.0 / 75.3 | 203.7 / 241.6 |

中文长句 Frontend breakdown P50/P95 为：G2PW 执行 86.2 / 88.8 ms，BERT 执行 50.4 / 52.2 ms；分词、张量准备、音素映射、特征扩展和拼接均低于 0.1 ms。图像外的请求时间线 JSON 为每个请求记录 Frontend、GPT Prefill、逐 token `gpt_step_run`、首次 VITS、SOLA 和 audio callback 的 request-relative 时间戳。GPT Step 首包逐 token P50 大多为 5.2–5.4 ms；具体每个 token 的 P50/P95 保存在原始 JSON 中。

首次 Chunk 没有前序 latent，因此首次 SOLA P50 约 0.001 ms；完整流的 SOLA 总耗时随 Chunk 数量增加，在长句约 3.8 ms。首次 CUDA VITS 的主要耗时为模型执行，P50 约 68–69 ms；同步约 0.016 ms，Host copy 和输出验证各低于 0.04 ms。

### ORT 内部传输诊断

单独启用 ORT trace 的诊断请求记录到：

| Session | 传输事件 | 事件数 | trace 声明输出字节 | 总时长 | 单事件 P50 / P95 |
|---|---|---:|---:|---:|---:|
| GPT Encoder | CUDA `MemcpyFromHost` | 24 | 96 B | 0.287 ms | 11 / 16 µs |
| GPT Step | CUDA `MemcpyToHost` | 167 | 1,336 B | 2.860 ms | 14 / 20 µs |
| VITS | CUDA `MemcpyFromHost` | 98 | 372,330 B | 1.474 ms | 15 / 20 µs |
| VITS | CUDA `MemcpyToHost` | 14 | 186,039 B | 0.265 ms | 16 / 53 µs |

GPT Step trace 中的小量 `MemcpyToHost` 对应采样所需的小输出；K/V Cache 输入输出保持 CUDA 设备内存，未逐 token 回传 CPU。上述字节数是 ORT profile event 的输出 payload 记录，不是 PCIe 流量计数器测得的物理链路流量。诊断 trace 文件在 `target/r1.3-first-timeline/`。

## Frontend 缓存与 G2PW 批次

Frontend 新增每个模型实例独立的精确归一化文本缓存，保存 phone 与 BERT feature 数组；LRU 上限为 32 项、数组 payload 上限 8 MiB，默认启用，可用 `ALICE_NATIVE_TTS_FRONTEND_CACHE=0` 关闭。缓存命中复制并返回已保存的数组，保留每次命中时的取消检查。缓存冷 miss 仍运行完整前端并在返回前存储结果；这部分插入成本没有单独做 30/20 对照。该上限是张量 payload 上限，不含 HashMap/字符串等容器开销。

相同 GPU、seed、模型和 Chunk 24 下，缓存开启后的 20 个测量样本都命中已由 30 次预热填入的条目：

| 语料 | 未缓存 TTFA P50 / P95 | 命中缓存 TTFA P50 / P95 | 未缓存完整耗时 P50 / P95 | 命中缓存完整耗时 P50 / P95 |
|---|---:|---:|---:|---:|
| zh_short | 238.7 / 247.7 | 201.7 / 204.8 | 529.3 / 542.0 | 483.6 / 494.7 |
| zh_long | 346.8 / 356.7 | 203.5 / 205.6 | 1,425.7 / 1,448.8 | 1,265.0 / 1,279.9 |
| en_short | 205.7 / 209.6 | 202.8 / 206.1 | 564.1 / 580.7 | 556.2 / 564.4 |
| en_long | 203.7 / 241.6 | 203.3 / 207.9 | 1,202.6 / 1,374.5 | 1,190.8 / 1,216.8 |

缓存前后四种语料的语义 token 数和 hash 相同。这个结果代表同一进程重复文本的热缓存收益；不代表新文本或冷进程的 TTFA。冷路径的 G2PW batch 8 单独跑了中文长句 30 次预热、20 次测量：G2PW P50 从 batch 4 的 86.2 ms 变为 82.7 ms，但 TTFA P50/P95 从 346.8 / 356.7 ms 变成 352.5 / 377.4 ms，故保留默认 batch 4。

## 语义、音频边界与取消

所有首包阈值的语义 token 数与 hash 均和阈值 24 相同：`zh_short` 51、`zh_long` 167、`en_short` 64、`en_long` 153。SOLA 分块后 PCM 总采样数不是逐样本相等；不同首包阈值改变了 VITS 调用形状和回调边界，因此直接按相同采样下标计算的波形误差不能作为音质结论。

每个阈值保存的首个完整合成结果均计算了回调边界处的最大绝对跳变（一次样本/阈值）：

| 语料 | 阈值 12（候选 / 24） | 阈值 16（候选 / 24） | 阈值 20（候选 / 24） |
|---|---:|---:|---:|
| zh_short | 0.03323 / 0.00005 | 0.01800 / 0.00005 | 0.00002 / 0.00005 |
| zh_long | 0.01704 / 0.02407 | 0.05099 / 0.02407 | 0.01974 / 0.02407 |
| en_short | 0.02971 / 0.00733 | 0.01151 / 0.00733 | 0.00656 / 0.00733 |
| en_long | 0.01409 / 0.02543 | 0.06908 / 0.02543 | 0.00782 / 0.02543 |

阈值 12/16 在部分文本上有更大的边界跳变；阈值 20 也只在这一个样本上未显示系统性恶化。工具没有完成主观音色和连续性判断，人工试听是未完成的验收项。试听对照文件在 `target/r1.3-audition/`，每个文件按 12、16、20、24 排列，片段间隔 0.75 秒。

每个阈值均执行一次“首个 callback 后取消，再合成恢复”检查：取消按预期返回 `Cancelled`，随后合成成功且所有样本有限值。该检查是语义/恢复冒烟，不是 20 次取消压力测试。

## DirectML 状态：BLOCKED

当前 ORT 1.30 runtime 未提供 `onnxruntime_providers_dml.dll`。DirectML 回归在 Execution Provider 初始化时返回 `Specified provider is not supported.`，没有进入模型加载或合成，因此不能报告 PASS。CUDA R1.3 数据不受此阻塞影响。

## 修改和验证

- `src/profiling.rs`：增加 Frontend 子阶段、request-relative 事件、逐 token 时间线、缓存计数和 SOLA 计时。
- `src/frontend.rs`：增加分阶段计时、默认启用的有界精确文本缓存，以及仅通过环境变量调整的 G2PW 批次。
- `src/inference.rs`：增加只用于 profiling 的首包阈值入口；公开常规流式 API 仍使用 24/24。
- `src/streaming.rs`：独立记录 SOLA 阶段。
- `examples/r0_profile.rs`：汇总 ORT CUDA memcpy trace 事件。
- `examples/r1_3_profile.rs`：新增阈值矩阵、原始逐 token 时间线、样本/边界对照、试听 WAV 和取消后恢复 runner。

结果文件保存在被忽略的 `target/` 下：

- `target/r1.3-threshold-uncached/r1.3-first-audio.json`：四个语料、四种阈值、每组 30/20 数据。
- `target/r1.3-cache-on/r1.3-first-audio.json`：重复文本缓存命中组。
- `target/r1.3-g2pw-batch8/r1.3-first-audio.json`：中文长句 batch 8 对照。
- `target/r1.3-first-timeline/`：分阶段 ORT trace 和 CUDA memcpy 事件。
- `target/r1.3-cpu-regression/results.json`：CPU 路径五种输入与取消恢复结果。
- `target/r1.3-audition/`：四个完整音频对照 montage 与顺序说明。

| 检查 | 状态 |
|---|---|
| CUDA GPT + CUDA EP VITS，四语料 × 四阈值，30 warmup + 20 样本 | PASS，VITS 显式允许审计的 CPU helper nodes |
| 缓存命中组，四语料，30 warmup + 20 样本 | PASS，所有计时样本为 1 hit / 0 miss，token 数/hash 不变 |
| G2PW batch 8 中文长句，30 warmup + 20 样本 | PASS，TTFA 未改善，未改默认 batch |
| CPU `stream_bench` 中文、混合、中英文、短句与取消恢复 | PASS，5 组 E2E 合成，取消后后续合成成功 |
| DirectML E2E | BLOCKED，当前 ORT 缺 DirectML EP |
| 人工试听 | NOT RUN，音频已导出；默认阈值保持 24 |

没有提交或推送修改。

## 后续建议

1. 人工试听 `target/r1.3-audition/` 的四个对照 montage，重点留意首段音色、边界咔哒声和后续连续性；在得到验收前不改默认阈值。
2. 为阈值 20 和 24 增加多次独立首音频边界统计，并使用时间对齐后再做波形误差比较。
3. 冷文本优先继续剖析 G2PW 和 BERT 的模型执行耗时；batch 8 未能降低 TTFA。缓存只对会重复出现的文本有帮助。
4. 提供与 CUDA 基准匹配的 DirectML provider runtime 后，重新跑 CPU/DirectML 合成、取消和恢复回归。
