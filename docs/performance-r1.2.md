# R1.2 流式 VITS CUDA EP 评估

测量日期：2026-10-26。基于 R1.1 的 CUDA GPT、GPU 常驻 KV Cache 和 CPU Token Sampling，评估 ABI-1 `sovits_stream` 迁移到 CUDA EP 后的流式首包与完整合成表现。模型包和参考音频保存在本机，没有加入仓库。

## 结论

在 RTX 5070 Ti、ORT 1.30.0、CUDA 13.0 上，CUDA EP VITS 的主要计算节点确实运行于 GPU。ORT 仍把 570 个 VITS 辅助/形状节点分配给 CPU；禁用 CPU fallback 时，VITS Session 因此无法创建。启用显式诊断 fallback 后，完整流式合成成功，CPU 节点没有 Gemm、MatMul、Conv、Attention 或归一化节点。

同一模型包、文本、采样种子和 Chunk 24 下，GPU GPT + CUDA EP VITS 相对 GPU GPT + CPU VITS 将 TTFA P50/P95 降低 **9.1%/8.4%**，VITS Session 总耗时降低 **58.6%/57.8%**，总耗时和 RTF 降低 **26.5%/25.1%**。数值和分块边界比较通过；没有做人工试听。

CUDA VITS 仍是实验性 opt-in，CPU VITS 仍是默认路径。实测路径需要显式允许已列出的 VITS CPU 节点，不能称作全 GPU VITS。

## 运行时和测量方法

| 项目 | 实测配置 |
|---|---|
| 系统 | Windows x64 |
| GPU | NVIDIA GeForce RTX 5070 Ti，SM 12.0，16 GiB |
| 驱动 | 610.74 |
| ONNX Runtime | 1.30.0.20260909.3，commit `f2c39fe`，动态加载 |
| CUDA / cuDNN | CUDA 13.0 / cuDNN 9.12 |
| Rust wrapper | `ort` 2.0.0-rc.12 |
| 模型 | 本机 ABI-1 GPT-SoVITS v2Pro 流式包 |
| 语料 | 固定中文测试句，seed 1234，Chunk 24 |

CPU VITS 和 CUDA EP VITS 都使用同一 CUDA GPT 后端、同一个 ORT 运行时和模型包。每条路径完成 30 次预热、20 次计时；两路均生成 167 个语义 token、7 个音频回调。TTFA 从合成请求开始计至首个音频回调；RTF 为合成墙钟时间除以生成音频时长。热态计时没有启用 ORT 算子级 trace；节点 trace 来自单独的一次诊断合成，避免把 trace 开销混入性能数据。

## 性能对照

表中均为 20 个测量样本的 P50/P95。阶段耗时由 `r0-profiling` 记录；VITS per-call 列将每个 Chunk 的 7 次调用合并统计。

| 指标 | GPU GPT + CPU VITS | GPU GPT + CUDA EP VITS | 变化 |
|---|---:|---:|---:|
| TTFA (ms) | 384.5 / 397.0 | **349.4 / 363.8** | −9.1% / −8.4% |
| 首次 VITS 调用 (ms) | 108.4 / 110.8 | **69.1 / 73.0** | −36.2% / −34.2% |
| VITS Session 总耗时 (ms) | 861.4 / 866.3 | **356.7 / 365.3** | −58.6% / −57.8% |
| 单次 VITS 调用 (ms，pooled) | 124.5 / 131.8 | **39.9 / 75.8** | P50 −68.0% / P95 −42.5% |
| GPT Encoder (ms) | 6.1 / 10.1 | 6.2 / 10.8 | 基本持平 |
| GPT Step 总耗时 (ms) | 921.9 / 956.5 | 910.5 / 951.7 | −1.2% / −0.5% |
| GPT Step 单 token (ms，pooled) | 5.35 / 6.50 | 5.35 / 6.11 | P50 持平，P95 −6.0% |
| Frontend (ms) | 137.0 / 140.1 | 137.5 / 140.2 | 基本持平 |
| HostDataCopy (ms) | 0.107 / 0.115 | 0.098 / 0.110 | 接近 |
| 显式 DeviceSync (ms) | 0 / 0 | 0.111 / 0.115 | CUDA 路径单独计时 |
| VITS 输出校验和拼接 (ms) | 3.96 / 4.24 | 4.00 / 4.11 | 接近 |
| Audio callback (ms) | 0.080 / 0.129 | 0.072 / 0.118 | 接近 |
| 完整合成 (ms) | 1,929.7 / 1,963.7 | **1,419.4 / 1,471.3** | −26.5% / −25.1% |
| RTF | 0.290 / 0.295 | **0.213 / 0.221** | −26.5% / −25.1% |

VITS 是原 CPU 路径和 GPU 路径的主要差异来源。切到 CUDA EP 后，GPT Step 仍是总时长最大阶段；Frontend 约 137–140 ms，没有随 VITS EP 改变。`VitsOutput` 统计输出验证和 SOLA 拼接，实际 Rust `to_vec` 输出拷贝另记在 `HostDataCopy`。

### 内存和传输

| 指标 | CPU VITS | CUDA EP VITS |
|---|---:|---:|
| 进程私有内存峰值增量 (post-load) | 600 MiB | 985 MiB |
| 工作集峰值增量 (post-load) | 446 MiB | 780 MiB |
| `nvidia-smi` 整机显存采样范围 | 2,510–3,787 MiB | 2,515–4,381 MiB |

显存以 250 ms 间隔采样，但 Windows WDDM 下这里只能取得整机用量，不能归因到 TTS 进程；两个区间也可能受到其他 GPU 进程影响。进程内存数据来自当前进程。

VITS I/O Binding 将参考频谱、说话人向量和文本条件保存在单个合成请求的 Binding 中，仅绑定一次；语义 token、结果长度和 overlap latent 按 Chunk 更新。一次完整合成传入的静态条件张量 payload 为 750,740 B，动态输入为 23,856 B；latent 和完整 audio 输出 payload 为 1,191,424 B。输出显式绑定到 CPU，因为现有 SOLA、latent overlap 和音频回调均消费主机内存。`DeviceSync` 单独统计 `synchronize_outputs()`；ORT trace 没有给出 DMA 字节计数，因此上述是张量 payload 记账，不是硬件实测传输量。

VITS 的输出形状随 Chunk 改变。若跨调用复用动态输出缓冲，最终 Chunk 会出现 `{1,192,52}` 对 `{1,192,48}` 的尺寸冲突。当前实现每次推理前清空并重绑 CPU 输出，避免错误复用；因此 ORT 原生分配和 GPU 分配没有被 Rust 全局分配器计数覆盖。Rust allocator 仅采集到每次诊断合成约 5.4 KiB（CPU VITS）和 3.4 KiB（CUDA VITS）的 `VitsRun` 分配，以及约 1.22 MiB 的 `HostDataCopy` 分配；这些数字不包含 ORT C/C++ 和 CUDA allocator。

## 节点放置和 CPU 回退

最终 ORT 诊断 trace 的节点分配：

| Session | CUDA EP 唯一节点 | CPU EP 唯一节点 | CPU 主计算节点* |
|---|---:|---:|---:|
| GPT Encoder | 826 | 272 | 0 |
| GPT Step | 864 | 0 | 0 |
| VITS Streaming | **1,515** | **570** | **0** |

\* 主计算节点按 Gemm、MatMul、Conv、Attention、LayerNormalization 和 RMSNormalization 统计。VITS trace 中的 CPU 节点由 Unsqueeze 188、Concat 140、Gather 129、Sub 48、Mul 36、Cast 13、Add 12、Equal 1、Reshape 1、Tile 1、Where 1 类组成。7 次流式调用中，CPU 侧为 3,990 个节点事件；这些 CPU 算子的 ORT trace 总耗时约 28 ms。所有 570 个 CPU 节点的名称、算子和事件数见[完整清单](performance-r1.2-vits-cpu-nodes.json)。

将 VITS 的 CPU fallback 关闭后，ORT 在 Session 创建阶段因上述 CPU 节点拒绝加载。基准使用 `ALICE_NATIVE_TTS_CUDA_VITS_DIAGNOSTIC_ALLOW_CPU=1` 明确允许这些节点；GPT Step 仍关闭 CPU fallback，trace 为 864/864 CUDA 节点。GPT Encoder 的 R1.1 辅助节点许可保持独立。CUDA VITS 没有改变默认路径，也没有修改 ONNX 图或模型权重。

## 数值与音频边界

| 对照项 | CPU VITS vs CUDA EP VITS |
|---|---:|
| Token 数 / hash | 167 / `15228212952513995076`，一致 |
| 输出采样数 | 212,777 / 212,777 |
| 回调数与样本分段 | 7 / 一致 |
| Waveform MAE / RMSE | `5.63e-7` / `1.88e-6` |
| Waveform 最大绝对误差 | `4.89e-5` |
| Cosine similarity / SNR | `0.999999999915` / `97.71 dB` |
| Chunk 边界绝对跳变 P50/P95/最大值 | CPU `0.004604 / 0.024068 / 0.024068`；CUDA `0.004604 / 0.024068 / 0.024068` |

相同 token 和分段、低误差及相同边界指标表明这次合成没有可测到的连续性回退。SHA-256 不同，因此不是逐采样点相同；本阶段没有人工试听，主观音色验收尚未完成。

## DirectML 回归状态：BLOCKED

当前 CUDA 基准使用的 ORT 1.30 runtime 目录没有 `onnxruntime_providers_dml.dll`。以该 runtime 启动最终 DirectML 回归时，ORT 返回 `Specified provider is not supported.`，合成未进入模型加载，不能计为 DirectML E2E PASS。已安装的 Python `onnxruntime` 是 1.24.3 CPU 包，也不是与当前基准匹配的 DirectML runtime。DirectML 代码路径未被本阶段改动，但需要提供匹配的 Windows ORT DirectML runtime 后才能完成本项验收。

## 修改与验证

- `src/inference.rs`：新增 CUDA VITS 显式 opt-in；保留 CPU VITS 默认值和 `set_streaming` 回退路径。GPT、KV Cache、CPU sampler、ABI-1 和公开 API 保持不变。
- `src/streaming.rs`：为流式 VITS 加入请求级 I/O Binding，静态参考/文本条件一次绑定；动态 Chunk 输入逐次更新，输出显式同步并落到 CPU。按变化的输出 shape 重绑输出缓冲，维持原取消检查和 SOLA/Chunk 行为。
- `src/profiling.rs`：增加 DeviceSync 阶段、单次 VITS run/output/sync 数据，以及静态/动态输入和输出 payload 记账。
- `examples/r0_profile.rs`：增加 `--cuda-vits`、`--cuda-vits-allow-cpu`、分阶段 P50/P95 汇总和完整 CPU 节点记录。
- `docs/performance-r1.2-vits-cpu-nodes.json`：保存当前 ORT VITS trace 的 570 个 CPU 节点。

| 验收项 | 结果 |
|---|---|
| Release check：`r0-profiling` | Passed |
| Release check：`r0-profiling r1-cuda-ep` | Passed |
| 库测试：`r0-profiling r1-cuda-ep` | 14 passed，0 failed，2 ignored（需要外部测试模型包） |
| CUDA GPT + CPU VITS E2E，30 warmup + 20 samples | Passed |
| CUDA GPT + CUDA EP VITS E2E，30 warmup + 20 samples | Passed，显式 CPU 节点诊断 fallback |
| 运行时-only `r1-cuda-ep` 冒烟（不含 `r0-profiling`） | 5 组中英合成通过；首包后取消无额外回调，后续合成恢复通过 |
| CUDA VITS 禁用 CPU fallback | Session 创建失败，570 个 CPU 节点阻止严格放置 |
| DirectML E2E | BLOCKED：ORT 1.30 未提供 DirectML EP |
| 人工试听 | 未执行 |

没有提交或推送修改。原始 benchmark JSON、波形和 `nvidia-smi` 采样保存在本地忽略目录 `target/r1.2-cuda-gpt-cpu-vits-final`、`target/r1.2-cuda-gpt-cuda-vits-final`；单次节点 trace 在 `target/r1.2-cuda-vits-profile-final`。

## 下一步

1. 先补齐匹配的 DirectML runtime 并跑现有公开路径回归。
2. 若要进一步降 TTFA，优先调查 GPT Step（约 910 ms）和约 137 ms Frontend；VITS CUDA 已把原来的声学解码成本压到约 357 ms。
3. 评估将 overlap latent 保留在 GPU、仅回传必须播放的音频；这可能省去部分 D2H payload，但要重新验证 SOLA、取消时机和音频边界。
4. 对 570 个 CPU shape/helper 节点单独实验图改写或算子融合；在节点 trace 和音频一致性验证前，不放宽严格模式，也不改变默认后端。
