# R1.7 GPT Launch & Copy Optimization

测量日期：2026-10-26。分支 `perf/r1-5-gpt-execution`，冻结基线 `880c394`（R1.4）。保留此前工作；本轮没有提交或推送。

## 结论

- R1.6 的 Nsight trace 表明每 Token 有 523 次 CUDA Kernel Launch、约 147 次 Async Copy API；GPU 操作区间并集约 1.34 ms，而 GPT Step 墙钟中位数约 6.33 ms，操作间累计空档约 4.84 ms。优先级仍是减少 Host 提交和缓存周边拷贝。
- CUDA Graph 独立验证未能创建 Session。ORT verbose 日志显示图转换插入了 `MemcpyToHost`（位于 `/Unsqueeze_3_output_0` 前），之后报告有计算节点未分配给 CUDA EP。Graph 实验为 0 warmup / 0 samples；没有性能或质量结果。
- 独立试验了版本化的 per-layer K/V cache 图和首 Token bridge。它保持相同 Token 序列与数值近似，但 30 warmup + 20 samples 的 TTFA、GPT Step 和完整耗时均更慢。因此不设为默认。
- R1.7 的 Nsight 短范围 A/B 未取得报告：Nsight CLI 启动目标后一直停在进程注入阶段，简单 `cmd /c echo` 探针也能复现。没有 R1.7 候选的 Kernel Launch 数、Memcpy 数/字节或 GPU gap 结果；不把 R1.6 数字外推到候选。
- `Memcpy` 活动没有导出源/目标地址；即使 R1.6 trace 中可见的 D2D 也不能逐地址映射到 KV Cache 或某个 ONNX 节点。本轮没有依据声称消除了 D2D 往返。

## 环境和实验设置

| 项目 | 配置 |
|---|---|
| 系统 / GPU | Windows x64 / NVIDIA GeForce RTX 5070 Ti 16 GiB / 驱动 610.74 |
| ORT | ONNX Runtime 1.30.0；Rust `ort` 2.0.0-rc.12，动态加载 |
| CUDA / cuDNN | CUDA 13.0 / cuDNN 9.12 |
| 模型 | 本地 ABI-1 GPT-SoVITS v2Pro 包；权重、参考音频、文本与采样配置不变 |
| Sampling / Chunk | CPU Sampling，seed 1234，Chunk 24 |
| 绑定 / Graph | `clear_rebind`；CUDA Graph 关闭（Graph-on 只用于独立创建 Session 验证） |
| 基准 | 默认图和 V2 图分别 30 warmup + 20 samples；相同文本、模型、硬件和前端缓存命中设置 |

Nsight Systems 2026.5.1.161-265138896106v0 已安装，本轮复用该安装，没有重新下载或安装。R1.7 runner 在进程内预热后，调用 `cudaProfilerStart/Stop` 将目标缩到首个音频回调；但 Nsight CLI 没有启动目标程序并输出报告。Nsight 的 CPU sampling / context-switch 环境检查也显示当前进程没有管理员采样权限。采集失败不改变无插桩的 30+20 性能数据。

## 根因分析

### Launch 与 GPU 空档

R1.6 诊断 trace `target/r1.6-nsys-local/gpt-step-nvtx-api.nsys-rep` 覆盖一次 167 Token 请求，没有预热，故只用于结构诊断。该 trace 将每步约 6.33 ms 的 ORT 调用分为：523 次 Kernel Launch API、约 147 次 `cudaMemcpyAsync` API、约 50 次 Event / Stream 同步 API；GPU 实际 Kernel 和 Memcpy 的区间并集约 1.34 ms，其余约 4.84 ms 是大量小间隙累计。Host API 时间可能与 GPU 执行重叠，不能直接相加为 ORT 墙钟。

### KV Cache 与 D2D

R1.6 SQLite 中 24,090 个 D2D 活动共 12,743,091,908 bytes，约 144.25 次 / Token、76.3 MB / Token；其中 8,016 次恰为 1 MiB（48 次 / Token）。Nsight SQLite 的 `virtualAddress` 和 `deprecatedSrcId` 对这些 D2D 记录为空。可见大小模式与 24 层 K/V cache 边界相符，但这只构成待验证线索，不能据此断言每次复制都来自 KV Cache。

本轮 ORT 详细日志 `target/r1.7-graph-verbose/ort-verbose.log` 有如下记录：

```text
TransformerMemcpyImpl::AddCopyNode: Add MemcpyToHost before /Unsqueeze_3_output_0 for CUDAExecutionProvider
MemcpyTransformer ... 1 Memcpy nodes are added to graph main_graph
This session cannot use graph capture ... as all compute graph nodes have not been partitioned to CUDAExecutionProvider
```

Graph-off 的 ORT profile 中，`/Slice_3` 仍由 CUDA EP 执行；计算图主体没有因此回退到 CPU。当前证据将 Graph 创建失败指向转换器插入的 Host copy 边界，但没有 CUPTI ORT 节点关联来精确追溯其输入来源。没有修改模型语义来绕过这条动态控制路径。

### Nsight 短范围采集阻塞

R1.7 runner 已单独构建并能独立运行：1 次预热后完成 167 Token 合成，输出有效 WAV 和 JSON。但运行以下类型的 Nsight capture 时，Nsight 进程停在启动阶段，目标进程和 `.nsys-rep` 均未出现：

```text
nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop ... r1_7_copy_trace.exe ...
```

移除 capture-range 后仍复现；禁用 CPU sampling / context-switch 后，对 `cmd.exe /c echo` 的 `--trace=none` 探针也复现。故这不是 GPT 图初始化或模型错误的证据，具体 Nsight 注入阻塞原因仍未定位。失败会话已中止，未留下 profiler 进程或有效报告。

## 优化候选：版本化 per-layer K/V Cache

新增 opt-in 的 `gpt_step_v2.onnx` 路径，将 ABI-1 打包 K/V 输入转换为 24 层 K/V 输入/输出；首步通过独立 `gpt_step_init_v2.onnx` bridge 图分割 packed cache，后续仍用双缓冲 I/O Binding，CPU Sampling、EOS、Chunk 和公开 API 不变。使用 `ALICE_NATIVE_TTS_GPT_STEP=2` 与 feature `r1-gpt-step-v2` 显式启用；ABI-1 仍是默认，Graph capture 开关保持关闭，两项实验没有混合。

`tools/prepare-gpt-step-init-v2.py` 离线生成首步 bridge 和哈希 manifest，不改原 ABI-1 图或权重。V2 只通过新增版本化图扩展模型包；未提供新图时返回明确错误，不静默切换实现。缓存 CUDA endpoint 断言在 167 步中共通过 668 次；这证明输入/输出 Tensor 位于 CUDA endpoint，不能代替 Nsight 对内部 D2D 或 Host staging 的检查。

### 30 warmup + 20 samples 对比

| 指标 | ABI-1 基线 P50 / P95 | V2 per-layer P50 / P95 | 变化 |
|---|---:|---:|---:|
| TTFA | 209.878 / 216.815 ms | 217.699 / 235.758 ms | +3.7% / +8.7% |
| 完整合成 | 1,302.167 / 1,319.907 ms | 1,369.850 / 1,463.700 ms | +5.2% / +10.9% |
| RTF | 0.19584 / 0.19850 | 0.2060 / 0.2201 | +5.2% / +10.9% |
| GPT Step 单 Token 墙钟 | 5.412 / 6.484 ms | 5.812 / 6.991 ms | +7.4% / +7.8% |
| GPT Step 全阶段 | 930.189 / 939.632 ms | 986.603 / 1,057.221 ms | +6.1% / +12.5% |

这些数据来自各自无插桩的热态基准，Nsight instrumentation 没有混入。V2 在所有核心延迟指标上退化，未达到性能 Gate。

内存为进程工作集：基线加载后约 916 MB、测量峰值约 1,757 MB；V2 加载后约 950 MB、峰值约 1,792 MB。候选峰值多约 35 MB。没有可靠的进程级 VRAM 数字；`nvidia-smi` 的 WDDM 整机显存读数会受其他进程影响，不能用来宣称 V2 的显存增量。

## Token、波形和流式检查

- 基线与 V2 各 20/20 次均完成 167 个语义 Token，Token hash 均为 `15228212952513995076`，音频长度均为 212,777 samples，完整合成正常结束。
- 以首段音频对照：`max_abs=2.384e-7`、`MAE=9.316e-9`、`RMSE=1.701e-8`、相关系数 `0.999999999999993`。CUDA 浮点结果不是逐位稳定，因此完整 WAV FNV hash 不同；数值误差仍在当前对照范围内。
- 独立流式回归覆盖中、英、混合语言、短句和较长分句。取消在首个回调后生效，额外回调为 0；后续合成成功恢复。此组用于语义与取消检查，不是 30+20 性能数据。
- Nsight 候选 A/B 没有跑成，所以本轮无法量化候选 Kernel launch 数、D2D 次数/字节和 GPU busy gap。质量通过不能抵消性能退化，也不据此切换默认图。

## 代码与运行验证

R1.7 本轮新增或扩展：

- `src/inference.rs`：加入显式 opt-in 的 V2 per-layer cache 状态、首步 bridge Session、双缓冲 CUDA I/O Binding 与 endpoint 校验；ABI-1 默认路径保留。
- `tools/prepare-gpt-step-init-v2.py`：离线从现有模型图生成版本化首步 bridge 和哈希 manifest。
- `examples/r1_7_copy_trace.rs`：预热后在首个回调停止 CUDA profiler capture，并完成剩余合成用于正确性检查。
- `examples/r1_5_profile.rs`、`Cargo.toml`：结果 JSON 标注图版本并注册新 runner。
- `docs/performance-r1.7.md`：本报告。

验证结果：

- `cargo fmt --all`：通过。
- `cargo check --release --features 'r0-profiling r1-cuda-ep r1-gpt-step-v2' --examples`：通过。
- `cargo check --release --features 'r1-cuda-ep r1-gpt-step-v2' --example stream_bench`：通过。
- CUDA V2 E2E 30 warmup + 20 samples：完成；Token、音频长度、数值波形检查完成。
- 中英混合流式取消与恢复：完成，取消后无额外回调且下一次合成成功。
- Nsight R1.7 baseline / V2 短范围 trace：**BLOCKED**，未生成报告。
- 修改没有自动提交或推送；ABI-1 默认、CPU / DirectML 路径、Chunk 24 与 CPU Sampling 默认行为未切换。

## 瓶颈排序和后续工作

1. **Host Kernel Launch / ORT 提交与 GPU 空档**：R1.6 每步 523 次 Launch API，GPU busy 约 1.34 ms 对比约 6.33 ms Run 墙钟，是首要优化方向。先恢复可工作的短范围 Nsight capture，再做 candidate launch/copy A/B。
2. **D2D 拷贝源未映射**：约 76.3 MB D2D / Token 是高优先级调查对象，但当前 trace 无地址。需要支持逐调用 src/dst/bytes 与 CUDA stream 的 CUPTI runtime callback 或等价受控插桩；再与 ORT 节点和 KV Buffer 生命周期关联。不要仅按尺寸猜测来源。
3. **CUDA Graph 的 Host copy 边界**：明确 `/Unsqueeze_3_output_0` 前有 `MemcpyToHost`。需用 ORT 节点 profiling 和动态图输入追踪查清来源，再单独设计保语义的图兼容修复；目前 graph-on 为 Session 创建失败，不应进入性能结论。
4. **Per-layer cache 图**：本候选已验证没有语义回归但变慢。后续只有在 Nsight 证实它确实减少缓存复制、且无插桩端到端也改善时才继续，不启用为默认。
5. **残余验证**：本轮没有 DirectML E2E、盲听或逐节点 CUPTI ORT 关联；这些不影响当前 CUDA A/B 结论，但仍是更广泛后端/音质验收的剩余项。
