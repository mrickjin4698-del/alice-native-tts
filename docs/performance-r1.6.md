# R1.6 GPU Kernel Profiling

测量日期：2026-10-26。实验继续使用 `perf/r1-5-gpt-execution`，冻结基线为 `880c394`（R1.4）。本轮没有提交或推送。

## 结论与状态

**R1.6 已采集真实 Nsight Systems CUDA + NVTX 时间线。** GPT Step 每 Token 平均触发 523 个 CUDA Kernel Launch、约 147 次异步拷贝 API；中位步的 GPU 实际执行约 1.34 ms，而 ORT 调用墙钟约 6.33 ms，其中约 4.84 ms 是相邻 GPU 操作之间的空档。Nsight 证据将优先级指向 Host API / ORT 提交路径和 GPT Step 内的大量 D2D 拷贝，而不是 GPU 算术吞吐。D2D 拷贝尚未逐个映射到 KV Tensor。该 trace 是单请求诊断采集，不作为性能 Benchmark；无 Nsight 插桩的 30 warmup + 20 samples 基线仍是性能比较依据。

CPU context-switch / 调度采样仍 **BLOCKED**：Nsight 明确要求 Windows 管理员令牌，当前进程没有该令牌。其余 CUDA API、Kernel、memcpy 与 NVTX 时间线已成功采集。当前 ORT 动态库没有启用 CUPTI ORT 节点关联；不影响 Nsight 的 CUDA timeline，但暂时不能把 ORT 内部节点名逐项对应到 CUDA Kernel。

CUDA Graph 已做 opt-in 最小验证，Session 创建时被 ORT 拒绝，未进入 Warmup 或推理。默认仍关闭 CUDA Graph，R1.5 的 CUDA EP、KV Cache、CPU Sampling、Chunk 24、取消及公开 API 均未改变。

## 运行环境与采集限制

| 项目 | 配置 / 状态 |
|---|---|
| 系统 / GPU | Windows x64 / NVIDIA GeForce RTX 5070 Ti 16 GiB / 驱动 610.74 |
| ORT | 1.30.0；Rust `ort` 2.0.0-rc.12，动态加载 |
| CUDA / cuDNN | CUDA 13.0 / cuDNN 9.12 |
| 模型 | 本地 ABI-1 GPT-SoVITS v2Pro；参考音频和权重未改 |
| Sampling / Chunk | CPU Sampling，seed 1234，Chunk 24 |
| Nsight Systems | 2026.5.1.161-265138896106v0；已安装于 `C:\Program Files\NVIDIA Corporation\Nsight Systems 2026.5.1`；Windows CUDA / NVTX trace 正常 |
| Nsight Compute | `ncu` 不在 PATH；本阶段只需 Nsight Systems Kernel 时间线 |
| CUPTI | PyTorch 目录有 `cupti64_2025.3.0.dll`，但无 CUPTI SDK 头文件 / 导入库；ORT 未启用 `--enable_cuda_profiling` |
| TensorRT | 详见下文；没有可运行的 TensorRT 11 SDK / `trtexec` |

Nsight Systems 通过用户授权进行系统安装；安装包签名为 NVIDIA Corporation。本阶段没有更改 GPU 驱动或 CUDA Runtime。NVIDIA 文档说明 Nsight Systems Windows CUDA trace 支持 CUDA 12.0+，硬件要求 Turing 或更新；当前 RTX 5070 Ti / CUDA 13 满足门槛。[Nsight Systems 下载与系统要求](https://developer.nvidia.com/nsight-systems/get-started)

ORT 的 CUDA Kernel profiling 需要 CUPTI，并要求 ORT 构建时启用 CUDA profiling；当前机器只有可加载的 CUPTI DLL，不能据此声称已得到 Kernel trace。[ORT Profiling 文档](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html)

## GPT Step 可观测成本

### 新跑的默认路径基线

以 `clear_rebind`、CUDA Frontend cache hit、同一短句、同一模型 / 参考音频 / seed / Chunk 24，Graph 关闭，运行 30 次预热 + 20 次测量。完整 JSON 保存在 `target/r1.6-default-control/r1.5-gpt-execution.json`。

| 指标 | P50 | P95 | 备注 |
|---|---:|---:|---|
| GPT Step 每 Token 墙钟 | 5.412 ms | 6.484 ms | 3,340 个 Token Step 池化；覆盖每次请求全部 167 Token；Rust 侧计时 |
| GPT Step 全程阶段墙钟 | 930.2 ms | 939.6 ms | 每次请求；完整 167 Token 生成阶段 |
| ORT `RunWithBinding` 每 Token 墙钟 | 5.372 ms | 6.570 ms | 每请求首个 Chunk 的 24 Token，共 480 个事件；含 ORT 调用及其返回前的等待；不是 GPU Kernel 时间 |
| I/O Binding 更新 | 0.0233 ms/token | 0.0357 ms/token | 每请求首个 Chunk 的 480 个事件；CPU/Rust 计时 |
| CPU Token Sampling | 0.0007 ms/token | 0.0011 ms/token | 每请求首个 Chunk 的 480 个事件；Sampling 本身远小于 ORT Run 墙钟 |
| 输出检查 | 0.0008 ms/token | 0.0013 ms/token | 每请求首个 Chunk 的 480 个事件；Rust 侧计时 |
| TTFA | 209.9 ms | 216.8 ms | 请求开始至首次音频 Callback |
| 完整合成 | 1,302.2 ms | 1,319.9 ms | 同一短句 |
| RTF | 0.1958 | 0.1985 | 全程墙钟 / 音频时长 |

20/20 输出均为 167 个语义 Token，Token 哈希相同（`15228212952513995076`），音频均为 212,777 samples。与 R1.5 同配置基线首条波形比较：最大绝对差 `2.38e-7`，MAE `9.31e-9`，RMSE `1.70e-8`，余弦相似度 `0.99999999999995`。这是默认路径复跑一致性检查，不是 CUDA Graph 候选的质量结果。

进程工作集约 916 MB（模型加载后）、1,757 MB（测量峰值）。`nvidia-smi` 全局显存使用量从加载前 2,460 MiB 到模型加载后 7,304 MiB、测量后 7,344 MiB。Windows WDDM 下该数字是整机 GPU 使用量，不能归属于此进程；Graph 失败前没有可用的候选显存数据。

注：Profiling timeline 在首次音频 Callback 处停止，因此逐阶段 `RunWithBinding`、绑定更新、Sampling 和输出检查统计覆盖首 Chunk 24 Token；全程 GPT Step 由 `gpt_step_token_ms` 单独统计，故 Pooled count 为 3,340。两种统计范围没有混用。

### Nsight Systems GPU 时间线

采集文件：`target/r1.6-nsys-local/gpt-step-nvtx-api.nsys-rep`；启用 CUDA 与 NVTX trace，通过 `cudaProfilerStart/Stop` 将 trace 收窄到一条合成请求。该条诊断请求没有 Nsight 预热（runner 参数为 0），捕获了 167 个 GPT Step；尝试 30 次 Nsight 预热的采集超时，所以不能把此 trace 作为热态 benchmark。热态性能数据来自上节独立的 30 warmup + 20 samples 默认路径实验。逐步时间由 Nsight CSV trace 和 `gpt_step_ort_run` NVTX 标记关联；CPU context-switch trace 未启用（需要管理员令牌）。CUDA Graph 候选在 Session 创建时失败，因此不存在可对比的 graph-on 20-sample 组。

| 每个 GPT Step | P50 | P95 | 口径 |
|---|---:|---:|---|
| ORT `RunWithBinding` NVTX 墙钟 | 6.329 ms | 7.134 ms | 167 个单 Token Run；Nsight 插桩，仅作诊断 |
| CUDA Kernel Launch API 时间 | 3.014 ms | 3.399 ms | 每 Token 523 次 Launch API；Host API 调用耗时，可能与 GPU 执行重叠 |
| `cudaMemcpyAsync` Host API 时间 | 1.118 ms | 1.367 ms | 每 Token 约 147 次 API；调用本身的 Host 时间 |
| 同步 API 时间 | 0.038 ms | 0.056 ms | 每 Token 约 50 次 Event / Stream Synchronize API；不等于所有等待时间 |
| CUDA Kernel 执行 | 1.177 ms | 1.783 ms | GPU 时间，不含 Memcpy |
| GPU Memcpy 执行 | 0.166 ms | 0.222 ms | 主要为 Device-to-Device |
| GPU 操作 busy union | 1.343 ms | 2.005 ms | 同一 Token 的 Kernel / Memcpy 实际时间区间并集 |
| GPU 操作间空档 | 4.836 ms | 5.604 ms | 首个至最后一个 GPU 操作跨度减去 busy union；不是单个长阻塞 |

一个中位附近的 Token（trace range index 83）用时 6.317 ms，GPU 首末操作跨度 6.289 ms；523 个 Kernel Launch API 花费 3.037 ms，147 个 `cudaMemcpyAsync` API 花费 1.069 ms，Event / Stream 同步约 0.035 ms。GPU 侧执行 523 个 Kernel、约 148 个 Memcpy operation；Kernel 执行 1.181 ms，Memcpy 约 0.163 ms，设备区间并集 1.346 ms，区间内累计空档约 4.943 ms，最大单个空档约 70 µs。空档由大量细小的 Host 提交间隙组成，而非单一 4.9 ms Stall。

首个捕获的 `gpt_step_ort_run` 为 39.073 ms，是明显冷启动离群值；原因未单独隔离。表中的 P50/P95 反映完整 167 步 trace，非热态或无插桩 benchmark。

该 Token 的 GPU Memcpy 里有 144 次 D2D，累计约 76.3 MB、执行约 0.163 ms；H2D / D2H 数量与字节数很小。结合 R1.1 对 GPU-resident KV Cache 的验证，可以排除每 Token 把完整 K/V Cache 搬回 CPU 再传回 GPU；不过这份 Nsight 记录没有逐个标注 Buffer 地址，不能断言所有 D2D 都是 KV ping-pong 副本。采集范围的 Nsight NVTX Projection Span 覆盖了绝大部分 ORT 墙钟；GPT Step 关联到的 GPU 操作 busy 只占其中约 21%。API 统计里 Kernel Launch 与 `cudaMemcpyAsync` 占据可见的 Host 时间，Nsight 没有管理员权限采集 CPU 线程切换，所以无法进一步将剩余空档精确分配给 ORT CPU 调度、CUDA runtime 内部等待或线程被抢占。

### 热点 CUDA Kernels

以下为 167 个 Token 的 Kernel 执行时间累计；“每 Token”按 167 除算。算子名来自 CUDA Kernel trace，而非 ONNX 节点名。

| CUDA Kernel | 调用数 | 总 GPU 时间 | 每 Token |
|---|---:|---:|---:|
| cuBLAS `gemvx::kernel` 主变体 | 12,191 | 70.85 ms | 424 µs |
| cuBLAS `gemvNSP_kernel` 16x4 | 4,008 | 30.62 ms | 183 µs |
| ORT Concat | 334 | 18.41 ms | 110 µs |
| ORT Split | 334 | 15.12 ms | 91 µs |
| cuBLAS `gemmk1_kernel` | 16,032 | 15.12 ms | 91 µs |
| ORT GatherScatterElements | 8,016 | 9.88 ms | 59 µs |
| ORT SkipLayerNorm | 8,016 | 9.44 ms | 57 µs |

最大 Kernel 总和只占约 213 ms / 167 次 GPT Step；逐 Token 约 1.28 ms 的 Kernel busy 无法解释约 6.3 ms 的 Run 墙钟。因此本阶段更强的优化候选是减少每步约 523 次 Kernel Launch、147 次 Async Copy / D2D KV 更新和同步边界，再评估减少算子数量；直接重写 GEMM Kernel 的潜在收益较小。该结论只用于下一阶段排序，本轮没有更改图或推理默认路径。

### ORT 节点事件线索（不是 Nsight Kernel 时间）

复用 R1.1 的 `target/r1.1-gpt-cuda-diagnostic/ort-profile/gpt_step-1_2026-10-26_09-20-46_029.json`：167 Token 对应 144,288 个 CUDA EP `*_kernel_time` 节点事件，即每个 Token 记录到 864 个节点；记录到的节点事件 Provider 均为 `CUDAExecutionProvider`。按 ORT JSON `dur` 汇总的高频节点如下：

| ORT 节点名 | 事件数 | 汇总 `dur` | 均值 / 事件 |
|---|---:|---:|---:|
| Transpose | 16,032 | 270.2 ms | 16.9 µs |
| ScatterElements | 8,016 | 185.9 ms | 23.2 µs |
| Slice | 8,016 | 154.3 ms | 19.3 µs |
| `gemm_output_reshape_token` | 15,865 | 146.6 ms | 9.2 µs |
| `SkipLayerNormalization_token` | 7,849 | 134.7 ms | 17.2 µs |
| Reshape | 12,525 | 116.0 ms | 9.3 µs |
| `gemm_input_reshape_token` | 11,857 | 108.9 ms | 9.2 µs |
| MatMul | 4,008 | 79.9 ms | 19.9 µs |

这些是 ORT 节点事件的累计值，不等同于 Nsight 中的 CUDA Kernel 执行区间；不同事件的 `dur` 不能相加后当作 GPU busy time。Nsight 现在已确认 Concat / Split / GatherScatter / Transpose 等对应的 CUDA Kernel 会重复出现在每个 Token，但尚未通过 CUPTI 将具体 ORT 节点逐个映射到 Kernel。两组数据共同支持优先审计 KV 更新周边的布局变换和缓存写入；本轮不改 ONNX 图。

| 目标拆分项 | R1.6 证据 |
|---|---|
| 实际 CUDA Kernel 名称、持续时间、launch 数 | **已采集**；523 Kernel Launch / Token，Kernel busy P50/P95 1.177 / 1.783 ms |
| CUDA API / ORT 调度耗时 | **部分已拆分**；Launch API、Memcpy API、sync API 已量化；Windows CPU context switch 与 ORT 内部节点关联仍 BLOCKED |
| Memcpy 次数与耗时 | **已采集**；约 147 次 Async Copy API / Token；典型步约 76.3 MB Device-to-Device，H2D / D2H 仅有很小数据 |
| GPU 空闲间隙 | **已采集**；GPU 操作之间的累计空档 P50/P95 4.836 / 5.604 ms；Nsight trace 可见较低实际 busy 占比 |
| 显式同步 | Nsight 观测到约 50 次 Event / Stream Synchronize API / Token，Host API 累计 P50/P95 38 / 56 µs；完整隐式等待仍与 Host 提交和 Runtime 重叠 |

## CUDA Graph 独立验证

加入实验性开关 `ALICE_NATIVE_TTS_GPT_STEP_CUDA_GRAPH=1`：启用 ORT CUDA EP Graph，并为两个 KV ping-pong binding 分别传入 `gpu_graph_id` 0 / 1。示例的最后一个参数支持 `cuda_graph=off|on`；默认 `off`。

Graph-on 在 `gpt_step` Session 创建阶段失败，原始 ORT 错误为：

> This session cannot use the graph capture feature as requested by the user as all compute graph nodes have not been partitioned to the CUDAExecutionProvider

因此本候选 **0 次 Warmup、0 次测量**，无 TTFA / RTF、Token / EOS、波形、取消恢复或候选显存结果；没有性能收益证据。旧 ORT trace 中所有记录到的节点事件均来自 CUDA EP，但这不足以解释 Graph Session 的拒绝原因。ORT 文档还要求 Graph 中的节点全部可分配到 CUDA EP，并要求通过 I/O Binding 保持输入 / 输出形状和地址稳定；当前 `samples` / `idx` 每步由 CPU Sampling 新建，top-k 输出也绑定在 CPU，和 Graph 所需固定 CUDA 地址存在结构性冲突。[ORT CUDA Graph 限制](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html#using-cuda-graphs-preview)

Graph 兼容方案尚未实现。需先为 `samples` / `idx` 建固定地址的 CUDA 输入缓冲，并验证 CPU Sampling 结果可按原语义写入；同时确认 top-k 输出与固定绑定能在不改变 CPU Sampling / EOS / 取消行为的条件下工作。任何 CUDA Graph 或 Sampling 迁移都必须单独做 A/B，且 Graph-on 未过数值与音频质量 Gate 前不能替换默认路径。

## TensorRT 路线审查

| 路线 | 当前兼容性与实测状态 |
|---|---|
| ORT TensorRT EP（经典 EP） | 本地存在 `onnxruntime_providers_tensorrt.dll`，但运行缺少 `nvinfer_10.dll`，最小 Add smoke 在 Provider 加载处失败；未创建 Session。ORT 当前公开矩阵列出 ORT `main` / `1.22` + TensorRT 10.9 + CUDA 12.0–12.8，没有列出 ORT 1.30 + CUDA 13 + TensorRT 11，因此当前环境没有已验证的 TRT 11 组合。 |
| TensorRT RTX EP | 与经典 TensorRT EP 是不同后端。官方构建矩阵当前列出 ORT `main` / `1.23` + TensorRT RTX EP 1.1 + CUDA 12.9（ORT 1.22 + EP 1.0 + CUDA 12.8）；支持 Ampere 及更新 RTX GPU。但本机 ORT 1.30 / CUDA 13.0 没有已验证的精确组合，仓库内也没有 `onnxruntime_providers_nv_tensorrt_rtx.dll`；内置 EP 已弃用，建议 standalone EP ABI plugin，当前需源码构建。未能运行最小 GPT Step 原型。 |
| 原生 TensorRT 11 | 本机没有 TensorRT 11 SDK 头文件 / 库、`nvinfer` runtime 或 `trtexec`；没有创建 engine，也没有 `execute_async_v3` 原型结果。 |

TensorRT 11 Plugin / EP 互操作性因此标记 **BLOCKED / UNVERIFIED**，不能依据 AnifLive-TTS 的公开实现推断本项目可直接使用。后续需固定 ORT、CUDA、TensorRT SDK、插件 ABI 版本组合，再用同一 ABI-1 `gpt_step` 图做最小单步数值对照。[ORT TensorRT EP 兼容表](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) · [ORT TensorRT RTX EP 状态](https://onnxruntime.ai/docs/execution-providers/TensorRTRTX-ExecutionProvider.html) · [ORT EP 构建兼容表](https://onnxruntime.ai/docs/build/eps.html) · [AnifLive-TTS 参考](https://github.com/Hiruynk/AnifLive-TTS)

## 验证与修改

- `cargo fmt --all -- --check`：通过。
- `git diff --check`：通过；仅有现存 Rust 文件 CRLF 提示。
- `cargo test --release --lib --features r1-cuda-ep -- --skip first_chunk_accepts_empty_context_tensor`：14 通过、0 失败、3 忽略、1 项过滤。
- 完整库测试和单独运行 `streaming::tests::first_chunk_accepts_empty_context_tensor` 都在该测试开始后超过 60 秒无结果；测试定义已存在于冻结基线 `880c394` 的 `src/streaming.rs`，本轮未改该文件。需要另行查明 ORT 零维 Tensor 创建阻塞原因。
- `cargo check --release --features 'r0-profiling r1-cuda-ep' --example r1_5_profile`：通过。
- `cargo check --release --features r1-cuda-ep --example r1_6_kernel_trace`：通过。
- CUDA 默认路径：30 warmup + 20 samples 成功，EOS 正常结束，20/20 Token 数与哈希一致；首音频与 R1.5 默认基线数值高度一致。
- CUDA Graph-on：Session 创建失败，未进入基准；取消恢复及音频质量测试不适用，不能记为通过。
- 未修改模型、权重、配置、ABI、公开 API、Chunk 默认值或 CPU Sampling；未改 CUDA / DirectML 默认路径。

本轮修改：

- `src/inference.rs`：仅添加 opt-in Graph Provider 配置与每个 ping-pong binding 的 `gpu_graph_id` RunOptions；Graph 默认关闭；为 GPT Step 加入环境变量控制的 NVTX 诊断范围。
- `src/nvtx.rs`：Windows 下按需动态加载 NVTX；只有设置 `ALICE_NATIVE_TTS_NSIGHT_NVTX=1` 时生效。
- `examples/r1_5_profile.rs`：支持选择 Graph on/off 并写入结果 JSON。
- `examples/r1_6_kernel_trace.rs`：CUDA 模型的单请求 Nsight Systems trace runner；支持预热参数与 profiler capture range。
- `Cargo.toml`、`src/lib.rs`：注册 R1.6 runner 与 Windows/CUDA 条件编译的 NVTX 模块。
- `docs/performance-r1.6.md`：本文和实测数据。

## 瓶颈排序与下一步

1. **Host Kernel Launch 与 ORT 调度**：523 次 Launch API / Token，Host API 调用 P50 3.014 ms；GPU 内核 busy 中位数 1.177 ms。GPU inter-op gap 中位 4.836 ms，表明大量细碎提交带来的空档是首要路线。需要管理员权限的 CPU context-switch timeline 来区分 ORT 调度与操作系统抢占。
2. **逐 Token 的 D2D 拷贝和 Async Copy**：约 147 次 `cudaMemcpyAsync` / Token，Host API P50 1.118 ms；典型 Token 的 GPT Step 有约 76.3 MB D2D，GPU 执行约 0.163 ms。它很可能与 KV ping-pong 更新有关，但当前 trace 未逐个关联 Buffer 地址。下一阶段应先审计 I/O Binding / ping-pong cache buffer 是否能直接复用或消除副本，并逐项验证 KV 输出与 Token 一致性。
3. **频繁小 Kernel / 节点序列**：每 Token 约 523 launches，Concat/Split、GatherScatterElements、SkipLayerNorm、Transpose 都反复调用；可在不改变 cache 值的前提下验证固定绑定、CUDA Graph 约束解决方案或安全图级融合。Graph 当前被 ORT 拒绝，不能把该路径视作已验证优化。
4. **GEMV / GEMM Kernel 算术**：主 `gemvx` 约 424 µs/token，是单个最大 kernel 家族；但总体 Kernel busy 仅约 1.28 ms/token，先减少 launch/copy 比重的理论空间更大。
5. **常规 Rust 绑定与 CPU Sampling**：约 23 µs/token 和 0.7 µs/token，明显低于 CUDA 提交成本，维持现状。

下一轮前置条件：如果要精确归因 CPU Thread Scheduling，允许一次 Nsight Systems `--cpuctxsw=process-tree` 管理员运行；或者在 Nsight Systems UI 中以管理员身份启动相同 trace runner。若要将 ONNX 节点与 CUDA API 做一对一关联，另需匹配 CUDA 13.0 / cuDNN 9 且用 `--enable_cuda_profiling` 构建的 ORT。TensorRT 原型仍需兼容 SDK / EP 组合。
