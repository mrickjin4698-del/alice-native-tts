# R1.5 GPT 执行优化

测量日期：2026-10-26。冻结基线为 `880c394`（R1.4）；实验分支为 `perf/r1-5-gpt-execution`。模型、参考音频、采样种子和模型配置均保留本机，未加入仓库。

## 结论

- GPT Step 的主耗时仍是 ORT CUDA `RunWithBinding` 调用墙钟时间。短句组每 Token P50 为 5.453 ms；CPU Sampling 只有 0.0007 ms/Token。现有计时能建立真实逐 Token 时间线，但当前 ORT 构建没有 CUDA Kernel trace，不能把 ORT 调度、设备等待和实际 GPU Kernel 时间继续拆开。
- 实验性地把固定的 `x_len` / `y_len` 从逐 Token 重绑改为每请求绑定一次，GPT Step P50 降低约 0.9%，TTFA P50 降低约 1.1%。绝对收益很小，且单文本 20 个测量样本不足以证明收益稳定；开关保留为 opt-in，默认实现不变。
- 同一多音字长句的冷缓存 CUDA Frontend 将 Frontend P50 从 455.9 ms 降至 25.4 ms，TTFA P50 从 680.0 ms 降至 235.5 ms；完整耗时 P50 降低约 18.5%。两路产生相同的语义 Token 数与哈希。音频数值差异很小，但人工盲听仍未完成。
- TensorRT 原生执行没有可运行的本机 SDK。最小 ORT TensorRT EP Add 原型在 provider 加载阶段因缺少 `nvinfer_10.dll` 失败；没有创建 TensorRT Session，也没有 GPT Step 数字。
- 没有改变默认后端、Chunk 24、公开 API、模型包 ABI 或模型图；没有提交或推送 R1.5。

## 环境与方法

| 项目 | 配置 |
|---|---|
| 系统 / GPU | Windows x64 / NVIDIA GeForce RTX 5070 Ti，16 GiB，驱动 610.74 |
| CPU | Intel Core Ultra 9 285K |
| ONNX Runtime | 1.30.0，Rust `ort` 2.0.0-rc.12，动态加载 |
| GPU Runtime | CUDA 13.0 / cuDNN 9.12 |
| 模型 | 本机 ABI-1 GPT-SoVITS v2Pro 包，参考音频固定，未改权重和配置 |
| 采样 / Chunk | seed 1234，Chunk 24，CPU Token Sampling |
| 计时 | 每组 30 次预热 + 20 次测量；TTFA 是开始请求至首次音频 Callback；RTF 为全程墙钟耗时 / 音频时长 |

AnifLive-TTS 是本轮明确采用的参考项目。其仓库记录了 TensorRT 11 `execute_async_v3`、持久化 GPT Context、固定 KV Buffer、保持 RNG 语义的 GPU Sampling Graph，以及每两步检查 EOS 等实现思路；它的模型、FP16、TensorRT/CUDA 版本和负载与本项目不同，因此其延迟数字不作本项目基准。[AnifLive-TTS README](https://github.com/Hiruynk/AnifLive-TTS)

## 逐 Token 计时与瓶颈

以下短句组使用固定 CUDA Frontend 配置及默认逐 Token 清空 / 重绑路径；每次运行生成 167 个语义 Token。R1.1 的节点 Trace 已验证 GPT Step 的 864 个图节点均分配给 CUDA，K/V Cache 通过两个交替 I/O Binding 保持设备端；R1.5 没有改写 ONNX 图。

| 项目 | P50 | P95 | 范围 |
|---|---:|---:|---|
| GPT Step 单 Token 墙钟时间 | 5.453 ms | 6.519 ms | 3,340 个 Token Step 池化 |
| GPT Step 全部 Token | 927.1 ms | 956.7 ms | 每次请求 |
| ORT `RunWithBinding` | 922.9 ms | 1,009.7 ms | 每次请求 |
| 输入 Tensor 准备 | 0.337 ms | 0.494 ms | 每次请求；含每步 CPU `samples` / `idx` 输入创建 |
| I/O Binding 更新 | 4.056 ms | 4.808 ms | 每次请求 |
| 输出检查 | 0.150 ms | 0.222 ms | 每次请求 |
| CPU Sampling | 0.0007 ms | 0.0012 ms | 每 Token；总计 0.129 / 0.140 ms 每请求 |
| VITS | 361.3 ms | 375.3 ms | 每次请求 |
| TTFA | 209.7 ms | 216.7 ms | 每次请求 |
| 完整耗时 / RTF | 1,301.0 ms / 0.1957 | 1,344.3 ms / 0.2022 | 每次请求 |

`RunWithBinding` 是 Rust 侧观察到的 ORT 调用墙钟时间，不等于 GPU Kernel 时间。现有 ORT GPT Trace 记录到 Node 事件但没有 CUDA Kernel 事件；ORT 文档要求 CUPTI 与以 `--enable_cuda_profiling` 构建的 ORT 才能采集 CUDA Kernel 时间。[ORT Profiling 文档](https://onnxruntime.ai/docs/performance/tune-performance/profiling-tools.html) 当前机器没有 Nsight Systems / Nsight Compute，故 Kernel 与 ORT 调度 / 隐式等待的细分为 **BLOCKED**。热态基准不启用高开销 ORT Trace。

同步探针在每次 `RunWithBinding` 后额外调用一次 `synchronize_outputs()`。新增等待 P50/P95 为 0.0066 / 0.0170 ms 每 Token（1.214 / 1.627 ms 每整次合成）。它只计量额外栅栏；正常路径的 CPU Sampling 仍要求 ORT 返回可读输出，所以这不能单独量化常规运行中包含的隐式等待，也没有证据支持移除该同步边界。

### 固定输入绑定 A/B

两组均使用同一短句、模型、参考音频、seed、Chunk、CUDA GPT/VITS 和 30/20 方法；差别只有固定 `x_len` / `y_len` 的绑定时机。

| 指标 | 默认清空并逐 Token 重绑 | 固定输入每请求绑定 | 变化 |
|---|---:|---:|---:|
| GPT Step Token P50 / P95 | 5.453 / 6.519 ms | 5.368 / 6.419 ms | -1.6% / -1.5% |
| GPT Step 总 P50 / P95 | 927.1 / 956.7 ms | 918.8 / 930.0 ms | -0.9% / -2.8% |
| TTFA P50 / P95 | 209.7 / 216.7 ms | 207.3 / 212.9 ms | -1.1% / -1.7% |
| 完整耗时 P50 / P95 | 1,301.0 / 1,344.3 ms | 1,291.3 / 1,308.8 ms | -0.7% / -2.6% |
| 绑定更新 P50 / Token | 22.9 µs | 15.1 µs | -34%，绝对约 7.8 µs |

候选路径每请求绑定 `x_len` / `y_len` 的 P50 为 0.040 ms。20/20 样本语义 Token 数为 167，语义 Token FNV-1a 64-bit 哈希相同；首个完整音频均为 212,777 samples。首样本波形的最大绝对差为 `2.38e-7`、MAE `9.29e-9`、RMSE `1.69e-8`、余弦相似度 `0.99999999999995`。CUDA VITS 重复运行会产生极小数值差异，不能要求 PCM 字节级相同。

该候选没有足够稳定的收益证据，继续通过 `ALICE_NATIVE_TTS_GPT_STEP_BINDING_MODE=reuse_static` 实验性启用；默认仍走原有清空 / 重绑路径。两个绑定模式的成功合成均到达 EOS，输出长度与 Token 哈希相同。`reuse_static` 的 CUDA 流式回归在首包后取消，恰有一次 Callback，返回取消并可完成下一条请求。

Session 与两个 I/O Binding 本来就在模型实例中跨请求持久化，K/V 已经采用设备端 ping-pong；本阶段没有新建持久 Context。仍逐 Token 创建 `samples` / `idx` CPU Tensor，计入 `GptStepInput`。后续可尝试固定地址的小型输入缓冲，但应先拿到 Kernel Trace，避免用 Rust 侧小开销掩盖主要瓶颈。ORT I/O Binding 适用于固定地址的设备输入输出，但当前 GPT Step 同时含动态 Token 和 CPU Sampling 边界。[ORT I/O Binding 文档](https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html)

## CUDA Frontend 冷缓存 A/B 与音频验证

测试句固定为“银行行长在重庆重看那段长长的乐章，朋友们还在行走；后来他着重介绍了重要的行程。”两组的 Frontend 缓存均强制每次 miss，G2PW 与 BERT 使用同一 Backend；GPT/VITS 都使用 CUDA，其他配置和采样语义不变。

| 指标 | CPU Frontend | CUDA Frontend | CUDA 变化 |
|---|---:|---:|---:|
| Frontend P50 / P95 | 455.9 / 490.6 ms | 25.4 / 28.3 ms | -94.4% |
| G2PW P50 / P95 | 391.4 / 421.3 ms | 18.8 / 21.9 ms | -95.2% |
| BERT P50 / P95 | 65.4 / 71.8 ms | 5.9 / 7.3 ms | -90.9% |
| TTFA P50 / P95 | 680.0 / 716.0 ms | 235.5 / 257.4 ms | -65.4% / -64.0% |
| 完整耗时 P50 / P95 | 2,214.3 / 2,334.4 ms | 1,804.4 / 1,962.7 ms | -18.5% / -15.9% |
| RTF P50 / P95 | 0.2515 / 0.2652 | 0.2050 / 0.2230 | -18.5% / -15.9% |
| 每 Token GPT Step P50 / P95 | 5.397 / 7.323 ms | 5.546 / 7.377 ms | 相近；未显示 Frontend 影响 GPT Step |

两组 20/20 合成都产生 221 个 Token，语义 Token hash 都为 `7023250444603240907`；成功返回证明生成在相同位置达到 EOS。10 个 Callback 的 Token 边界和音频 sample 数完全相同。完整首样本波形有 281,689 个 sample，最大绝对差 `2.38e-7`、MAE `8.79e-9`、RMSE `1.62e-8`、余弦相似度 `0.99999988`。数值相近与语义序列相同不能代替主观试听。

模型整体加载时间为 CUDA Frontend 5,549 ms、CPU Frontend 6,565 ms；这是完整 `NativeTts::load` 时间，不是单独 Frontend 初始化计时。模型加载后工作集约为 866 MiB / 2,519 MiB；测量期间峰值约为 1,714 MiB / 3,343 MiB（CUDA / CPU）。`nvidia-smi` 全局显存使用在 CUDA 组加载前后为 2,391→7,107 MiB，CPU 组为 2,598→4,351 MiB。WDDM 下没有进程级显存归属，因此两组全局差分只能估算，不能声称是精确的 CUDA Frontend 显存增量；当前数据暗示 CUDA Frontend 比 CPU Frontend 多占约 2.9 GiB GPU 显存，同时 CPU 工作集明显更低。

R1.4 的 ORT 节点 Trace 已记录 CUDA Frontend 主要计算实际执行在 GPU：BERT 630 个 CUDA 节点、2 个 CPU 辅助节点；G2PW 344 个 CUDA 节点、14 个 CPU 辅助节点。此基准仍显式开启诊断 CPU helper fallback，不将其描述为全节点 CUDA。

匿名盲听包已生成，映射文件不在压缩包中：

- [盲听包 ZIP](../target/r1.5-frontend-cold-blind/listening-pack.zip)
- [盲听说明](../target/r1.5-frontend-cold-blind/listening-instructions.txt)
- 本地映射：`target/r1.5-frontend-cold-blind/listening-key.local.txt`（盲听完成前不要打开）

**人工盲听与实际播放设备 underrun 检查尚未完成**。本项目没有播放设备回调接口；当前测量不能把 Callback 到达等同于可听播放。

## TensorRT、DirectML 与回归

- 已编译最小 ORT TensorRT EP Add smoke。运行时在 provider 加载处失败：`onnxruntime_providers_tensorrt.dll` 依赖的 `nvinfer_10.dll` 缺失；本机也没有 `trtexec` / TensorRT SDK。没有创建 TensorRT Session，没有原生 TensorRT GPT Step 原型结果。ORT 当前兼容表列出 `main` / `1.22` 对应 TensorRT 10.9 + CUDA 12.0–12.8，没有确立本机 ORT 1.30 + CUDA 13 的 TensorRT 组合。[ORT TensorRT EP 矩阵](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- AnifLive 的持久 Context、固定 KV Buffer、GPU Sampling Graph 和批量 EOS 是值得后续参考的候选，但它使用 TensorRT 11.2.1.2、CUDA 12.8、FP16 和独立模型包；不能将其公开 benchmark 直接归因于本项目。[AnifLive 执行器说明](https://github.com/Hiruynk/AnifLive-TTS)
- CPU 后端使用同一公开 `stream_bench` 跑完中文、混合、英文、长句和短句；退出码为 0，取消后恢复通过。CUDA `reuse_static` 路径同样跑完五组并通过首包取消 / 后续恢复。
- DirectML：当前 ORT 1.30 运行时缺少 `onnxruntime_providers_dml.dll`，返回 `Specified provider is not supported.`；R1.5 未把 DirectML 计作通过，状态仍为 **BLOCKED**。

## 验证记录

- `cargo fmt --all -- --check`、`git diff --check`、默认 `cargo check --release --lib`：通过。
- `cargo test --release --lib --features r1-cuda-ep`：15 通过、0 失败、3 忽略（均需模型包 / 对照资源）。动态 ORT 配置解决了测试链接。
- `cargo check --release --features 'r0-profiling r1-cuda-ep' --example r1_5_profile`、`cargo check --release --features r1-tensorrt-ep --example r1_5_trt_smoke`：通过。
- 默认静态链接的 `cargo test --release --lib` 在链接阶段因缺少 `onnxruntime.lib` / `OrtGetApiBase` 失败；改用本机 ORT 动态运行时的测试配置后，上述单元测试通过。
- 真实 CUDA 与 CPU `stream_bench` 回归均通过。最小 TensorRT EP Add 运行在加载 provider 时失败于缺少 `nvinfer_10.dll`；这是本机 Runtime 缺失，不以编译通过代替运行验收。

## 修改与下一阶段

修改文件：

- `src/inference.rs`：增加逐 Token ORT 调用、绑定更新、输出检查与显式同步探针；加入 `reuse_static` opt-in 实验模式。
- `src/profiling.rs`：增加相应阶段及 Token 时间线事件。
- `examples/r1_5_profile.rs`：增加 30/20 基准、CPU/CUDA Frontend、cache hit/miss、文本选择和音频/内存快照。
- `examples/r1_5_trt_smoke.rs`、`Cargo.toml`：增加 TensorRT EP 最小 smoke 示例。
- `docs/performance-r1.5.md`：记录方法、结果、失败项和决策。

建议下一阶段先取得带 CUPTI CUDA Kernel Trace 的 ORT 构建或受支持的 Nsight 采集环境，再分离 GPU Kernel、Launch/ORT 调度与同步成本；TensorRT 需先提供和 ORT/CUDA 相容的原生 SDK，并用 ABI-1 图验证 engine build / enqueue。GPU Sampling、EOS 批量检查、`samples` / `idx` 固定地址缓冲应独立 A/B；在 Token/RNG 与质量 Gate 完成前不替换 CPU Sampling。
