# R2.2 — Native TensorRT GPT + CUDA VITS

## 结论

增加了一个显式启用的混合后端：GPT Encoder/Step 使用原生 TensorRT 11，VITS Streaming Session 使用 ONNX Runtime CUDA EP。默认 CPU 后端及既有 CPU、DirectML、CUDA EP 和原生 TensorRT 路径均未改为新策略。GPT Chunk 24、ABI-1 模型包、采样器和权重保持不变。

在 RTX 5070 Ti、同一模型包/参考音频/文本、Chunk 24、30 次预热 + 20 次计时样本下，与原生 TensorRT GPT + CPU VITS 相比：

- VITS 总耗时 P50 从 883.33ms 降至 364.71ms（-58.7%）；首个 VITS 调用从 111.17ms 降至 70.25ms（-36.8%）。
- TTFA P50/P95 从 182.79/188.84ms 降至 140.66/145.40ms，P50 改善 23.0%。
- 完整合成耗时 P50/P95 从 1365.49/1373.60ms 降至 827.13/854.40ms；RTF 从 0.20536 降至 0.12439，P50 改善 39.4%。
- 两组 20/20 样本的语义 Token 数和 Hash 一致。CUDA VITS 波形与 CPU VITS 有微小数值差异；数值/边界检查通过，但本轮没有人工盲听，因此不将主观音质 Gate 标记为通过。

因此该混合策略只作为实验选项，不改变默认路径。

## 实现范围

- 新增 `r2-trt-native-cuda-vits` Cargo feature，依赖 `r2-trt-native` 与 `r1-cuda-ep`。
- `Backend::TensorRtNative` 只有在编译了这两个路径且显式设置 `ALICE_NATIVE_TTS_CUDA_VITS=1` 时才将 VITS 交给 ORT CUDA EP；GPT 仍由 TensorRT 原生 C++ 实现执行。
- 示例 `r0_profile` 接受 `tensor_rt_native --cuda-vits`，并能同时记录 ORT VITS/Frontend Profile 与原生 GPT 分阶段计时。
- VITS CPU 辅助节点需要显式传入 `--cuda-vits-allow-cpu`，用于当前已审计的图形状/辅助节点。默认 CPU VITS 不受影响。
- 参考音频编码、BERT/G2PW 等 Frontend Session 仍由 ORT CPU 执行。没有迁移模型权重、改变采样或修改音频拼接语义。

## 真实执行位置

在一个无预热的诊断合成中，ORT CUDA EP trace 捕获了 7 次 VITS Streaming 调用。`sovits_stream` 共有 2,085 个节点：1,515 个节点由 CUDA EP 执行（10,605 次节点事件），570 个节点由 CPU EP 执行（3,990 次节点事件）。按本项目的主要数学算子分类（Gemm、MatMul、Conv、Attention、LayerNormalization 等），CPU EP 节点数为 0；CPU 节点主要是 Unsqueeze、Concat、Gather 一类辅助/形状路径。该计数来自 ORT trace，不是对 TensorRT GPT 的推断。

同一诊断 Trace 记录到 CUDA VITS Session 的 98 个 `MemcpyFromHost` 事件、合计约 372,330 字节/1.640ms，以及 14 个 `MemcpyToHost` 事件、合计约 186,039 字节/0.336ms。ORT 的传输事件反映输入/输出和 CPU 辅助节点边界；因此 VITS CUDA 路径不是“零拷贝”。它消除了主卷积/矩阵计算的 CPU 执行，但仍有可继续压缩的 Host/Device 交互。

Front-end 的这次未缓存诊断仍在 CPU：BERT Trace 有 632 个 CPU 节点，其中 199 个属于主要数学算子；G2PW 有 178 个 CPU 节点，其中 75 个属于主要数学算子。前端耗时不是本次 CUDA VITS A/B 的主要变量；冷 Frontend 仍是独立 TTFA 优化方向。

Profiling 运行使用 0 warmup + 1 次合成，仅用于节点位置与传输取证，不作为时延样本。下面的性能数据来自独立、无 ORT Trace 的 30+20 热态运行。

## 运行环境与复现

- GPU：NVIDIA GeForce RTX 5070 Ti 16GB，驱动 610.74。
- TensorRT：Windows CUDA 12.9 Enterprise SDK，11.2.1 build 2；CUDA Runtime 12.9.79；cuBLAS 12.9.1.4。
- ORT DLL：1.30.0.20260909.3.f2c39fe，CUDA EP 运行时依赖通过本机 DLL 搜索路径加载。
- TRT、CUDA/ cuBLAS redist、模型包、Profile 和输出样本位于被忽略的 `target/`；未安装全局 SDK、未改驱动，也未把模型权重加入仓库。

在已设置 R2.1 的 `ALICE_TRT_HOME`、`ALICE_CUDA_HOME`、`ALICE_CUDA_NVCC_HOME`、`ORT_DYLIB_PATH`，并把 TensorRT/CUDA/cuBLAS/ORT 和 CUDA EP 所需 DLL 目录加入 `PATH` 后运行：

```powershell
cargo build --release --features "r0-profiling r2-trt-native-cuda-vits" --example r0_profile
cargo run --release --features "r0-profiling r2-trt-native-cuda-vits" --example r0_profile -- `
  ./model-pack ./target/r2.2-bench tensor_rt_native `
  --chunks=24 --warmups=30 --repeats=20 --cuda-vits --cuda-vits-allow-cpu
```

节点跟踪使用独立输出目录，并追加 `--ort-profile`、`--warmups=0 --repeats=1`。不要将带 Trace 的数字与无插桩 Benchmark 混合。

## A/B 数据

同一中文长句、相同 v2Pro 包、参考音频、种子 1234、Chunk 24；每后端 30 次预热 + 20 次无插桩测量。毫秒指标为 P50/P95。

| 指标 | TRT GPT + CPU VITS | TRT GPT + CUDA VITS | P50 变化 |
|---|---:|---:|---:|
| GPT Encoder | 4.27 / 5.12ms | 4.15 / 4.41ms | -2.8% |
| GPT Step 总耗时 | 469.49 / 476.96ms | 453.33 / 462.25ms | -3.4% |
| GPT 每 Token | 2.662 / 3.751ms | 2.684 / 3.094ms | +0.8% |
| 首 Callback TTFA | 182.79 / 188.84ms | 140.66 / 145.40ms | -23.0% |
| 首个 VITS 调用 | 111.17 / 114.18ms | 70.25 / 73.07ms | -36.8% |
| VITS 总耗时 | 883.33 / 891.48ms | 364.71 / 383.13ms | -58.7% |
| 完整合成 | 1365.49 / 1373.60ms | 827.13 / 854.40ms | -39.4% |
| RTF | 0.20536 / 0.20658 | 0.12439 / 0.12849 | -39.4% |

GPT 行的轻微变化属于两次独立运行的波动，VITS 是本次被切换的执行后端。CUDA VITS 单次调用波动较大：首个调用 P95 为 73.07ms，而全部 VITS 调用时间分布的尾部仍受长调用影响；完整合成 P95 比 P50 多约 27ms。

CPU VITS 运行的 CUDA 整卡基线/峰值为 2,457/4,217MiB（增量约 1,760MiB）；这是 750ms 周期的整卡采样，不能隔离其他进程。混合运行进程在加载后 Private Bytes 为 4.65GB、Working Set 为 2.72GB；峰值增量约 0.97GB/0.80GB。CPU/进程内存不是显存指标。

## 数值、Token 与边界

- 20/20 样本均生成 167 个语义 Token；Token Hash 与 TRT GPT + CPU VITS 对照逐条一致。
- 对照保存的首条原始音频均为 212,777 个 f32 样本。CUDA VITS 的最大绝对差约 `4.87864e-5`、RMSE `1.8771e-6`、cosine similarity `0.999999999915229`。GPU 与 CPU 卷积计算具有浮点舍入差异，因此音频文件并非 bitwise identical。
- 7 个 Chunk 的数量和长度一致：`28160, 30438, 30631, 30451, 30637, 30470, 31990` 个采样点。6 个边界的跳变与 CPU VITS 对照近似相同，P50 约 `0.004604`、最大/P95 约 `0.024068`。
- 本轮未完成匿名人工盲听或实际播放 underrun 验收。数值相似和边界连续不能替代听感 Gate；生产默认保持不变。

## 当前瓶颈与后续

在此次 6.65 秒音频的混合运行中，GPT Step P50 约 453ms，VITS 总耗时约 365ms；GPT Step 仍略大于 VITS，冷 Frontend 另会增加首次延迟。Token Sampling 约 0.112ms/请求，不是当前总耗时瓶颈。ORT trace 中 CPU 辅助节点及其数据边界仍产生 Host/Device 传输。下一步适合独立验证的是降低 VITS 辅助节点复制和首次 ORT/CUDA 初始化开销，之后再做 GPU Sampling / 减少每 Token 同步；每项都应与现有基线单独 A/B。

人工盲听、DirectML E2E 及跨语言语料回归尚未完成。实验功能未默认开启，未自动提交或推送。
