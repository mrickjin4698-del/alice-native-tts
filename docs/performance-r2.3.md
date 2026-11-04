# R2.3 — 冷中文文本 Frontend

## 结论

在当前 Native TensorRT GPT + CUDA VITS 路径上，Frontend 默认仍使用 CPU。R2.3 将 G2PW + BERT CUDA 接入 Native TensorRT 配置，并保留显式 opt-in。

为了与 R1.1/R2.1/R2.2 的历史结果对齐，本轮新增固定 167-token 中文文本基线（2026-10-27）：同一 RTX 5070 Ti、本地 v2Pro 包、参考音频、seed 1234、Chunk 24、Native TensorRT GPT + CUDA VITS。CPU Frontend 与 CUDA Frontend 各做 30 次预热 + 20 次计时，每次请求强制产生 Frontend cache miss。这里的“冷”指新文本缓存未命中；模型和 ORT Session 已在预热阶段加载，不代表冷进程或冷 Session。

- 167-token 组的 Frontend P50/P95 从 CPU miss 的 139.71/142.12ms 降至 CUDA miss 的 13.06/14.21ms，P50 降低 90.6%。
- TTFA P50/P95 从 280.71/287.98ms 降至 156.17/160.20ms；完整合成 P50/P95 从 981.33/995.31ms 降至 867.01/881.00ms；RTF P50/P95 从 0.14758/0.14969 降至 0.13039/0.13249。
- GPT Step 和 VITS 在两组之间基本持平，主要收益来自冷 Frontend：GPT Step P50 为 462.83/469.96ms，VITS P50 为 366.78/371.70ms。它们约 1% 的差异与两次独立运行的波动同量级。
- 20/20 样本均生成 167 个相同语义 Token，Hash 为 `15228212952513995076`。PCM 均为 212,777 个采样点，7 个 Chunk 长度相同；最大绝对差 `2.3841858e-7`、RMSE `1.7195e-8`、cosine `0.999999999999928`。

原有中文多音长句 221-token 的冷 Frontend miss 结果也保留在下文单独报告。由于文本和音频长度不同，不应将它与 167-token 基线直接混为一组。

CUDA Frontend 现通过显式 `ALICE_NATIVE_TTS_CUDA_FRONTEND=1` 启用；CPU/DirectML 和 Native TensorRT 默认配置不变。为了运行当前导出图，还需明确允许审计过的 CPU shape/helper 节点。生产级盲听、真实播放缓冲及跨文本/跨硬件覆盖仍未完成，因此保留为 opt-in。

## 修改

- `src/inference.rs`：增加 CUDA Frontend 高层选择开关。启用时只把 CUDA-capable 后端的 Frontend 默认 EP 设为 CUDA，不改变参考音频编码、GPT、VITS 或默认路径。CPU/DirectML 后端忽略这个高层开关并保留原 Frontend 路径；Native TensorRT 若未编入 CUDA EP 则明确返回缺少 feature 的错误。
- `examples/r0_profile.rs`：增加 `--text=...`、`--frontend=cpu|cuda`、`--frontend-allow-cpu` 和 `--frontend-cache-miss`；输出记录选择的 G2PW/BERT 后端及缓存 miss 状态。
- `README.md`：说明 Native TensorRT/ CUDA 后端的 CUDA Frontend opt-in 方法。
- `docs/performance-r2.3.md`：记录基准、执行位置、数值结果与未完成的质量 Gate。

公开 Rust API 签名、ABI-1 模型包、模型权重、语义采样、Chunk 24、取消检查及音频流行为都未改变。新增选择仅由进程环境开关显式启用；CUDA Session 仍默认拒绝未审计的 CPU 回退。每个 Frontend Cache Key 仍包含语言路由、G2PW/BERT 后端、G2PW batch 与 schema。

## 167-token 对照基线

固定文本：

> 今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。

每格为 20 次计时样本的 P50/P95。两组均为 Native TensorRT GPT + CUDA VITS、Chunk 24、seed 1234、强制 Frontend cache miss；未启用 ORT operator trace。

| 阶段/指标 | CPU Frontend miss | CUDA Frontend miss | CUDA 相对 CPU 的 P50 变化 |
|---|---:|---:|---:|
| Frontend | 139.71 / 142.12ms | 13.06 / 14.21ms | -90.6% |
| GPT Encoder | 4.20 / 5.17ms | 4.15 / 5.14ms | -1.3% |
| GPT Step 总耗时 | 462.83 / 469.54ms | 469.96 / 477.14ms | +1.5% |
| GPT Step 每 Token | 2.653 / 3.644ms | 2.697 / 3.668ms | +1.6% |
| 首个 VITS 调用 | 68.62 / 74.51ms | 71.60 / 73.35ms | +4.3% |
| VITS 总耗时 | 366.78 / 381.97ms | 371.70 / 382.70ms | +1.3% |
| TTFA | 280.71 / 287.98ms | 156.17 / 160.20ms | -44.4% |
| 完整合成 | 981.33 / 995.31ms | 867.01 / 881.00ms | -11.7% |
| RTF | 0.14758 / 0.14969 | 0.13039 / 0.13249 | -11.7% |

Frontend P50 节省约 126.65ms，TTFA P50 缩短约 124.54ms；两者吻合，说明首包收益基本来自 Frontend。完整耗时减少约 114.32ms；两组 GPT/VITS 的轻微差异抵消了约 12ms 的 Frontend 收益。与 R2.2 的 TTFA P50 140.66ms、完整耗时 P50 827.13ms 相比，本轮 CUDA Frontend cache-miss 分别多约 15.5ms 和 39.9ms；R2.2 的 Frontend 是缓存命中条件，不能直接视为相同冷 miss 测量。

进程主机内存采样（不含可归属到进程的 VRAM 计数）：CPU Frontend 组加载后 private bytes / working set 为 4.65 / 2.72GB，峰值为 5.58 / 3.48GB；CUDA Frontend 组加载后为 6.17 / 0.99GB，峰值为 7.15 / 1.78GB。CUDA Frontend 组的 private bytes 峰值较 CPU 组高约 1.57GB；Windows working set 数值受驱动和映射内存影响，不应当作显存用量。

两组 JSON、PCM 和每样本记录保存在忽略目录 `target/r2.3-frontend-167-cpu/` 与 `target/r2.3-frontend-167-cuda/`。

### 同一 167-token 测试文本的历史参照

下表沿用各阶段报告中的 P50/P95。它展示整体演进，但不是完全同条件的单变量 A/B：R1 到 R2 更换了 GPT/VITS 执行后端，R2.1/R2.2 使用 Frontend 缓存命中，而本轮 R2.3 两组强制冷 miss。

| 阶段/执行路径 | TTFA ms | GPT Step ms | 完整合成 ms | RTF | Frontend 状态 |
|---|---:|---:|---:|---:|---|
| R1：ORT CPU GPT + CPU VITS，8 线程 | 623 / 638 | 2,176 / 2,209 | 3,265 / 3,296 | 0.491 / 0.496 | 旧报告未单列缓存条件 |
| R1.1：ORT CUDA GPT + CPU VITS | 387 / 400 | 914 / 933 | 1,913 / 1,935 | 0.288 / 0.291 | Frontend CPU，约 137ms |
| R2.1：TensorRT GPT + CPU VITS | 182.79 / 188.84 | 469.49 / 476.96 | 1,365.49 / 1,373.60 | 0.2054 / 0.2066 | 命中 |
| R2.2：TensorRT GPT + CUDA VITS | 140.66 / 145.40 | 453.33 / 462.25 | 827.13 / 854.40 | 0.12439 / 0.12849 | CPU Frontend 命中，约 0.04ms |
| R2.3：TensorRT GPT + CUDA VITS + CPU Frontend | 280.71 / 287.98 | 462.83 / 469.54 | 981.33 / 995.31 | 0.14758 / 0.14969 | 强制 miss |
| R2.3：TensorRT GPT + CUDA VITS + CUDA Frontend | 156.17 / 160.20 | 469.96 / 477.14 | 867.01 / 881.00 | 0.13039 / 0.13249 | 强制 miss |

R1.4 使用不同的八组语料并另测首包 Chunk 20/24，故其 TTFA 数据不放进这张固定 167-token 表。R1.4 的独立结果和语料条件保留在 [performance-r1.4.md](performance-r1.4.md)。

## 221-token 多音长句结果

同一 RTX 5070 Ti、本地 v2Pro 包、参考音频、中文多音长句、seed 1234、Chunk 24。两组都先做 30 次预热，再测 20 次；每次请求都强制清空 Frontend 缓存并记录一次 miss。此组同样测的是新文本缓存未命中，不是冷进程/冷 Session 首次运行。

- Frontend P50/P95：CPU 352.16/380.91ms，CUDA 25.74/27.16ms，P50 降低 92.7%。
- TTFA P50/P95：CPU 496.62/575.89ms，CUDA 168.98/181.34ms，P50 降低 66.0%。
- 完整合成耗时 P50：1494.53ms 降到 1169.21ms；RTF P50：0.16978 降到 0.13282。
- G2PW 与 BERT 主计算都实际分配到 CUDA EP；CPU EP 上未发现主要矩阵、注意力、卷积或归一化算子。
- 20/20 样本生成 221 个相同语义 Token，Hash 一致。两组 Chunk 边界样本数相同；音频最大绝对差约 `2.38e-7`，RMSE `1.64e-8`，cosine `0.999999999999949`。

每项均为毫秒 P50/P95；各组相同测试句：

> 银行行长在重庆重看那段长长的乐章，朋友们还在行走；后来他着重介绍了重要的行程。

| 阶段/指标 | TRT GPT + CUDA VITS + CPU Frontend | TRT GPT + CUDA VITS + CUDA Frontend | P50 变化 |
|---|---:|---:|---:|
| G2PW 推理 | 294.65 / 318.17 | 19.47 / 20.46 | -93.4% |
| BERT 推理 | 57.44 / 62.24 | 5.88 / 7.07 | -89.8% |
| 完整 Frontend | 352.16 / 380.91 | 25.74 / 27.16 | -92.7% |
| TTFA | 496.62 / 575.89 | 168.98 / 181.34 | -66.0% |
| GPT Encoder | 4.23 / 4.89 | 4.15 / 4.46 | 基本相同 |
| GPT Step 总耗时 | 617.28 / 1228.91 | 620.38 / 641.57 | P50 基本相同 |
| VITS 总耗时 | 509.70 / 635.77 | 508.49 / 532.27 | 基本相同 |
| 完整合成 | 1494.53 / 2483.96 | 1169.21 / 1211.20 | P50 -21.8% |
| RTF | 0.16978 / 0.28218 | 0.13282 / 0.13759 | P50 -21.8% |

两个后端的 GPT Step 和 VITS P50 几乎相同，主要收益来自 Frontend。CPU 对照的 GPT Step 与完整耗时 P95 有明显长尾，GPU 对照没有同等幅度长尾；P95 变化不能全部归因于 Frontend。

样本均使用固定文本但每个 warmup/测量调用都清空精确文本缓存，因此 20 个计时样本均是 Cache Miss，不代表 20 种不同句子的覆盖。R1.4/R1.5 的其它 CPU/CUDA 前端样本和不同句子证据见对应报告。

## ORT 执行位置

单独 ORT Profile 运行用于节点分配检查，启用 Trace 的时延不纳入上述 30+20 无插桩样本：

| 图 | CUDA EP 节点 | CPU EP 节点 | CPU 主要数学节点 |
|---|---:|---:|---:|
| BERT | 630 | 2 | 0 |
| G2PW | 172 | 7 | 0 |

主要 CPU 节点是索引/形状辅助节点；ORT Trace 主要 CUDA 运算包含 BERT 的 MatMul、Softmax、LayerNormalization、BiasGelu，以及 G2PW 的 Attention、MatMul、Gemm。G2PW 诊断 Trace 记录 6 个 `MemcpyFromHost` 事件，合计 48 字节、约 0.091ms；这些是单次 Profiling 调用的 trace 事件，不代表 CUDA Frontend 没有其他隐式传输。

本次完整合成使用 CUDA VITS helper fallback，VITS CPU/CUDA 主节点分布与 R2.2 保持一致：1,515 CUDA / 570 CPU 辅助节点。CUDA Frontend 节点数、CUDA VITS 节点数是分别从对应 Session 的 ORT trace 统计的；GPT Encoder/Step 由 TensorRT 原生 Engine 执行，不出现在 ORT Node Profile 中。

## 数值、输出与资源边界

- 167-token 对照的 CPU/CUDA Frontend 各自 20/20 测量样本均生成 167 个语义 Token，semantic token hash 均为 `15228212952513995076`。两条首测音频均有 212,777 个 f32 样本，7 个 Chunk 的样本长度一致：`28160, 30438, 30631, 30451, 30637, 30470, 31990`。两组波形有微小浮点差异，最大绝对差 `2.3841858e-7`、RMSE `1.7195e-8`、cosine `0.999999999999928`。
- 221-token 多音长句对照的 CPU/CUDA Frontend 各自 20/20 测量样本均生成 221 个语义 Token，semantic token hash 均为 `7023250444603240907`。保存的两条首测音频均有 281,689 个 f32 样本，PCM 最大绝对差 `2.3841858e-7`、RMSE `1.6386808e-8`、cosine `0.9999999999999486`；10 个 Chunk 长度相同：`28160, 30646, 30679, 30572, 30498, 30621, 30400, 30702, 30456, 8955`。
- 尚未完成匿名盲听和实际播放器 32/64ms 缓冲 underrun 测试。Token 与波形数值对照是客观检查，不替代主观听感和真实播放连续性验收。
- 167-token 新基线采集了进程 private bytes 和 working set，但没有独立的 VRAM 差分计数。221-token 旧组没有该组进程内存采样；R1.5 的 WDDM 整卡采样暗示 CUDA Frontend 有明显显存成本，不能当作 R2.3 的精确 VRAM 测量。
- 实测运行时仍使用本机已验证的 Windows ORT 1.30 CUDA EP + CUDA 13/cuDNN DLL，与原生 TensorRT Enterprise CUDA 12.9 组合。不同 ORT/CUDA/TensorRT 包需重新验证；没有引入 Python 服务。

## 验证与后续

- 167-token 对齐基线和 221-token 多音长句基线均完成 30 warmup + 20 samples、无 ORT Trace 的 CPU/CUDA Frontend A/B。167-token 原始 JSON、PCM、20 条样本记录及内存摘要在 `target/r2.3-frontend-167-cpu/` 与 `target/r2.3-frontend-167-cuda/`；221-token 数据在 `target/r2.3-frontend-cold-cpu/` 与 `target/r2.3-frontend-cold-cuda/`。
- ORT Trace：完成，证明 G2PW/BERT 的主要计算由 CUDA EP 执行。Profile 产物保存在 `target/r2.3-cuda-frontend-profile/`。
- 30+20 A/B 通过每图 EP 覆盖选择 CPU/CUDA；随后通过高层 `ALICE_NATIVE_TTS_CUDA_FRONTEND=1` 做 Native TensorRT + CUDA VITS + CUDA Frontend E2E smoke（1 warmup + 1 sample）：Frontend 25.97ms、TTFA 168.82ms、221 Token，Hash 与基准相同。
- 将高层开关设为 1、但选择 CPU Backend 的单样本 smoke 仍实际使用 CPU Frontend 并成功合成。DirectML 后端保留自己的 Frontend 选择；DirectML 本机运行时缺 provider，未做 E2E 回归。
- CUDA Session 对 CPU 辅助节点继续要求显式的 `ALICE_NATIVE_TTS_FRONTEND_CUDA_DIAGNOSTIC_ALLOW_CPU=1`。
- 人工盲听、实际播放缓冲、更多中英文混合语料、独立显存差分和 DirectML E2E 尚未完成。

下一步应保留 CUDA Frontend 为可选路径，在多句冷文本与目标播放器上完成质量和显存 Gate；若合格再由宿主应用按设备能力选择。当前冷文本 Frontend miss 的参考值为：167-token 固定文本约 13ms P50，221-token 多音长句约 26ms P50；后续 GPT/VITS 继续按 R2.2 路径执行。
