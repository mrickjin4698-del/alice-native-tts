# alice-native-tts R1.8 — Early Preview & Refill

测量日期：2026-10-26。分支 perf/r1-5-gpt-execution，HEAD 880c394。保留此前工作；本轮没有提交或推送。

## 结论

- ABI-1 的 VITS Streaming 图可以接收完整且不断增长的语义前缀，并通过动态输出只解码新增段和 overlap；参考频谱、说话人向量及文本条件在单次合成中保持不变。结构上可以进行 Preview→Refill。
- 单句探针中，17/20 Token Preview 分别把首次 Callback P50 从 200ms 降到 161/182ms，首个 Refill 分别在 265/270ms 到达；首个 SOLA 相关性为 0.805/0.753。输出有限、Token 序列相同，音频长度与 Chunk 24 相差不到 4ms。
- 但 Preview 首 Callback 波形与 Chunk 24 的公共采样前缀差异很大；长句 cosine 为 -0.075/0.361。八组测量中 60/160 个首个 Preview→Refill 接缝触发低相关回退，其中英文长句两种阈值均为 20/20。部分边界最大跳变也高于 Chunk 24。
- 全量基准显示首 Callback 更早，但完整合成中位耗时增加约 8.6–8.8%，RTF 变差。32/64ms 离线队列模拟没有 underrun，但这不是实际播放设备验证。
- **质量 Gate 未通过：Preview 仅保留在显式 feature-gated 实验入口，默认 Chunk 24、公开 API、ABI-1 图、模型权重及 CUDA GPT/VITS 路径均未改变。人工盲听和真实播放尚未进行。** 流式 Callback 已交付 Preview 后无法撤回；低相关时的零偏移 Crossfade 只是诊断候选，不能当作已验证的安全回退。现有图结构没有阻塞，未解决的是声学前缀变化和接缝无法稳定对齐。

## ABI-1 图与单句探针

ABI-1 sovits_stream 每次接收完整 pred_semantic，result_length 描述本次新增段及 overlap；图按动态长度输出 latent 和 PCM。Rust 解码器缓存上次 latent 与 2 个语义 Token 的尾部（80ms），保留同一 refer_spec、sv_emb 和 text_seq，再对 overlap 做最多 320 个样本（10ms）的 SOLA 搜索和 Hann Crossfade。因此，增加一次 17/20→24 Token 解码不要求改模型包或 ABI。

这只证明动态形状和调用顺序可工作，不代表短语义前缀生成的声学内容与较长前缀等价。R1.8 不改变 noise、采样或模型条件。单句探针使用同一中文长句、seed 1234、Chunk 24 后续步长和 CUDA GPT/VITS；它先于八组测量运行。

| 策略 | 首 Callback | 首个 Refill / 下一 Callback | 首段音频长度 | Refill 接缝 |
|---|---:|---:|---:|---|
| Chunk 24 | 200ms | 394ms | 880ms | 首个常规边界最大跳变 0.0241 |
| Preview 17→24 | 161ms | 265ms | 600ms | offset 106 samples（3.31ms），相关性 0.805，边界跳变 0.00209 |
| Preview 20→24 | 182ms | 270ms | 720ms | offset 131 samples（4.09ms），相关性 0.753，边界跳变 0.00451 |

三种策略产生相同的 167 个语义 Token，样本均有限；最终样本数分别为 212,777、212,681、212,656。Preview 17/20 首 Callback 与 Chunk 24 的同起点公共采样前缀 cosine 分别为 -0.075/0.361，RMSE 为 0.266/0.210；这说明波形不是数值等价，不能据此通过音质 Gate。探针稍后的一个接缝相关性为 0.149，触发低相关回退。全量测量仅用于描述性能和接缝失败分布，不代表单句质量 Gate 获得批准。

## 环境与方法

| 项目 | 配置 |
|---|---|
| 系统 / GPU | Windows x64 / NVIDIA GeForce RTX 5070 Ti 16 GiB / 驱动 610.74 |
| ORT / Rust Binding | ONNX Runtime 1.30.0 / ort 2.0.0-rc.12 |
| CUDA / cuDNN | CUDA 13.0 / cuDNN 9.12 |
| 运行路径 | CUDA GPT + CUDA VITS、CPU Token Sampling、ABI-1 |
| 模型与输入 | 本机 v2Pro 模型包、参考音频及配置固定；seed 1234 |
| 策略 | Chunk 24 对照；17→24；20→24；后续步长均为 24 |
| 语料 | 8 组：中文/英文/混合语言，短句、长句及中文多音字 |
| 基准 | 每个语料/策略 30 次预热 + 20 次无插桩测量，共 720 次预热和 480 次测量 |

每个语料内三种策略的 Token hash 在 20 次测量中均固定且相同。音频活动定义为非重叠 10ms 窗口 RMS ≥ 1e-4；Callback TTFA 是实际回调时刻，首个活动 PCM 是从首段 PCM 检出的信号起点，不是扬声器端实测可听时间。P50/P95 按每个语料的 20 次测量计算。

## 首 Callback TTFA

表格单元为 P50 / P95，单位 ms。

| 语料 | Chunk 24 | Preview 17→24 | Preview 20→24 |
|---|---:|---:|---:|
| zh_short | 201.7 / 207.7 | 164.6 / 166.5 | 180.9 / 184.1 |
| zh_poly_short | 210.9 / 226.2 | 169.1 / 172.6 | 189.5 / 191.8 |
| zh_long | 206.7 / 209.3 | 167.8 / 172.4 | 184.3 / 187.0 |
| zh_poly_long | 209.4 / 211.7 | 167.6 / 175.0 | 184.2 / 186.4 |
| en_short | 203.8 / 209.9 | 169.6 / 172.5 | 182.8 / 187.1 |
| en_long | 207.0 / 209.6 | 167.2 / 171.1 | 185.8 / 188.4 |
| mixed_short | 206.5 / 209.4 | 168.4 / 171.7 | 184.3 / 190.6 |
| mixed_long | 209.2 / 218.3 | 169.6 / 174.4 | 185.9 / 193.3 |

按 8 组中位数等权比较，Preview 17 平均提前 38.9ms（18.8%），Preview 20 提前 22.2ms（10.7%）。这只衡量首 Callback。

## 首个活动 PCM 与后续段到达

单元格为 P50 / P95，单位 ms。首个活动 PCM 仍是离线信号门限指标；实际播放起点还需加播放设备缓冲。

| 语料 | Chunk 24 | Preview 17→24 | Preview 20→24 |
|---|---:|---:|---:|
| zh_short | 201.7 / 207.7 | 164.6 / 166.5 | 180.9 / 184.1 |
| zh_poly_short | 210.9 / 226.2 | 169.1 / 172.6 | 189.5 / 191.8 |
| zh_long | 216.7 / 219.3 | 177.8 / 182.4 | 194.3 / 197.0 |
| zh_poly_long | 459.4 / 461.7 | 407.6 / 415.0 | 424.2 / 426.4 |
| en_short | 203.8 / 209.9 | 169.6 / 172.5 | 182.8 / 187.1 |
| en_long | 547.0 / 549.6 | 507.2 / 511.1 | 525.8 / 528.4 |
| mixed_short | 206.5 / 209.4 | 168.4 / 171.7 | 184.3 / 190.6 |
| mixed_long | 449.2 / 458.3 | 189.6 / 194.4 | 425.9 / 433.3 |

首个活动 PCM 的平均提前量为 Preview 17 的 67.6ms（21.0%）和 Preview 20 的 23.4ms（8.6%）。各语料第二段 Callback 的 P50 / P95（ms）：

| 语料 | Chunk 24 | Preview 17→24 | Preview 20→24 |
|---|---:|---:|---:|
| zh_short | 399.7 / 408.7 | 270.7 / 272.6 | 270.3 / 274.3 |
| zh_poly_short | 411.9 / 444.3 | 276.7 / 282.2 | 280.1 / 284.3 |
| zh_long | 409.4 / 415.9 | 274.2 / 281.6 | 275.5 / 279.2 |
| zh_poly_long | 409.4 / 413.7 | 276.7 / 289.2 | 274.0 / 278.6 |
| en_short | 402.8 / 414.1 | 280.9 / 284.1 | 273.0 / 279.5 |
| en_long | 407.9 / 413.3 | 274.1 / 280.7 | 276.9 / 281.8 |
| mixed_short | 406.8 / 413.6 | 275.9 / 280.3 | 275.6 / 282.3 |
| mixed_long | 417.8 / 433.1 | 278.5 / 283.5 | 278.2 / 290.3 |

首段 PCM 总时长 / RMS 活动时长的 P50（ms）如下。Preview 的总时长较短是因为首段只有 17/20 Token；它不表示这段内容覆盖了与 Chunk 24 相同的语音。

| 语料 | Chunk 24 | Preview 17→24 | Preview 20→24 |
|---|---:|---:|---:|
| zh_short | 880 / 640 | 600 / 420 | 720 / 530 |
| zh_poly_short | 880 / 300 | 600 / 410 | 720 / 520 |
| zh_long | 880 / 790 | 600 / 520 | 720 / 640 |
| zh_poly_long | 880 / 630 | 600 / 360 | 720 / 480 |
| en_short | 880 / 380 | 600 / 340 | 720 / 340 |
| en_long | 880 / 540 | 600 / 260 | 720 / 380 |
| mixed_short | 880 / 700 | 600 / 490 | 720 / 600 |
| mixed_long | 880 / 640 | 600 / 370 | 720 / 480 |

## 完整耗时与 RTF

每格为完整合成 P50 ms / RTF P50。Preview 增加一次 VITS 解码；按 8 组的中位数等权，完整耗时分别增加 65.3ms（8.6%）和 67.3ms（8.8%），RTF 分别变差约 8.5% 和 8.7%。因此这是首包延迟与总吞吐的交换，不是整体加速。

| 语料 | Chunk 24 | Preview 17→24 | Preview 20→24 |
|---|---:|---:|---:|
| zh_short | 484 / 0.238 | 551 / 0.270 | 553 / 0.271 |
| zh_poly_short | 531 / 0.233 | 597 / 0.262 | 601 / 0.264 |
| zh_long | 1,299 / 0.195 | 1,357 / 0.204 | 1,358 / 0.204 |
| zh_poly_long | 1,723 / 0.196 | 1,789 / 0.203 | 1,787 / 0.203 |
| en_short | 542 / 0.223 | 617 / 0.253 | 613 / 0.252 |
| en_long | 1,214 / 0.199 | 1,278 / 0.209 | 1,283 / 0.210 |
| mixed_short | 700 / 0.214 | 769 / 0.234 | 767 / 0.234 |
| mixed_long | 1,041 / 0.206 | 1,099 / 0.217 | 1,110 / 0.219 |

480 个测量样本合并后的阶段耗时 P50（ms）：GPT 485 / 472 / 470，VITS 254 / 322 / 322，Frontend 0.02 / 0.03 / 0.03（依次为 Chunk 24、Preview 17、Preview 20）。样本复用同一语料且预热后测量，Frontend 数字是热缓存路径，不代表全新文本冷缓存。

## 接缝、缓冲与质量证据

- 首 Preview→Refill 接缝的低相关回退：Preview 17 为 40/160，Preview 20 为 20/160；合计首接缝 60/320。en_long 两种策略均为 20/20 回退，en_short 的 Preview 17 为 20/20 回退。回退原因是 low_waveform_correlation，诊断相关性低于 0.20，实际偏移改为 0 后再 Crossfade。
- 包含后续常规边界时，全部 860 个候选接缝中有 240 个低相关回退。候选的组内最大相邻样本跳变最高为 Preview 17 的 0.0409（en_long）及 Preview 20 的 0.0491（mixed_long）；Chunk 24 对应最大值分别为 0.0254 和 0.0257。当前相关性和边界指标不支持宣称接缝安全。
- 32ms、64ms 缓冲队列模拟在 480 次测量中均未发现 underrun。此模拟只按回调时间和字节数推进队列，未连接音频设备，也未验证实际播放、音频连续感或停顿。
- 每组 20 次中三种策略 Token hash 完全相同，音频样本全部有限。Token 一致不等同于词语、韵律或音质一致；没有自动 ASR 词级判定。
- 17/20 Token 首 Preview 均达到回调门槛；没有短/静音回退实际触发。对应回退逻辑通过单元测试覆盖。取消测试分别在 Preview 回调后取消，均观察到取消错误，随后同一模型实例合成恢复（65,222 samples）。
- **人工试听：NOT_RUN；真实扬声器/播放设备验收：NOT_RUN。** 自动指标不能确认是否重词、漏词、停顿或音色退化。

## 文件、结果与决策

- Cargo.toml：新增 opt-in feature r1-r8-preview 和 CUDA 实验 example。
- src/inference.rs：增加隐藏实验入口，支持首段 17/20 Token Preview，在 24 Token 进行 Refill；短/静音 Preview 在回调前丢弃并重置 decoder 状态。
- src/streaming.rs：保留默认 SOLA 路径；feature 实验记录 SOLA 相关性，并对低相关窗口采用零偏移 Crossfade。该后回调行为尚不构成可证明的安全回滚。
- examples/r1_8_preview_refill.rs：单句探针、8 组 30+20 CUDA 基准、Token hash/边界/缓冲/取消检查及匿名 WAV 导出。
- 原始结果：target/r1.8-preview-probe-v3/probe.json 和 target/r1.8-preview-bench/results.json。环境无模型时不可复现这些本机结果。
- 盲听材料：target/r1.8-preview-bench/alice-native-tts-r1.8-blind.zip；映射 key 单独保存在同目录的 blind-key.json，不放入 zip。包内评分表不标注策略。

下一步先人工盲听并用真实 32/64ms 播放缓冲验证 Preview、首个 Refill 和后续边界。如果英语低相关仍存在，需研究可回滚/延后发布语义的流式契约，或调整 VITS 图让 refill 的声学 overlap 对完整语义上下文稳定；在找到可靠方案前保持 Chunk 24 默认策略。未引入 TensorRT、CUDA Graph、新模型、Python 在线服务或手写 Kernel。

## 验证

- cargo fmt --all -- --check：通过。
- cargo test --release --lib --features 'r1-cuda-ep r1-r8-preview' streaming::tests::preview_ -- --nocapture：2 passed。
- cargo check --release --lib：通过，默认 API / feature 配置可编译。
- cargo check --release --lib --features 'r1-cuda-ep r1-r8-preview'：通过，实验 feature 可编译。
- git diff --check：通过；Git 仅提示已有 Rust 文件的 LF→CRLF 工作区转换警告。
- 未运行人工试听、真实播放设备、ASR 词级校验或 DirectML；这些仍是质量 Gate 的待办项。
