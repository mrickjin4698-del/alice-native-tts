# R1.4 冷前端与流式音质评估

测量日期：2026-10-26。基于本机 ABI-1 GPT-SoVITS v2Pro 模型包；没有把模型权重加入仓库。

## 结论

- 中文新文本的主要前端瓶颈是 G2PW，其次是 BERT。`zh_poly_long` 冷 miss 下，CPU G2PW/BERT 分别为 290.2/56.0 ms；两个图都使用 CUDA EP 后分别为 17.9/5.9 ms，Frontend 合计约 24.4 ms。
- 双图 CUDA 在多音长句上把 TTFA P50/P95 从 CPU serial 的 532/545 ms 降至 208/217 ms；CPU 并行也把 TTFA 从 532/545 ms 降至 460/478 ms。两个优化均保持 semantic token 数与 hash 一致。
- 缓存只帮助同一模型实例内重复的精确文本：多音长句 Frontend P50 从冷 miss 的 347 ms 降至命中时的 0.045 ms；新文本仍执行完整 G2PW/BERT。
- 首包阈值 20 在完整 8 语料矩阵中降低了所有语料的 TTFA P50（3.5%–12.3%），但 `zh_poly_long`、`mixed_long` 的 TTFA P95 变差；20-token 的回调边界在部分中英/多音语料更大，`mixed_long` 队列模拟还出现 1/20 underrun 样本。人工盲听和真实播放设备验收未完成，默认仍为 24。
- CPU 路径的 5 组 E2E 流式合成、首包后取消和后续恢复通过。DirectML 因当前 ORT runtime 不支持该 EP 而 BLOCKED。

| 验收项 | 状态 |
|---|---|
| CPU Frontend、Chunk 20/24，8 语料 × 每组 30 warmup + 20 测量 | PASS |
| 精确 cache miss/hit 对照，每模式 2 语料 × 2 阈值 × 每组 30/20 | PASS |
| CPU 并行 Frontend，每组 30/20；phones/features 对照 | PASS |
| G2PW-only、BERT-only、双图 CUDA EP E2E，每组 30/20 | PASS，显式允许审计过的 CPU shape helpers |
| GPU/CPU Frontend 数值对照 | PASS，详见下文；不等同于音质盲听 |
| 实际播放设备 underrun、人工盲听 | NOT RUN；当前 crate 没有播放设备接口，盲听材料已导出 |
| DirectML E2E | BLOCKED：`Specified provider is not supported.` |
| 修改默认首包阈值 / 默认 Frontend EP | 未修改 |

## 运行环境与测量方法

| 项目 | 配置 |
|---|---|
| 系统 / GPU | Windows x64 / NVIDIA GeForce RTX 5070 Ti，16 GiB，驱动 610.74 |
| CPU | Intel Core Ultra 9 285K |
| ONNX Runtime | 1.30.0，Rust wrapper `ort` 2.0.0-rc.12，动态加载 |
| GPU 运行时 | CUDA 13.0 / cuDNN 9.12 |
| 模型 | 本地 ABI-1 GPT-SoVITS v2Pro 包，权重和配置未改 |
| 语义采样 | 每次请求重置 seed 1234 |
| Chunk | 候选首包 20/24；其后固定 24 |
| 缓存 | 冷计算组与 cache miss 组分开；命中组由 30 次 warmup 填充精确文本项 |

每组都先预热 30 次，再测 20 次。TTFA 是请求开始到首次音频 callback 的时间；RTF 是完整合成墙钟时间除以生成音频时长。P50/P95 使用 20 个样本的排序分位数。热态计时没有启用 ORT operator trace；节点位置使用单独的诊断 trace，避免 trace 开销污染基准。

完整原始样本、阶段时间线、逐 Token GPT Step 时间、回调记录、音频差异和资源快照保存在本机 `target/`，不纳入 Git：

- 全语料冷计算：`target/r1.4-cpu-cold/r1.4-first-audio.json`
- 实际精确 Key miss：`target/r1.4-cache-miss/r1.4-first-audio.json`
- 精确 Key hit：`target/r1.4-cache-hit/r1.4-first-audio.json`
- CPU 并行：`target/r1.4-parallel-cpu/r1.4-first-audio.json`
- 单图及双图 CUDA：`target/r1.4-g2pw-cuda/`、`target/r1.4-bert-cuda/`、`target/r1.4-frontend-both-cuda-memory/`
- CUDA provider 节点 trace：`target/r1.4-probe-front-both-diagnostic/ort-profile/`

全语料冷计算矩阵使用禁用 Frontend 缓存的路径，确保 warmup 不会污染冷文本结果。补充 cache miss 组启用缓存，但每次请求前清空精确 Key，再执行 lookup、记录 miss 并插入；因此 20 个计时样本都是真实 cache miss。为保持相同文本和采样条件，这些样本重复同一测试句，不代表 20 种不同句子的语言覆盖。

## Frontend 依赖与缓存 Key 审计

中文流程先分词并准备输入。G2PW 为多音字选择读音；BERT 输入只依赖 tokenizer token IDs 和 attention/token type mask，不依赖 G2PW 的读音结果。BERT hidden states 的特征扩展需要之后由读音映射产生的 `word2ph`，所以安全并行范围是 **G2PW 执行与 BERT 执行并行，随后串行做音素映射和 BERT 特征扩展**。英语路径不执行 G2PW/BERT 图。

图的动态 shape 与计算量也符合上述依赖：BERT 的 `input_ids`、`attention_mask`、`token_type_ids` 为 `[1, seq_len]`，输出 hidden states 为 `[1, seq_len, 1024]`；G2PW 的 tokenizer 输入为 `[batch, seq_len]`，多音字 mask 为 `[batch, 1305]`，输出概率为 `[batch, 1305]`。本包的 `bert.onnx` 约 1.20 GB，`g2pw.onnx` 约 635 MB；CPU 多音长句的 G2PW 耗时随候选多音字数量和 batch 数明显增加。

缓存 Key 现在包含：归一化文本、语言路由（Chinese/English/Mixed/Empty）、Key schema、G2PW batch size、G2PW EP 和 BERT EP。缓存属于单个 `Frontend` 实例，模型权重和字典配置不会跨实例共享；新增配置字段会随 schema 版本更新。单元测试覆盖语言路由、两个 EP、batch size 与 schema 区别。

冷计算 CPU Frontend P50/P95：普通中文短句约 28–30 ms；普通中文长句 138/143 ms，其中 G2PW 87/91 ms、BERT 51/52 ms；含较多多音字的长句 351/362 ms，其中 G2PW 295/305 ms、BERT 57/59 ms；中英混合长句 109/141 ms，其中 BERT 84/111 ms、G2PW 25/30 ms；英文路径低于 0.1 ms。

冷 miss 和命中缓存对照（首包阈值 24，P50）：

| 文本 | 冷 miss Frontend | cache hit Frontend | 冷 miss TTFA | cache hit TTFA |
|---|---:|---:|---:|---:|
| `zh_poly_long` | 343.5 ms | 0.041 ms | 552.3 ms | 201.5 ms |
| `mixed_long` | 106.4 ms | 0.032 ms | 313.9 ms | 202.6 ms |

命中组的 20/20 样本为 1 hit/0 miss；miss 组为 0 hit/1 miss。数组插入耗时 P50 约 0.006–0.016 ms，远小于模型执行。命中收益仅适用于同一实例里重复的精确文本和相同 Frontend 配置。

## CUDA EP 与安全并行对照

ORT trace 的优化图节点 kernel 事件分配：

| 图 | CUDA EP | CPU EP | 观察到的主要 CPU 节点 |
|---|---:|---:|---|
| BERT | 630 | 2 | Gather、Unsqueeze |
| G2PW | 344 | 14 | Gather、Concat、Unsqueeze |

主要计算节点实际在 CUDA：BERT 的 MatMul/LayerNormalization/Softmax/BiasGelu，G2PW 的 MatMul/Attention/LayerNormalization/BiasGelu。无 CPU 节点的严格配置在 Session 初始化时报 `strict GPU placement rejected CPU-assigned nodes`；基准明确使用诊断 fallback 开关，且 trace 显示 CPU 节点仅为已审计的形状/索引辅助节点。上述计数来自 ORT trace 的节点 kernel 事件，不把事件数误当成静态模型 node 数。

`zh_poly_long` 冷 miss、同一 v2Pro 模型、30/20：

| Frontend 配置 | TTFA 20 P50/P95 | TTFA 24 P50/P95 | Frontend 20 P50 | G2PW/BERT 执行 P50（20） |
|---|---:|---:|---:|---:|
| CPU serial | 532/545 ms | 552/572 ms | 347 ms | 290/56 ms |
| G2PW CUDA、BERT CPU | 253/261 ms | 272/280 ms | 72 ms | 18/54 ms |
| G2PW CPU、BERT CUDA | 474/482 ms | 493/505 ms | 291 ms | 285/6 ms |
| G2PW + BERT CUDA | 208/217 ms | 232/239 ms | 24 ms | 18/6 ms |

双图 CUDA 相对 CPU serial 冷 miss 的 TTFA P50 降低约 61%，P95 降低约 60%；Chunk 20/24 的完整 RTF P50 分别约 0.199/0.199，完整耗时 P50 约 1,751/1,756 ms。该 GPU Frontend E2E 只在中文多音长句上测量，不能据此推断所有机器或所有文本的收益。

GPU/CPU Frontend 对照用两条含多音字的中文句子验证：G2PW CUDA 生成的 phones 与 CPU 完全相同；BERT CUDA 输出特征最大绝对误差 `7.63e-6`、RMSE `4.5–5.9e-7`；双图 CUDA 同样通过。各 GPU 组的 semantic token 数/hash 与 CPU serial 及阈值 24 一致。

双图 CUDA 期间 `nvidia-smi` 全局显存快照约 7.1 GiB 已用，工作集约 1.7 GiB、Windows private bytes 约 6.9 GiB。WDDM 的进程级 CUDA memory 字段返回 N/A，显存数字是整张卡全局用量，不作进程精确归因。

CPU 安全并行在同两条长句上测量：

| 文本 | serial Frontend / TTFA P50，20 | parallel Frontend / TTFA P50，20 | serial Frontend / TTFA P50，24 | parallel Frontend / TTFA P50，24 |
|---|---:|---:|---:|---:|
| `zh_poly_long` | 347 / 532 ms | 283 / 460 ms | 344 / 552 ms | 288 / 491 ms |
| `mixed_long` | 105 / 292 ms | 88 / 271 ms | 106 / 314 ms | 88 / 293 ms |

每组 30/20。中文多音长句的 Frontend P50 降约 18%，混合长句降约 16%；TTFA P50 分别降约 13% 和 7%。profiling 时间线同时包含 BERT、G2PW 各自运行段与两者并行 wall span。模型包上的 CPU 标量、batch、parallel 测试确认 phones 与 features 完全一致。

## Chunk 20 与 24：首包、RTF、边界和播放队列

全语料冷计算矩阵，每格 TTFA 为 P50/P95（ms），RTF 为 P50/P95：

| 语料 | TTFA 20 | TTFA 24 | TTFA P50 变化 | RTF 20 | RTF 24 |
|---|---:|---:|---:|---:|---:|
| `zh_short` | 218/222 | 249/274 | -12.2% | 0.260/0.265 | 0.269/0.295 |
| `zh_poly_short` | 239/250 | 250/258 | -4.6% | 0.259/0.269 | 0.253/0.257 |
| `zh_long` | 325/343 | 347/381 | -6.1% | 0.222/0.229 | 0.215/0.310 |
| `zh_poly_long` | 544/618 | 572/587 | -4.9% | 0.243/0.271 | 0.242/0.248 |
| `en_short` | 185/213 | 211/227 | -12.3% | 0.229/0.246 | 0.229/0.251 |
| `en_long` | 185/194 | 210/219 | -12.0% | 0.200/0.207 | 0.203/0.211 |
| `mixed_short` | 253/263 | 269/275 | -6.0% | 0.239/0.248 | 0.235/0.243 |
| `mixed_long` | 304/396 | 315/331 | -3.5% | 0.233/0.389 | 0.222/0.226 |

所有 case 的 20/24 semantic token 数与 hash 相同。20-token 首个 callback 提供约 720 ms 音频，24-token 约 880 ms。完整合成耗时没有稳定改善：例如 `zh_poly_short` 的 20-token P50 慢约 13 ms，`mixed_long` 的 20-token RTF P95 为 0.389，高于 24 的 0.226。

20 次测量样本聚合的回调边界绝对跳变 P95（candidate 20 / reference 24）：

| 语料 | 20 / 24 边界跳变 P95 | 20 / 24 最大跳变 |
|---|---:|---:|
| `zh_short` | 0.00002 / 0.00005 | 0.00002 / 0.00005 |
| `zh_poly_short` | 0.00136 / 0.00125 | 0.00136 / 0.00125 |
| `zh_long` | 0.01974 / 0.02407 | 0.01974 / 0.02407 |
| `zh_poly_long` | 0.02660 / 0.02159 | 0.02660 / 0.02159 |
| `en_short` | 0.00334 / 0.00196 | 0.00334 / 0.00196 |
| `en_long` | 0.00782 / 0.02543 | 0.00782 / 0.02543 |
| `mixed_short` | 0.05333 / 0.00904 | 0.05333 / 0.00904 |
| `mixed_long` | 0.04912 / 0.02573 | 0.04912 / 0.02573 |

阈值 20 在两个混合语料和中文多音长句上的最大边界跳变较高。队列模型以 callback 到达时间和媒体时长离线模拟 32/64 ms 预缓冲；只有 `mixed_long` 阈值 20 有 1/20 样本出现模拟 underrun，其他矩阵组为 0。首 Chunk 远长于 32/64 ms，所以模拟首个可播放时间基本等于首 callback。crate 没有声卡/播放队列接口，此结果不是实际设备 underrun 通过。

同一阈值的重复输出数值稳定：cache miss 附加组中 sample 0/1 波形 cosine similarity 大于 0.9999999999999。跨阈值分块时，语义 token 相同但 VITS 输入形状、SOLA 拼接边界不同；原始未对齐波形数值只用于记录差异，不作为音质评分。人工盲听仍是上线门槛。

盲听 A/B WAV 和说明在 `target/r1.4-cpu-cold/blind-audio/`、`target/r1.4-cpu-cold/blind-listening-prompts.txt`；阈值映射单独保存在 `blind-listening-key.jsonl`，试听前不要打开。

## 回归、修改和后续

- CPU public streaming API：中文、混合、中英文和短句 5 组均返回有限音频且正常多 Chunk；首 Chunk 后取消时没有额外 callback，之后的新合成恢复成功。结果为 `target/r1.4-cpu-regression/results.json`。
- DirectML：使用同一 ORT 1.30 动态库启动时立即返回 `Specified provider is not supported.`，没有进入图执行；缺少 `onnxruntime_providers_dml.dll`，因此保留 BLOCKED。
- 修改：`src/frontend.rs` 加入缓存 Key 配置覆盖、按 G2PW/BERT 配置 EP、仅 profiling 环境开启的冷 miss 与中文并行路径及比较测试；`src/inference.rs` 加 Frontend provider placement/profile 支持；`src/profiling.rs` 加并行 wall span；`examples/r1_4_profile.rs` 加 30/20 前端/首包/边界/队列/音频/内存记录；`Cargo.toml` 注册 runner；本报告记录证据。
- 未引入 Python 推理服务、TensorRT、手写 Kernel 或新模型；没有提交或推送修改。

下一步应先完成导出的盲听 A/B 和目标播放设备 32/64 ms 缓冲实测。质量通过后再决定是否为 CUDA 用户开放双图 Frontend EP；无 CUDA 的 CPU 并行可以作为独立可选项继续做取消响应和更多文本验证。当前默认 CPU Frontend 与 Chunk 24 保持不变。
