# v3–V5 原生 TensorRT 性能实验

仍使用 Windows 11、桌面 RTX 5070 Ti 和上一轮的三个文本、模型权重及参考音频。
每个新用例预热 2 次，计时 6 次。脚本复用同一 GPU、驱动和模型下的 checkpoint
数据，并检查文本、权重与参考 WAV 的摘要。合计 24 个首段与整句 P50 指标。
追平要求原生耗时不超过 checkpoint，1.2× 要求不超过 checkpoint 耗时除以 1.2。
原始口径及 checkpoint 结果见[同权重实机对比](performance-native-diffusion.md)。

## TensorRT 声学图与 GPT 精度

目标特征、条件、DiT step 和声码器可以使用本地 TensorRT 11 plan。
Rust 仍管理文本分片、参考和滚动窗口。TensorRT 在窗口内复用两份 GPU 状态，
四步一组等待并处理取消。v3 的目标特征保留 CUDA EP，其余三个家族的目标特征
使用经过实际执行的 TensorRT tactic 配置。

标准句 P50 单位为 ms，每格依次为首段和整句。

| 家族 | Checkpoint TensorRT | 原生 TensorRT 声学 + FP32 GPT | 原生 TensorRT 声学 + 混合 FP16 GPT |
| --- | ---: | ---: | ---: |
| v3 | 492 / 1797 | 558 / 1753 | 636 / 1768 |
| v4 | 518 / 1812 | 469 / 1393 | 441 / 1299 |
| V5 Turbo | 233 / 1079 | 242 / 680 | 209 / 622 |
| V5 Dev | 748 / 2781 | 799 / 2291 | 780 / 2326 |

第一轮 20/24 项追平，11 项达到 1.2×。第二轮 22/24 项追平，10 项达到 1.2×。
两轮的 12 个整句 P50 都低于 checkpoint。混合 FP16 的 GPT 使用半精度 KV，
保留首尾接口与部分算子的 FP32 精度。语义采样和实际 PCM 时长会随精度变化，
所以整句延迟和 RTF 分别报告。V5 Dev 的 32 步和 guidance 1.30 一直保留。

## v3 目标特征

v3 的目标特征 plan 在有效输入上也能复现 TensorRT 进程异常。改写 15/8
最近邻缩放和将 FP16 权重拓宽到 FP32 两组尝试都出现同样的异常。CUDA EP
保留目标图时，启用 TF32 将标准句整句 P50 从 1,768 ms 降到约 1,525 ms。

随后为目标图导出 96×64 和 256×128 两个固定形状。每次显式传入有效 token
数和音素数，编码器与 WN 继续使用有效长度的掩码。六组固定输入对照中，
有效帧最大相对 L2 为 0.112%。请求超出分桶时走原动态 CUDA 图。
同一 GPT 配置下，分桶把标准句首段与整句 P50 降至 451 和 1,216 ms。

## V5 Dev 去噪

每个扩散窗口已经使用真实帧数。该窗口内 DiT 的注意力掩码全真。
省掉这张掩码后，三个固定输入长度的 PyTorch 单步输出逐值一致。
用于固定提示区的最终生成掩码仍参与更新。合批实验在同一去掩码图中
用 batch 2 执行两路 CFG，固定输入的单步相对 L2 约为 0.012%。

| V5 Dev 方案 | 标准句首段 / 整句 P50，ms | 滚动句首段 / 整句 P50，ms |
| --- | ---: | ---: |
| 原掩码，串行 CFG | 780 / 2,326 | 2,471 / 2,471 |
| 去全真掩码，串行 CFG | 417 / 1,243 | 1,133 / 1,133 |
| 去全真掩码，CFG 合批 | 464 / 1,390 | 1,348 / 1,348 |

## 统一去掩码与首窗口出声

同样的去掩码改动扩到 v3、v4 和 V5 Turbo。三个固定输入长度下，每个模型的
PyTorch 单步结果都逐值一致。混合 FP16 GPT、串行 CFG 与 v3 分桶一起使用后，
四个家族共 24/24 项追平 checkpoint，22/24 项达到 1.2×。V5 Turbo 的
标准句与滚动长句首段还低于 1.2× 目标。全矩阵保存于
`matched-unmasked/benchmark.json`。

提前交付的实验在首个滚动窗口结束后执行一次声码器，保留 128 帧作为右侧
上下文。第二次对完整 mel 执行声码器并补发剩余样本。以下单位为 ms。

| 家族 | Checkpoint 标准句首段 / 整句 | 去掩码标准句首段 / 整句 | 滚动句首段，提前交付前 → 后 | 滚动句整句，提前交付前 → 后 |
| --- | ---: | ---: | ---: | ---: |
| v3 | 492 / 1797 | 361 / 1025 | 1068 → 850 | 1068 → 1129 |
| v4 | 518 / 1812 | 306 / 902 | 935 → 775 | 935 → 941 |
| V5 Turbo | 233 / 1079 | 195 / 589 | 655 → 622 | 655 → 655 |
| V5 Dev | 748 / 2781 | 455 / 1349 | 1242 → 947 | 1242 → 1242 |

提前交付后的 24 项均追平，22 项达到 1.2×。V5 Turbo 的滚动句首段
从 655 ms 降到 622 ms，距离 605 ms 目标约 17 ms。v3 声码器两次执行
使整句多约 61 ms。原始数据在 `matched-unmasked-rolling-pcm/benchmark.json`。
四个家族的滚动句在提前交付前后，六次输出的完整 PCM SHA-256 均一致。

在相同 Turbo 配置下，GPT Softmax 也使用 FP16 时，标准句首段为 204 ms，
滚动句首段为 618 ms。两项都距 1.2× 目标有差距。数据在
`matched-turbo-softmax/benchmark.json`。

## 1.2× 实验基线

使用原混合 FP16 GPT 图，统一切到 TensorRT 优化级别 5 的缓存。encoder 档位的
音素与参考 token 设为 72 和 96，step 的 active length 档位为 272。
四个家族共用这些 GPT plan。声学使用去掩码图，v3 目标特征使用分桶，
滚动句提前交付首个窗口。24/24 项达到 1.2×，范围为 1.223× 到 3.023×。
最窄的是 V5 Turbo 标准句首段，190.130 ms 对 checkpoint 的 232.546 ms。
基线原始文件为 `matched-opt5/benchmark.json`。

| 家族 | 短句首段 / 整句，ms | 标准句首段 / 整句，ms | 滚动句首段 / 整句，ms |
| --- | ---: | ---: | ---: |
| v3 | 282 / 282 | 350 / 980 | 839 / 1112 |
| v4 | 275 / 275 | 298 / 872 | 776 / 947 |
| V5 Turbo | 145 / 145 | 190 / 511 | 570 / 601 |
| V5 Dev | 392 / 392 | 432 / 1235 | 884 / 1159 |

## 基线上的续实验

保持 opt5 GPT、声学引擎和输入，关掉提前交付后复测滚动句。
下表按整窗交付 → 提前交付排序。两组中，每个家族六次完整 PCM 的 SHA-256
均相同。原始数据在 `matched-opt5-whole-rolling/benchmark.json` 和
`opt5-rolling-pcm-control.json`。

| 家族 | 首段 P50，ms | 整句 P50，ms | 提前交付的首段音频，s |
| --- | ---: | ---: | ---: |
| v3 | 1021 → 839 | 1021 → 1112 | 5.33 |
| v4 | 943 → 776 | 943 → 947 | 5.45 |
| V5 Turbo | 596 → 570 | 596 → 601 | 5.12 |
| V5 Dev | 1157 → 884 | 1157 → 1159 | 5.12 |

提前交付在 v3 上用约 91 ms 整句开销换取约 182 ms 首段收益。其他三个
家族的整句增加约 2–4 ms。整窗交付的八项滚动 P50 也均过 1.2×。

## 验收

四个家族均通过 JSONL 分段、首段后取消、同会话恢复和 CLI WAV 检查。
另用 V5 Dev 验证在滚动首窗口 PCM 后取消，收到一个 PCM 包后终止，
同一会话的下一次请求正常完成。默认和 TensorRT feature 构建各通过
24 个 Rust 单元测试。
原 ABI-2 CUDA 声学路径通过 WAV 复核，ABI-1 v2Pro 通过 token 流式、取消和恢复复核。

## 复现原生 TensorRT

```powershell
cargo build --locked --release --features r2-trt-native-cuda-vits --bin alice-tts
python tools/prepare-native-diffusion-tensorrt.py `
  --trtexec "$env:ALICE_TRT_HOME/bin/trtexec.exe" --output ./output/engines `
  --pack ./packs/v3 --pack ./packs/v4 --pack ./packs/v5turbo --pack ./packs/v5dev
$env:ALICE_NATIVE_TTS_DIFFUSION_TRT_ROOT = (Resolve-Path ./output/engines).Path
$env:ALICE_NATIVE_TTS_DIFFUSION_TRT_TARGET = 'v4,v5turbo,v5dev'
./target/release/alice-tts worker ./packs/v5turbo --device tensor_rt_native
```

混合 FP16 GPT 使用与当前包相同 GPT 权重生成的两张图。它们可在共享 GPT 权重的
包之间复用。

```powershell
& $GptSovitsPython -I -B tools/prepare-native-diffusion-gpt-fp16.py `
  --pack ./packs/v5turbo --output ./output/gpt-fp16
$env:ALICE_NATIVE_TTS_DIFFUSION_GPT_FP16_DIR = (Resolve-Path ./output/gpt-fp16).Path
```

用独立缓存构建优化级别 5 的 GPT plan，档位和原分支的动态长度契约兼容。
GPT checkpoint、ONNX 权重和采样代码都沿用前一轮。

```powershell
./tools/prepare-native-diffusion-gpt-plans.ps1 `
  -GptDir $env:ALICE_NATIVE_TTS_DIFFUSION_GPT_FP16_DIR `
  -Trtexec "$env:ALICE_TRT_HOME/bin/trtexec.exe" -CacheDir ./output/gpt-cache-opt5
$env:ALICE_NATIVE_TTS_TRT_CACHE_DIR = (Resolve-Path ./output/gpt-cache-opt5).Path
```

`tools/benchmark-native-diffusion.py --checkpoint-report <benchmark.json>`
复用已有 checkpoint 数据。其余参数示例见[性能对比的复现命令](performance-native-diffusion.md#复现)。

去掩码的 step 使用 `tools/export-native-diffusion-step-variant.py` 导出。
将每个家族的 `--output` 设为 `./output/steps/<family>`，选择
`--variant unmasked`。准备 TensorRT plan 时加 `--step-root ./output/steps`。
V5 Dev 的对照可使用 `--variant batched` 和 `--step-variant batched`。

```powershell
& $GptSovitsPython -I -B tools/export-native-v3-target-buckets.py `
  --checkpoint ./manifests/v3.json --runtime $LegacyRuntime `
  --pack ./packs/v3 --output ./output/v3-buckets
$env:ALICE_NATIVE_TTS_DIFFUSION_V3_BUCKETS = (Resolve-Path ./output/v3-buckets).Path
$env:ALICE_NATIVE_TTS_DIFFUSION_ROLLING_PCM = '1'
```
