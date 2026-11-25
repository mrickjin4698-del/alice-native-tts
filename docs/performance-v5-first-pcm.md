# V5 原生首包 PCM

2026-11-25 在 Windows 11、Core Ultra 9 285K 和桌面 RTX 5070 Ti 16 GB 上测量。
使用 TensorRT 11.2、混合 FP16 GPT opt5 plan 和相同的 ABI-2 声学 plan。
原生 A/B 用同一可执行文件、文本、参考 WAV、权重和种子，用开关控制首窗。
Turbo 每窗 4 步、CFG 0，Dev 每窗 32 步、guidance 1.30，均保留完整参考窗。

短句、标准句和滚动长句与[原生扩散对照](performance-native-diffusion.md)
相同。每组预热 2 次，计时 6 次。首包从发送请求到收到首个 PCM 事件。
可听起点用 10 ms PCM 窗的 −45 dBFS RMS 阈值估算，包含已交付音频的前导
静音。播放断流按 PCM 到达时刻和剩余可播放长度估算，播放器从首包立即开始。
整句计到 `done`，RTF 除以实际输出的音频长度。计时排除模型加载和引擎构建。

## 原生结果

最终选择 Turbo 16 token、Dev 28 token。相对关闭首窗的原生路径，六组首包
P50 全部超过 1.2×，加速比为 1.40–9.44×。六组估算播放断流 P95 均为 0 ms。

| 家族 | 场景 | 首包 P50（ms），关闭 → 开启 | 首包 P95（ms） | 可听 P50（ms） | 整句 P50（ms），关闭 → 开启 | RTF P50 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| V5 Turbo | 短句 | 141.4 → 59.0 | 61.9 | 69.0 | 141.4 → 166.2 | 0.077 |
| V5 Turbo | 标准句 | 185.2 → 57.9 | 58.9 | 67.9 | 498.8 → 516.9 | 0.065 |
| V5 Turbo | 滚动长句 | 548.0 → 58.0 | 59.8 | 68.0 | 577.7 → 600.0 | 0.066 |
| V5 Dev | 短句 | 368.5 → 263.8 | 270.6 | 273.8 | 368.6 → 508.4 | 0.234 |
| V5 Dev | 标准句 | 418.6 → 262.1 | 267.0 | 272.1 | 1,186.1 → 1,382.1 | 0.173 |
| V5 Dev | 滚动长句 | 838.3 → 262.9 | 298.3 | 272.9 | 1,105.7 → 1,367.9 | 0.150 |

新增声学窗口使标准句整句 P50 增加约 18 ms（Turbo）和 196 ms（Dev）。
Turbo 首包含 300 ms PCM，Dev 含 790 ms。实际输出比原生基准少 180 ms
（Turbo）和 170 ms（Dev）的前导低电平音频。

## 调度与 PCM

首个文本片段在 GPT 生成到语义前缀时做一次声学解码，保留 120 ms 右侧上下文
和 40 ms 交叉淡化。首包包含实际语音，保留 −60 dBFS 起点前的 10 ms 音频。
较长的首片段在 72 token 再交付一次。后续窗口用已生成的 mel 作为左侧条件，
原有 GPT 继续生成到 EOS。后续文本片段沿用整段路径和相同噪声序列。

12 token 的直接首包实测为 Turbo 51.5 ms、Dev 206.0 ms。滚动长句在首包
后有 281.5 ms 和 864.2 ms 的 P95 播放断流，整句 Dev 还增加约 470 ms。
渐进交付把额外首窗限制在首个文本片段，较长片段提前补发一次，避免耗尽首包。
将首参考窗缩为 128 帧在标准句上只获得 1.03×（Turbo）和 1.08×（Dev），
因此最终配置使用完整参考窗。

Dev 的 20 和 24 token 方案让离线 Whisper 将滚动长句开头的“清晨”转写为
“京城”。28 token 在三组输入上的转写错误均未增加，Turbo 16 token 的三组
归一化转写与基准一致。标准句的后两个文本片段在两个模型各六次测试中均和基准
逐样本一致。首片段因使用有限语义上下文和独立扩散窗口而改变波形。两种配置均
通过首包后取消、同会话恢复和 CLI WAV 验证。

## 同机 gsv.cpp 对照

使用 gsv.cpp v0.1.3 官方 Windows runtime，RTX 5070 Ti Vulkan NV_coopmat2，
FP16 AR 和 F32 声码器。复用同一参考 WAV（校验 SHA-256）、参考转录、三个
合成文本、种子 1234、扩散步数和 CFG。gsv.cpp 经过本地 HTTP 长连接，扣除
WAV 头后记录首批 PCM，读完响应为整句。Alice 使用常驻 JSONL worker。
gsv.cpp 的首批读取含约 85 ms PCM，采样、切句和整句音频长度随各实现而定。
原始引擎与使用说明见 [gsv.cpp v0.1.3](https://github.com/KakaruHayate/gsv.cpp/releases/tag/v0.1.3)。

| gsv.cpp | 场景 | 首包 P50 / P95（ms） | 可听 P50（ms） | 整句 P50（ms） | RTF P50 |
| --- | --- | ---: | ---: | ---: | ---: |
| V5 Turbo | 短句 | 204.6 / 233.3 | 354.6 | 234.4 | 0.105 |
| V5 Turbo | 标准句 | 243.6 / 285.4 | 403.6 | 732.8 | 0.091 |
| V5 Turbo | 滚动长句 | 543.4 / 577.4 | 693.4 | 595.1 | 0.074 |
| V5 Dev | 短句 | 812.4 / 919.9 | 962.4 | 846.3 | 0.378 |
| V5 Dev | 标准句 | 862.4 / 881.9 | 1,022.4 | 2,515.8 | 0.313 |
| V5 Dev | 滚动长句 | 2,044.1 / 2,065.9 | 2,204.1 | 2,070.4 | 0.256 |

另用 `dbcache=auto` 测 Dev。短句、标准句和滚动长句首包 P50 分别为
1,127.3、1,260.3 和 2,997.5 ms。本次硬件与输入下，表中的常规 Dev 更快。
相对该常规配置，Alice 六组首包 P50 加速比为 3.08–9.36×，六组可听起点均
提前。Turbo 滚动长句整句为 600.0 ms 对 595.1 ms，对应返回 9.12 秒和
8.08 秒音频。其余五组 Alice 整句 P50 更低。

AnifLive-TTS v1.5 在同型号 GPU 上公开的 V5 Turbo HTTP 长连接首包为
66.89 / 70.29 ms（P50 / P95），可听起点为 80.93 / 84.33 ms。公开数据
使用项目私有的文本与声线。数据和口径见
[AnifLive-TTS HTTP 基准](https://github.com/Hiruynk/AnifLive-TTS/blob/main/docs/BENCHMARKS.md#v15-http-measurements)。

## 复现

按[原生 TensorRT 性能实验](performance-native-diffusion-tensorrt.md)准备 opt5
GPT 和声学 plan。以 Turbo 为例，设置 16 后运行同一 benchmark，Dev 使用 28。
删除该环境变量即可复测整段基准。`--save-wav` 保留每组第一条 PCM。

```powershell
$env:ALICE_NATIVE_TTS_V5_FIRST_PCM_TOKENS = '16'
python -X utf8 tools/benchmark-native-diffusion.py `
  --executable ./target/release/alice-tts.exe --python $GptSovitsPython `
  --model ./packs/v5turbo ./manifests/v5turbo.json $V5Runtime `
  --output ./output/v5turbo-first-pcm --samples 6 --warmups 2 --save-wav
```

gsv.cpp 端启动文本 sidecar 和 Vulkan 服务，用 `--allow-path` 允许参考 WAV
所在目录。选择 `v5turbo` 和 `v5dev` voice 的 FP16 AR，下面以 Dev 为例。

```powershell
python -X utf8 tools/benchmark-v5-gsv.py --port 9880 `
  --manifest ./manifests/v5dev.json --ref-audio ./peer-refs/reference.wav `
  --output ./output/gsv-v5dev --dbcache off
```
