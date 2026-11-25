# v3–V5 原生图与 checkpoint 实机对比

测试日期为 2026-11-24。主机使用 Windows 11 x64、Core Ultra 9 285K、桌面
RTX 5070 Ti 16 GB 和 610.88 驱动。原生路径使用 TensorRT 11.2 GPT、
ONNX Runtime 1.30 CUDA EP 和 FP16 声学包。checkpoint 路径使用 PyTorch
2.7.0+cu128，GPT Transformer、DiT Transformer 块与声码器走 TensorRT，
其余计算和调度由 GPT-SoVITS 执行。两条路径都通过同一个 `alice-tts` 可执行文件的 JSONL 入口测试。

每个家族使用相同的 GPT、SoVITS checkpoint 和参考 WAV。脚本逐项核对
SHA-256。每条路径依次测试三条文本，每条预热 2 次后独立计时 6 次。
首段从发送请求到收到第一个 `audio` 事件，整句从发送请求到收到 `done`。
RTF 按该次请求实际输出的 PCM 时长计算。计时含前端处理、JSONL 传输和
PCM 编码，模型加载与 TensorRT 引擎构建在计时区间之外。

| 场景 | 输入 |
| --- | --- |
| 短句 | 今天测试本地语音。 |
| 标准句 | 今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。 |
| 滚动长句 | 清晨的列车沿着河岸缓缓驶过田野和村庄车窗里映着远处的山影和刚刚亮起的天空。 |

## 性能结果

每个单元按 checkpoint TensorRT → 原生 ABI-2 排列。P95 取这 6 次样本的插值分位数。

### 短句

| 家族 | 首段 P50 / P95，ms | 整句 P50 / P95，ms | RTF P50 | PCM 时长 P50，s |
| --- | ---: | ---: | ---: | ---: |
| v3 | 498 / 532 → 893 / 896 | 735 / 759 → 893 / 896 | 0.307 → 0.336 | 2.39 → 2.66 |
| v4 | 509 / 512 → 869 / 889 | 725 / 739 → 869 / 889 | 0.330 → 0.327 | 2.20 → 2.66 |
| V5 Turbo | 216 / 218 → 265 / 267 | 437 / 439 → 265 / 267 | 0.228 → 0.111 | 1.92 → 2.38 |
| V5 Dev | 731 / 792 → 1507 / 1597 | 978 / 1022 → 1507 / 1597 | 0.509 → 0.633 | 1.92 → 2.38 |

### 标准句

| 家族 | 首段 P50 / P95，ms | 整句 P50 / P95，ms | RTF P50 | PCM 时长 P50，s |
| --- | ---: | ---: | ---: | ---: |
| v3 | 492 / 514 → 1062 / 1872 | 1797 / 1839 → 3628 / 5149 | 0.229 → 0.470 | 7.86 → 7.73 |
| v4 | 518 / 521 → 1074 / 1077 | 1812 / 1825 → 3785 / 3801 | 0.249 → 0.489 | 7.28 → 7.74 |
| V5 Turbo | 233 / 668 → 708 / 711 | 1079 / 1592 → 2081 / 2090 | 0.154 → 0.251 | 7.00 → 8.30 |
| V5 Dev | 748 / 764 → 1921 / 1980 | 2781 / 2834 → 5723 / 5829 | 0.397 → 0.690 | 7.00 → 8.30 |

### 滚动长句

| 家族 | 首段 P50 / P95，ms | 整句 P50 / P95，ms | RTF P50 | PCM 时长 P50，s |
| --- | ---: | ---: | ---: | ---: |
| v3 | 1622 / 1688 → 2635 / 2692 | 1865 / 1923 → 2635 / 2692 | 0.216 → 0.300 | 8.63 → 8.78 |
| v4 | 1576 / 1578 → 2649 / 2663 | 1798 / 1801 → 2650 / 2663 | 0.213 → 0.302 | 8.44 → 8.78 |
| V5 Turbo | 726 / 819 → 966 / 971 | 955 / 1058 → 966 / 971 | 0.107 → 0.104 | 8.92 → 9.26 |
| V5 Dev | 2606 / 2702 → 4689 / 4884 | 2827 / 2946 → 4689 / 4884 | 0.317 → 0.506 | 8.92 → 9.26 |

标准句的原生首段和整句耗时在四个家族上均高于同权重 checkpoint TensorRT。
V5 Turbo 短句的原生整句 P50 为 265 ms，checkpoint 为 437 ms。首段 P50 分别
为 265 ms 和 216 ms。滚动长句的 V5 Turbo 整句约为 0.96 秒。

## PCM 对照

每组在两条路径内各自保持相同输出时长。12 个模型与输入组合在跨路径比较时
PCM SHA-256 均不同。两条路径各自执行文本切分、GPT 采样和初始噪声生成。
下表是同一路径的后续样本相对于首个计时样本的 PCM 相对 L2 最大值。
checkpoint 的重复 PCM 在本次每组 6 个样本中均为字节一致。

| 家族 | 短句 | 标准句 | 滚动长句 |
| --- | ---: | ---: | ---: |
| v3 | 0.41% | 0.34% | 0.49% |
| v4 | 0.13% | 0.11% | 0.14% |
| V5 Turbo | 0.11% | 0.10% | 0.11% |
| V5 Dev | 0.10% | 0.10% | 0.11% |

固定输入张量的对照和实际滚动调度见[调度测试](performance-native-diffusion-scheduling.md)。

## 复现

生成各家族的 FP16 包，按[原生扩散指南](native-diffusion.md)设置原生
TensorRT 和 ORT CUDA 库。checkpoint 端另外需要 TensorRT Python 绑定；
绑定及缓存路径见[独立使用指南](standalone.md#checkpoint-tensorrt)。
V5 对照使用支持 Dev 和 Turbo 的 GPT-SoVITS
[`6d51ffce`](https://github.com/RVC-Boss/GPT-SoVITS/tree/6d51ffceafbf240145619af4c92408a6e54b7924)。

```powershell
python -X utf8 tools/benchmark-native-diffusion.py `
  --executable ./target/release/alice-tts.exe --python $GptSovitsPython `
  --output ./output/native-diffusion-benchmark --samples 6 --warmups 2 `
  --model ./packs/v3 ./manifests/v3.json $LegacyRuntime `
  --model ./packs/v4 ./manifests/v4.json $LegacyRuntime `
  --model ./packs/v5turbo ./manifests/v5turbo.json $V5Runtime `
  --model ./packs/v5dev ./manifests/v5dev.json $V5Runtime
```

输出 `short.json`、`standard.json`、`rolling.json` 和汇总
`benchmark.json`。每次测试保留延迟、PCM 样本数对应的时长、分段数、
PCM SHA-256、与同路径首个样本的 PCM 相对 L2 及原生阶段计时。
单独选择场景可使用 `--workload`。
分段、去噪和取消结果见[调度测试](performance-native-diffusion-scheduling.md)。
