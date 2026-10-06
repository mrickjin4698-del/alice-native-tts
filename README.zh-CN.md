# Alice Native TTS

[English](README.md)

用 Rust 实现的 GPT-SoVITS v2Pro ONNX 本地推理引擎。推理时不启动 Python
或 HTTP 服务，支持中文和英文、CPU、Windows 上的实验性 DirectML、按分句
交付音频，以及需要额外模型图的实验性 token 流式模式。

仓库提供推理源码和模型包准备工具。模型权重和参考声线由模型包作者单独
提供，本仓库不包含这些文件。

## 功能

- 使用版本化 ONNX 图进行原生 Rust 推理。
- 中文发音前端使用批量 G2PW；英文 G2P 使用随模型包提供的权重和 crate
  内的 GRU 推理代码。
- 支持若干数字、日期、时间、金额、百分比和温度格式的文本规范化。
- 默认使用 CPU；Windows 可显式选择实验性的 DirectML 后端。
- 模型包带有匹配的 ABI-1 流式图时，可启用 token 流式合成。
- 在 ONNX 图运行之间协作式响应取消。

文本前端范围有限，不等同于 GPT-SoVITS Python 前端的完整实现。具体限制见
[当前限制](#当前限制)。

## 快速开始

安装稳定版 Rust 工具链，并准备一个兼容的 ABI-1 模型包。所需文件及准备
方法见[模型包](#模型包)。

使用 CPU 合成 WAV：

```powershell
cargo run --release --example synthesize -- ./model-pack "你好，今天我们测试本地语音合成。" ./output.wav cpu
```

在 Windows 上显式选择 DirectML：

```powershell
cargo run --release --example synthesize -- ./model-pack "Hello, Alice." ./output.wav directml
```

若要在指定毫秒数后取消，先传入后端，再传时长：

```powershell
cargo run --release --example synthesize -- ./model-pack "你好。" ./output.wav cpu 500
```

`stream_bench` 示例需要额外带流式图的模型包：

```powershell
cargo run --release --example stream_bench -- ./model-pack ./output cpu
```

## 模型包

推理需要 ABI-1、`v2Pro` 架构、32 kHz 采样率及 512-token GPT KV 容量的模型
包。基础包包含 ONNX 图、前端数据、参考 WAV 及其文本，并须带有记录来源的
`LICENSES.txt`。`pack.json` 声明 ABI 和参考文件。启用 token 流式还需
`sovits_stream.onnx` 以及匹配的流式元数据。

可以使用 `tools/prepare-native-tts-pack.py`，从本地兼容模型导出文件和
GPT-SoVITS 安装准备模型包。Python 只用于这个离线准备步骤，运行时推理由
Rust 完成。可选流式图由 `tools/export-native-tts-stream.py` 导出并进行数值
对齐检查。

工具要求操作者自行提供模型文件和源码目录。请核对各上游模型的授权，并在
模型包中保留来源声明。本源码仓库不应放入模型权重、声线参考录音或生成的
模型包。

## API 与 Alice 集成

crate 对外提供 `Config`、`Backend`、`NativeTts`、模型包检查和文本规范化。
最简嵌入示例见 `examples/synthesize.rs`，可用 `cargo doc --open` 生成 API
文档。

Project Alice 的桌面适配器留在主项目中，负责提供方选择、共享缓存、事件、
播放队列和应用级取消；这些应用集成逻辑不属于本 crate。

## 性能说明

目前只有一台机器上的初步测量，并非跨设备基准，也不构成性能保证。尚未进行
系统性的模型量化、权重去重或多硬件调优；现有代码已经包含批量发音推理、分句
优先交付、可选 token 流式和 DirectML 等针对性处理。

硬件信息、计时口径、结果和复现命令见[性能记录](docs/performance.md)。

## 当前限制

- 仅支持 v2Pro 模型包及中文、英文。
- 导出图的语速固定为 1.0，声学噪声固定为 0.0。
- Alice 适配器的进程内缓存只保留一种基础声线配置；crate 本身将缓存所有权
  留给宿主程序。
- 取消只能在 ONNX 图运行之间协作处理，不能强制中断正在执行的 ONNX kernel。
- 中文轻声/儿化、依赖词性的英文同形异音词、日语和完整地区化数字语法尚未
  覆盖。
- 长文本长度受导出的 GPT KV 缓存容量限制。

## 许可证与来源

crate 使用 Apache-2.0。中文发音表、tokenizer 数据和拼音拆分代码的上游来源
见 [NOTICE.md](NOTICE.md)。模型包组件各自保有来源和授权信息，请查阅模型包的
`LICENSES.txt`。
