# R2.1 — 原生 TensorRT GPT 与常驻 KV Cache

## 结论

在 Windows + RTX 5070 Ti 上，新增可选的 TensorRT 11 原生 GPT Encoder/Step 后端。保持 Chunk 24、ABI-1 模型、随机种子 1234、CPU Token Sampling 和 CPU VITS。GPT K/V Cache 使用两组 CUDA 缓冲区 ping-pong 更新，不逐 Token 传回主机。

在同一模型包、参考音频、文本和进程环境下，30 次预热后各测 20 次：

- GPT Step P50 从 1888.44ms 降至 469.49ms（4.02 倍）；每 Token P50 从 11.308ms 降至 2.662ms。
- TTFA P50 从 432.04ms 降至 182.79ms；P95 从 437.09ms 降至 188.84ms。
- 完整流式合成 P50 从 2836.29ms 降至 1365.49ms，RTF P50 从 0.4266 降至 0.2054。
- 20/20 测量的语义 Token 数及 Token Hash 一致；保存的首条测量 WAV 为逐样本 bitwise identical。

TensorRT 仍通过 `r2-trt-native` feature 显式启用，未改变默认后端。此轮只迁移 GPT Encoder/Step，Frontend 与 VITS 仍走 ORT CPU；没有声称已把九个神经阶段全部迁到 TensorRT。

## 实现

- `Backend::TensorRtNative` 仅在 Windows x64 的 `r2-trt-native` feature 下出现。该 feature 使用 ORT 动态加载，ORT 仍处理 Frontend、VITS 和参考音频模型。
- 新增 `native/trt_gpt.cpp` 原生桥接：构建或加载 TensorRT Engine、固定 I/O 缓冲区与执行 Context、一个非阻塞 CUDA Stream、固定尺寸 pinned-host 暂存区。
- GPT Encoder 输入使用 ABI-1 原图；动态文本/音素轴 Profile 为 min 1、opt 128、max 512。GPT Step 的 ABI-1 图与权重不变。
- TensorRT 构建为 FP32、关闭 TF32、关闭辅助 Stream。输出仍是原图的 top-50 FP32 logits 与 INT64 Token ID。
- Encoder 的 K/V 输出进入 CUDA 分配的第一组 cache。每个 Step 从当前 K/V 读取并写入另一组，下一步交换索引。Step 只把 top-k logits/IDs复制到 CPU，让现有 Rust LCG/采样代码决定 Token；K/V 不做 Host/Device 往返。
- 每次新请求在 Encoder 前重置 cache 状态；取消由原有 Rust Token 边界检查返回 `Error::Cancelled`。
- Engine Plan 缓存在 `ALICE_NATIVE_TTS_TRT_CACHE_DIR` 或 `%LOCALAPPDATA%\\AliceNativeTts\\Trt11GptCache`。缓存按 ONNX 文件内容指纹隔离。Plan 和 SDK 不加入模型包。

## 环境与复现

- GPU：NVIDIA GeForce RTX 5070 Ti，16GB，驱动 610.74。
- TensorRT：本地解压包是 Windows CUDA 12.9 TensorRT Enterprise SDK；头文件为 11.2.1 build 2，含 SM120 builder resource。该 engine 在当前 GPU/驱动上已真实构建并执行。
- CUDA 运行时：12.9.79；cuBLAS 12.9.1.4；ORT DLL：1.30.0.20260909.3.f2c39fe。
- SDK 和 CUDA 运行时均解压在忽略目录 `target/`，没有运行全局安装器或修改驱动。
- TRT SDK 压缩包 SHA-256：`8653DF0B9A57EA5FD349E12D20243F6131F02052EF8041E756366E5779348C16`。

兼容性边界：本机验证的是包内标记为 CUDA 12.9 的 Enterprise SDK build 2。NVIDIA 当前通用 11.2.1 Release Notes 对标准发行包写明 CUDA 13.3；该通用说明与这里下载的 Enterprise 包并非同一构建目标。因此本报告只确认“这份本地 SDK + CUDA 12.9 runtime + RTX 5070 Ti”的实测组合，不把它推广为任意 TensorRT 11.2 包或 TensorRT EP 的兼容保证。[TensorRT 11.2.1 Release Notes](https://docs.nvidia.com/deeplearning/tensorrt/11.4.0/getting-started/release-notes-11/11.2.1.html)

配置 `ALICE_TRT_HOME`、`ALICE_CUDA_HOME`、`ALICE_CUDA_NVCC_HOME` 指向已解压的 TensorRT、CUDA Runtime 和 CUDA NVCC 头文件目录；运行时还需把 TensorRT、CUDA Runtime、cuBLAS DLL 目录加入 `PATH`，并令 `ORT_DYLIB_PATH` 指向本机 ONNX Runtime DLL。

```powershell
cargo build --release --features r2-trt-native --example synthesize
cargo run --release --features "r0-profiling r2-trt-native" --example r0_profile -- `
  target/r1-pack target/r2.1-trt-native/bench-trt tensor_rt_native `
  --chunks=24 --warmups=30 --repeats=20
```

基准报告、原始 PCM、逐样本数据、TensorRT Plan、下载文件及环境日志均保存在忽略的 `target/r2.1-trt-native/`。`r0_profile` 运行命令的 JSON 汇总含每 Token 时间、阶段耗时、RTF、语义 Token Hash 和主机进程内存。

## A/B 结果

每组 30 次预热 + 20 次测量；同一组中文文本，Chunk 24，固定参考音频和模型配置。测量发生在各后端首次加载/预热后，Frontend 文本缓存命中。单位均为毫秒，RTF 无量纲。

| 指标 | ORT CPU GPT P50 / P95 | TensorRT 11 GPT P50 / P95 | P50 变化 |
|---|---:|---:|---:|
| GPT Encoder | 65.65 / 66.85 | 4.27 / 5.12 | -93.5% |
| GPT Step 总耗时 | 1888.44 / 1896.93 | 469.49 / 476.96 | -75.1% |
| 每 Token | 11.308 / 12.150 | 2.662 / 3.751 | -76.5% |
| 首 Callback TTFA | 432.04 / 437.09 | 182.79 / 188.84 | -57.7% |
| 完整合成耗时 | 2836.29 / 2845.77 | 1365.49 / 1373.60 | -51.8% |
| RTF | 0.4266 / 0.4280 | 0.2054 / 0.2066 | -51.8% |
| 首 VITS Streaming Call | 110.58 / 111.76 | 111.17 / 114.18 | +0.5% |
| 全部 VITS Run | 873.68 / 879.10 | 883.33 / 891.48 | +1.1% |

GPU 显存由 `nvidia-smi` 每 750ms 采样。TensorRT 基线为 2678MiB，峰值 3884MiB，峰值增量约 1206MiB；该值是整卡使用量差分，不能隔离系统其他 GPU 进程。R0 进程加载后的工作集约 2795MiB、Private Bytes 4100MiB；CPU 基线分别约 3228MiB、3297MiB。R0 进程的内存字段是主机内存，不是显存。

### 一致性与取消

- 两个后端各有 20 条测量记录，每条均生成 167 个语义 Token。全部 20 个 64-bit semantic token hash 完全一致，固定采样器继续由 CPU 执行。
- 两边保存的首条测量原始 f32 PCM 均为 212,777 个样本、851,108 字节；字节逐项相同，最大绝对误差和 RMSE 均为 0。没有执行人工盲听；本次波形相同，因此这不是对其他模型或采样设置的主观音质认证。
- 500ms 取消场景在已有 1 次音频 Callback 后返回 `Error::Cancelled`；同一个 `NativeTts` 对象随后完整恢复，生成 167 Token、212,777 个音频样本，耗时 1368.1ms。

## 构建与回归

- `cargo check --release --lib`：默认功能通过。
- `cargo check --release --features r1-cuda-ep --lib`：CUDA EP 既有实现编译通过。
- `cargo test --release --lib --features r1-cuda-ep`：21 passed、0 failed、3 ignored；3 个忽略项需要外部模型包或 CUDA Runtime。
- `cargo check --release --features "r0-profiling r2-trt-native" --example r2_1_trt_native_cancel`：通过。
- 原生 TensorRT 示例完成 30+20 热态 E2E。无 ORT GPT 节点回退：该后端分支不创建 ORT GPT Encoder/Step Session，而是直接将两个 ONNX 图建成 TensorRT Engine 并通过 `enqueueV3` 执行。
- 直接 `cargo test --release --lib` 的默认链接失败，因为本机没有配置 ORT 导入库；通过 `r1-cuda-ep` 动态 ORT 构建执行相同单测后通过。R2.1 feature 已启用 `ort/load-dynamic`，运行时必须显式提供 ORT DLL 路径。

首次运行中修正了两个 TensorRT shape/buffer 问题：Step Engine 将 `x_len`/`y_len` 暴露为动态绑定，现按 ABI-1 的长度为 1 设置输入 shape；CUDA 分配器原先把 8 字节 INT64 缓冲区向上补齐到 16 字节，导致合法标量校验误报，现保留精确字节数。

## 瓶颈与下一阶段

TensorRT 将 GPT Step 压低约四倍后，CPU VITS 曾占总耗时约 65%。R2.2 已单独验证可选 CUDA VITS：它将 TTFA P50 降低约 23%、完整耗时降约 39%，但仍保留 CPU 辅助节点，且主观盲听尚未完成，故未改变默认路径。单 Token TRT Step 仍包含一次 CUDA Stream 同步，且每步仍把 top-50 输出传回 CPU。CPU Sampling 本身只约 0.116ms/请求，单独改它预计收益有限；若要减少每 Token 同步，必须在独立实验中把采样 Token 保持在 GPU，并按两步批量读回 EOS，同时验证 LCG Token 序列、取消响应和 Chunk 边界。

参考项目记录的 GPU Sampling CUDA Graph、每两步 EOS 检查、25 秒 Warm Retention 与首段 Preview 均需按本项目 ABI 和质量门单独验证。Preview 已有 R1.8/R1.9 低相关接缝失败记录，因此不启用。CUDA Graph 全 GPT Step 尚未在原生 TRT 路径验证。现阶段未迁移 Frontend/其他图至原生 TRT，也未进行跨语言语料、不同 GPU 或 DirectML 实机 E2E。

本地 TensorRT 图以 FP32 构建，并关闭 TF32；这与 AnifLive-TTS 当前 FP16 环境不同，所以两边公开的绝对 TTFA/RTF 不可直接对比。本文只报告同一设备上 CPU GPT 与原生 TensorRT GPT 的受控 A/B。
