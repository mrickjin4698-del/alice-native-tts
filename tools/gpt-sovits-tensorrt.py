"""TensorRT engines for the GPT-SoVITS checkpoint inference worker."""

import hashlib
import io
import os
from pathlib import Path
import sys
import tempfile
from contextlib import ExitStack
from unittest.mock import patch


class CheckpointTensorRT:
    def __init__(self, torch):
        python_path = os.environ.get("ALICE_NATIVE_TTS_TRT_PYTHON_PATH")
        if python_path:
            sys.path.insert(0, str(Path(python_path).resolve()))
        self.dll_directory = None
        sdk = os.environ.get("ALICE_TRT_HOME")
        if sdk and os.name == "nt":
            bin_path = str(Path(sdk).resolve() / "bin")
            self.dll_directory = os.add_dll_directory(bin_path)
            os.environ["PATH"] = bin_path + os.pathsep + os.environ["PATH"]
        try:
            import tensorrt as trt
        except (ImportError, OSError) as error:
            raise RuntimeError("TensorRT requires NVIDIA TensorRT Python bindings and runtime libraries") from error

        cache_root = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache"
        self.cache = Path(os.environ.get("ALICE_NATIVE_TTS_CHECKPOINT_TRT_CACHE_DIR") or
                          Path(cache_root) / "alice-native-tts" / "tensorrt").resolve()
        self.cache.mkdir(parents=True, exist_ok=True)
        self.torch = torch
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        gpu = torch.cuda.get_device_properties(torch.cuda.current_device())
        self.device_key = f"{gpu.name}:{gpu.major}.{gpu.minor}:{trt.__version__}"
        self.stream = torch.cuda.Stream()
        self.stream.wait_stream(torch.cuda.current_stream())

    def stream_context(self):
        return self.torch.cuda.stream(self.stream)

    def engine(self, key, graph, example, inputs, outputs, dynamic_axes, profiles):
        label = key[0]
        digest = hashlib.sha256(repr((self.device_key, key, profiles)).encode()).hexdigest()
        plan = self.cache / f"{label}-{digest}.engine"
        if plan.is_file():
            engine = self.runtime.deserialize_cuda_engine(plan.read_bytes())
            if engine is not None:
                return TensorRTEngine(self, label, engine, inputs, outputs)

        print(f"[alice-tts] building TensorRT {label}", file=sys.stderr, flush=True)
        graph_bytes = io.BytesIO()
        with self.torch.inference_mode():
            self.torch.onnx.export(
                graph.eval(), example, graph_bytes, input_names=inputs, output_names=outputs,
                dynamic_axes=dynamic_axes, opset_version=17, dynamo=False,
            )
        trt = self.trt
        builder = trt.Builder(self.logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
        parser = trt.OnnxParser(network, self.logger)
        if not parser.parse(graph_bytes.getvalue()):
            diagnostics = "\n".join(str(parser.get_error(idx)) for idx in range(parser.num_errors))
            raise RuntimeError(f"TensorRT could not parse {label}:\n{diagnostics}")
        graph_bytes.close()
        config = builder.create_builder_config()
        config.builder_optimization_level = 2
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
        profile = builder.create_optimization_profile()
        for name, bounds in profiles.items():
            profile.set_shape(name, *bounds)
        if config.add_optimization_profile(profile) < 0:
            raise ValueError(f"could not add the TensorRT profile for {label}")
        serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError(f"TensorRT could not build {label}")
        engine = self.runtime.deserialize_cuda_engine(serialized)
        if engine is None:
            raise RuntimeError(f"TensorRT could not load the new {label} engine")
        # Readers only see completed, loadable engine plans, even with concurrent workers.
        with tempfile.NamedTemporaryFile(dir=self.cache, prefix=label, suffix=".tmp", delete=False) as stream:
            pending = Path(stream.name)
            try:
                stream.write(memoryview(serialized))
                stream.close()
                pending.replace(plan)
            finally:
                pending.unlink(missing_ok=True)
        return TensorRTEngine(self, label, engine, inputs, outputs)


class TensorRTEngine:
    def __init__(self, backend, label, engine, inputs, outputs):
        self.backend = backend
        self.label = label
        self.engine = engine
        self.context = engine.create_execution_context()
        if self.context is None:
            raise RuntimeError(f"could not create TensorRT execution context for {label}")
        dtypes = {backend.trt.float16: backend.torch.float16, backend.trt.float32: backend.torch.float32,
                  backend.trt.bool: backend.torch.bool}
        self.inputs = [(name, dtypes[engine.get_tensor_dtype(name)]) for name in inputs]
        self.outputs = [(name, dtypes[engine.get_tensor_dtype(name)]) for name in outputs]
        if any(engine.get_tensor_format(name) != backend.trt.TensorFormat.LINEAR for name in inputs + outputs):
            raise RuntimeError(f"TensorRT {label} requires linear PyTorch tensor bindings")
        self.calls = 0

    def __call__(self, *values):
        torch = self.backend.torch
        values = [value.contiguous() for value in values]
        if len(values) != len(self.inputs):
            raise ValueError(f"wrong number of TensorRT inputs for {self.label}")
        for (name, dtype), value in zip(self.inputs, values):
            if value.dtype != dtype or value.device.type != "cuda" or value.device != values[0].device:
                raise ValueError(f"invalid CUDA tensor for {self.label}.{name}, expected {dtype} on one device")
            if not self.context.set_input_shape(name, tuple(value.shape)):
                raise ValueError(f"shape {tuple(value.shape)} is outside the TensorRT profile for {self.label}.{name}")
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"could not bind TensorRT tensor {self.label}.{name}")
        result = []
        for name, dtype in self.outputs:
            shape = tuple(self.context.get_tensor_shape(name))
            if not shape or any(dim < 0 for dim in shape):
                raise RuntimeError(f"could not resolve TensorRT shape for {self.label}.{name}: {shape}")
            value = torch.empty(shape, device=values[0].device, dtype=dtype)
            if not self.context.set_tensor_address(name, value.data_ptr()):
                raise RuntimeError(f"could not bind TensorRT tensor {self.label}.{name}")
            result.append(value)
        stream = torch.cuda.current_stream(values[0].device)
        if not self.context.execute_async_v3(stream.cuda_stream):
            raise RuntimeError(f"TensorRT execution failed for {self.label}")
        self.calls += 1
        return result


def checkpoint_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def module_digest(module):
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy())
    return digest.hexdigest()


def fp32_layer_norm(torch, x, norm):
    weight = norm.weight.float() if norm.weight is not None else None
    bias = norm.bias.float() if norm.bias is not None else None
    return torch.nn.functional.layer_norm(x.float(), norm.normalized_shape, weight, bias, norm.eps).to(x.dtype)


def gpt_graph(torch, decoder, step):
    from torch.nn import functional as F

    class Transformer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = decoder.h.layers
            self.hidden = decoder.model_dim
            self.heads = decoder.num_head
            self.head_dim = self.hidden // self.heads

        def heads_first(self, x):
            return x.reshape(1, -1, self.heads, self.head_dim).transpose(1, 2)

        def forward(self, x, mask_or_keys, past_values=None):
            keys, values = [], []
            allowed = None if step else ~mask_or_keys
            for index, layer in enumerate(self.layers):
                attention = layer.self_attn
                q, k, v = F.linear(x, attention.in_proj_weight, attention.in_proj_bias).split(self.hidden, dim=-1)
                if step:
                    k = torch.cat((mask_or_keys[index], k), dim=1)
                    v = torch.cat((past_values[index], v), dim=1)
                attended = F.scaled_dot_product_attention(
                    self.heads_first(q), self.heads_first(k), self.heads_first(v), attn_mask=allowed,
                )
                attended = attended.transpose(1, 2).reshape(1, -1, self.hidden)
                x = fp32_layer_norm(torch, x + attention.out_proj(attended), layer.norm1)
                x = fp32_layer_norm(torch, x + layer.linear2(F.relu(layer.linear1(x))), layer.norm2)
                keys.append(k)
                values.append(v)
            return x, torch.stack(keys), torch.stack(values)

    return Transformer()


class TensorRTCheckpointGpt:
    def __init__(self, backend, decoder, checkpoint):
        if decoder.norm_first or decoder.model_dim != decoder.embedding_dim:
            raise ValueError("this GPT-SoVITS GPT architecture does not match the TensorRT exporter")
        torch = backend.torch
        weights = next(decoder.parameters())
        hidden = decoder.model_dim
        heads = decoder.num_head
        layers = decoder.num_layers
        weights_key = (1, checkpoint_digest(checkpoint), hidden, heads, layers)
        full = torch.zeros((1, 144, hidden), device=weights.device, dtype=weights.dtype)
        mask = torch.zeros((1, heads, 144, 144), device=weights.device, dtype=torch.bool)
        self.prefill = backend.engine(
            ("gpt-prefill", *weights_key), gpt_graph(torch, decoder, False), (full, mask),
            ["hidden", "mask"], ["encoded", "keys", "values"],
            {"hidden": {1: "prompt"}, "mask": {2: "prompt", 3: "prompt"},
             "encoded": {1: "prompt"}, "keys": {2: "prompt"}, "values": {2: "prompt"}},
            {"hidden": tuple((1, length, hidden) for length in (1, 144, 2048)),
             "mask": tuple((1, heads, length, length) for length in (1, 144, 2048))},
        )
        token = torch.zeros((1, 1, hidden), device=weights.device, dtype=weights.dtype)
        cache = torch.zeros((layers, 1, 192, hidden), device=weights.device, dtype=weights.dtype)
        cache_shapes = tuple((layers, 1, length, hidden) for length in (1, 192, 4096))
        self.step = backend.engine(
            ("gpt-step", *weights_key), gpt_graph(torch, decoder, True), (token, cache, cache.clone()),
            ["hidden", "past_keys", "past_values"], ["encoded", "keys", "values"],
            {"past_keys": {2: "past"}, "past_values": {2: "past"},
             "keys": {2: "present"}, "values": {2: "present"}},
            {"past_keys": cache_shapes, "past_values": cache_shapes},
        )

    def process_prompt(self, hidden, attn_mask, padding_mask=None, torch_sdpa=True):
        if padding_mask is not None or not torch_sdpa:
            raise ValueError("checkpoint TensorRT GPT expects single-item prompt inference")
        return self.prefill(hidden, attn_mask)

    def decode_next_token(self, hidden, keys, values, attn_mask=None, torch_sdpa=True):
        if attn_mask is not None or not torch_sdpa:
            raise ValueError("checkpoint TensorRT GPT expects single-item token inference")
        return self.step(hidden, keys, values)


def install_checkpoint_tensorrt(model, checkpoint, torch):
    backend = CheckpointTensorRT(torch)
    decoder = model.t2s_model.model
    gpt = TensorRTCheckpointGpt(backend, decoder, checkpoint)
    decoder.t2s_transformer = gpt
    waveform = install_tensorrt_waveform(backend, model)
    stages = (gpt.prefill, gpt.step, waveform)
    if model.vocoder is not None:
        stages += (install_tensorrt_dit(backend, model),)
    print(f"[alice-tts] TensorRT {backend.trt.__version__} ready: {', '.join(stage.label for stage in stages)}",
          file=sys.stderr, flush=True)
    return stages


def install_tensorrt_waveform(backend, model):
    torch = backend.torch
    vocoder = model.vocoder
    is_vits = vocoder is None
    module = model.vits_model.dec if is_vits else vocoder
    weight = next(module.parameters())
    channels = module.conv_pre.in_channels
    wave_input = torch.zeros((1, channels, 192), device=weight.device, dtype=weight.dtype)
    inputs = ["features"]
    example = (wave_input,)
    if is_vits:
        inputs.append("speaker")
        example += (torch.zeros((1, module.cond.in_channels, 1), device=weight.device, dtype=weight.dtype),)
    label = "vits-waveform" if is_vits else "vocoder"
    with ExitStack() as cleanup:
        if module.__class__.__name__ == "BigVGAN":
            patch_bigvgan_resampling(cleanup, module)
        engine = backend.engine(
            (label, 1, module.__class__.__name__, module_digest(module)), module, example, inputs, ["waveform"],
            {"features": {2: "frames"}, "waveform": {2: "samples"}},
            {"features": tuple((1, channels, length) for length in (1, 192, 4096))},
        )

    if is_vits:
        def forward(features, g=None):
            if g is None:
                raise ValueError("TensorRT VITS waveform decoder requires a speaker embedding")
            return engine(features, g)[0]
    else:
        def forward(features):
            return engine(features)[0]

    module.forward = forward
    return engine


def patch_bigvgan_resampling(cleanup, module):
    from BigVGAN.alias_free_activation.torch.filter import LowPassFilter1d
    from BigVGAN.alias_free_activation.torch.resample import UpSample1d
    from torch.nn import functional as F

    # The channel count comes from the preceding convolution and is fixed across
    # frame lengths. Give the exporter a constant grouped convolution kernel.
    def upsample(x, layer):
        channels = int(x.shape[1])
        kernel = layer.filter.repeat(channels, 1, 1)
        padded = F.pad(x, (layer.pad, layer.pad), mode="replicate")
        result = layer.ratio * F.conv_transpose1d(padded, kernel, stride=layer.stride, groups=channels)
        return result[..., layer.pad_left : -layer.pad_right]

    def lowpass(x, layer):
        channels = int(x.shape[1])
        kernel = layer.filter.repeat(channels, 1, 1)
        if layer.padding:
            x = F.pad(x, (layer.pad_left, layer.pad_right), mode=layer.padding_mode)
        return F.conv1d(x, kernel, stride=layer.stride, groups=channels)

    for layer in module.modules():
        if isinstance(layer, UpSample1d):
            cleanup.enter_context(patch.object(layer, "forward", lambda x, layer=layer: upsample(x, layer)))
        elif isinstance(layer, LowPassFilter1d):
            cleanup.enter_context(patch.object(layer, "forward", lambda x, layer=layer: lowpass(x, layer)))


def install_tensorrt_dit(backend, model):
    torch = backend.torch
    estimator = model.vits_model.cfm.estimator
    weight = next(estimator.parameters())
    channels = estimator.dim
    example_frames = 384
    frames = (1, example_frames, model.vocoder_configs["T_chunk"])
    example = torch.zeros((1, example_frames, channels), device=weight.device, dtype=weight.dtype)
    time = torch.zeros((1, channels), device=weight.device, dtype=weight.dtype)
    mask = torch.ones((1, example_frames), device=weight.device, dtype=torch.bool)
    freqs, scale = estimator.rotary_embed.forward_from_seq_len(example_frames)
    if not isinstance(scale, (float, int)) or scale != 1:
        raise ValueError("TensorRT DiT requires standard rotary embeddings")

    class DiTBlocks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = estimator.transformer_blocks

        def forward(self, x, time, mask, freqs):
            for block in self.blocks:
                x = block(x, time, mask=mask, rope=(freqs, 1.0))
            return x

    graph = DiTBlocks()
    # TensorRT normalization requires its input, weight and bias types to match.
    with ExitStack() as cleanup:
        for norm in graph.modules():
            if isinstance(norm, torch.nn.LayerNorm):
                cleanup.enter_context(patch.object(norm, "forward", lambda x, norm=norm: fp32_layer_norm(torch, x, norm)))
        engine = backend.engine(
            ("dit-blocks", 1, model.configs.version, module_digest(graph)), graph,
            (example, time, mask, freqs), ["hidden", "time", "mask", "rope"], ["denoised"],
            {"hidden": {1: "frames"}, "mask": {1: "frames"}, "rope": {1: "frames"},
             "denoised": {1: "frames"}},
            {"hidden": tuple((1, length, channels) for length in frames),
             "mask": tuple((1, length) for length in frames),
             "rope": tuple((1, length, freqs.shape[-1]) for length in frames)},
        )

    def execute(x, time, mask, rope):
        return engine(x, time, mask, rope[0])[0]

    if hasattr(estimator, "run_blocks"):
        def run_blocks(x, time, mask, rope, start=0, end=None, use_grad_ckpt=False):
            if start or (end is not None and end != estimator.depth) or use_grad_ckpt:
                raise ValueError("TensorRT DiT requires the complete inference block stack")
            return execute(x, time, mask, rope)

        estimator.run_blocks = run_blocks
    else:
        class TensorRTBlocks(torch.nn.Module):
            def forward(self, x, time, mask, rope):
                return execute(x, time, mask, rope)

        estimator.transformer_blocks = torch.nn.ModuleList([TensorRTBlocks()])
    return engine
