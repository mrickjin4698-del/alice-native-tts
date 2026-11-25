"""TensorRT execution of a prebuilt Irodori RF-DiT denoiser."""

import hashlib
import json
from pathlib import Path
import sys


def verified_plan(plan_path, weights, hash_key):
    plan = Path(plan_path)
    metadata = json.loads(plan.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["schema_version"] != 1:
        raise ValueError(f"unsupported Irodori-TTS engine metadata in {plan}")
    checksum = hashlib.sha256()
    with open(weights, "rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            checksum.update(block)
    if checksum.hexdigest() != metadata[hash_key]:
        raise ValueError(f"Irodori-TTS engine {plan} was built from different weights")
    plan_bytes = plan.read_bytes()
    if hashlib.sha256(plan_bytes).hexdigest() != metadata["plan_sha256"]:
        raise ValueError(f"Irodori-TTS engine {plan} does not match its metadata")
    return metadata, plan_bytes


class ConditionEncodingCache:
    def __init__(self, torch, encode):
        self.torch = torch
        self.encode = encode
        self.clear()

    def clear(self):
        self.previous = None
        self.hits = 0

    def same_input(self, current, previous):
        if current is previous:
            return True
        if self.torch.is_tensor(current) and self.torch.is_tensor(previous):
            return (
                current.dtype == previous.dtype
                and current.device == previous.device
                and current.stride() == previous.stride()
                and self.torch.equal(current, previous)
            )
        return isinstance(current, str) and isinstance(previous, str) and current == previous

    def __call__(self, *args, **kwargs):
        if self.previous is not None:
            old_args, old_kwargs, states = self.previous
            if (
                len(args) == len(old_args)
                and kwargs.keys() == old_kwargs.keys()
                and all(self.same_input(a, b) for a, b in zip(args, old_args))
                and all(self.same_input(value, old_kwargs[key]) for key, value in kwargs.items())
            ):
                self.hits += 1
                return states
        states = self.encode(*args, **kwargs)
        self.previous = (args, kwargs, states)
        return states


class TensorRtDenoiser:
    def __init__(self, model, plan, checkpoint):
        import tensorrt as trt
        import torch

        if model.cfg.flow_parameterization != "rf_velocity":
            raise ValueError("the TensorRT denoiser expects an RF velocity checkpoint")
        metadata, plan_bytes = verified_plan(plan, checkpoint, "checkpoint_sha256")
        self.torch = torch
        self.model = model
        self.eager = model.forward_with_encoded_conditions
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(plan_bytes)
        if self.engine is None:
            raise ValueError(f"could not deserialize Irodori-TTS engine {plan}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise ValueError(f"could not create an execution context for {plan}")
        cache_type, self.context_dtype = {
            "fp16": (trt.float16, torch.float16),
            "fp32": (trt.float32, torch.float32),
        }[metadata["precision"]]
        types = {
            "latent": trt.float32, "timestep": trt.float32, "context_mask": trt.bool,
            "context_k": cache_type, "context_v": cache_type,
        }
        signature = {name: (kind, trt.TensorIOMode.INPUT) for name, kind in types.items()}
        signature["velocity"] = (trt.float32, trt.TensorIOMode.OUTPUT)
        actual = {
            name: (self.engine.get_tensor_dtype(name), self.engine.get_tensor_mode(name))
            for index in range(self.engine.num_io_tensors)
            for name in (self.engine.get_tensor_name(index),)
        }
        if signature != actual:
            raise ValueError(f"unexpected Irodori-TTS engine I/O signature in {plan}")
        self.inputs = tuple(types)
        self.profiles = {name: self.engine.get_tensor_profile_shape(name, 0) for name in self.inputs}
        self.shapes = {}
        self.warned = False
        # Duration prediction and RF sampling encode the same inputs within a
        # request. The caption encoder depends only on the fixed model and its
        # own token IDs and mask, so its last result can span requests.
        self.encode_conditions = ConditionEncodingCache(torch, model.encode_conditions)
        self.caption_encoding = None
        if model.caption_encoder is not None:
            self.caption_encoding = ConditionEncodingCache(torch, model.caption_encoder.forward)
            model.caption_encoder.forward = self.caption_encoding
        self.reset_request()

    def reset_request(self):
        self.encode_conditions.clear()
        if self.caption_encoding is not None:
            self.caption_encoding.hits = 0
        self.condition_cache = {}
        self.executed_steps = 0
        self.eager_steps = 0

    def pack_conditions(self, cache, text_mask, speaker_mask, caption_mask):
        key = id(cache)
        if key in self.condition_cache:
            return self.condition_cache[key][1]

        torch = self.torch
        attention = self.model.blocks[0].attention
        masks = [text_mask]
        if attention.has_speaker_condition:
            masks.append(speaker_mask)
        if attention.has_caption_condition:
            masks.append(caption_mask)
        masks = [
            mask if mask is not None else torch.ones(
                layer_kv.shape[:2], dtype=torch.bool, device=layer_kv.device
            )
            for mask, layer_kv in zip(masks, cache[0][::2], strict=True)
        ]
        # Drop only trailing positions masked in every CFG branch. The remaining
        # mask preserves differing text, speaker and caption visibility per branch.
        extents = torch.stack([
            torch.where(mask.any(dim=0), torch.arange(1, mask.shape[1] + 1, device=mask.device), 0).amax()
            for mask in masks
        ]).tolist()
        if not any(extents):
            extents[0] = 1
        context_mask = torch.cat([mask[:, :length] for mask, length in zip(masks, extents) if length], dim=1)
        packed = []
        for offset in (0, 1):
            layers = [
                torch.cat([tensor[:, :length] for tensor, length in zip(layer[offset::2], extents) if length], dim=1)
                for layer in cache
            ]
            packed.append(torch.stack(layers).to(self.context_dtype))
        result = (context_mask, *packed)
        # Keep the list alive to prevent its identity from being reused during a request.
        self.condition_cache[key] = (cache, result)
        return result

    def fallback(self, kwargs):
        self.eager_steps += 1
        if not self.warned:
            print("Irodori-TTS denoiser inputs exceed its TensorRT profile or require the PyTorch path", file=sys.stderr)
            self.warned = True
        return self.eager(**kwargs)

    def forward(self, **kwargs):
        x, timestep, cache = (kwargs[name] for name in ("x_t", "t", "context_kv_cache"))
        if cache is None or kwargs.get("latent_mask") is not None or kwargs.get("delta_t") is not None:
            return self.fallback(kwargs)
        mask, keys, values = self.pack_conditions(
            cache, kwargs["text_mask"], kwargs.get("speaker_mask"), kwargs.get("caption_mask")
        )
        tensors = (x, timestep, mask, keys, values)
        for name, tensor in zip(self.inputs, tensors, strict=True):
            minimum, _, maximum = self.profiles[name]
            shape = tuple(tensor.shape)
            if len(shape) != len(minimum) or any(
                size < low or size > high for size, low, high in zip(shape, minimum, maximum)
            ):
                return self.fallback(kwargs)
        for name, tensor in zip(self.inputs, tensors, strict=True):
            shape = tuple(tensor.shape)
            if shape != self.shapes.get(name):
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(f"invalid Irodori-TTS engine shape for {name}: {shape}")
                self.shapes[name] = shape
            if not self.context.set_tensor_address(name, tensor.data_ptr()):
                raise RuntimeError(f"could not bind Irodori-TTS engine input {name}")
        if tuple(self.context.get_tensor_shape("velocity")) != tuple(x.shape):
            raise RuntimeError("Irodori-TTS engine returned an invalid velocity shape")
        velocity = self.torch.empty_like(x)
        if not self.context.set_tensor_address("velocity", velocity.data_ptr()):
            raise RuntimeError("could not bind Irodori-TTS engine output")
        stream = self.torch.cuda.current_stream(x.device).cuda_stream
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("Irodori-TTS TensorRT denoising failed")
        self.executed_steps += 1
        return velocity


class TensorRtDecoder:
    def __init__(self, codec, plan, weights):
        import tensorrt as trt
        import torch

        metadata, data = verified_plan(plan, weights, "weights_sha256")
        if metadata["sample_rate"] != codec.sample_rate or metadata["hop_length"] != codec.model.hop_length:
            raise ValueError("Irodori-TTS decoder audio format differs from the loaded codec")
        self.torch = torch
        self.eager = codec.decode_latent
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self.runtime.deserialize_cuda_engine(data)
        if (
            self.engine is None
            or self.engine.num_io_tensors != 2
            or self.engine.get_tensor_mode("latent") != trt.TensorIOMode.INPUT
            or self.engine.get_tensor_mode("pcm") != trt.TensorIOMode.OUTPUT
            or self.engine.get_tensor_dtype("latent") != trt.float32
            or self.engine.get_tensor_dtype("pcm") != trt.float32
        ):
            raise ValueError(f"unexpected Irodori-TTS decoder engine I/O in {plan}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise ValueError(f"could not create an execution context for {plan}")
        self.minimum, _, self.maximum = self.engine.get_tensor_profile_shape("latent", 0)
        self.shape = None
        self.last_backend = None

    def __call__(self, latent):
        shape = tuple(latent.shape)
        if len(shape) != len(self.minimum) or any(
            size < low or size > high for size, low, high in zip(shape, self.minimum, self.maximum)
        ):
            self.last_backend = "pytorch"
            return self.eager(latent)
        if shape != self.shape:
            if not self.context.set_input_shape("latent", shape):
                raise RuntimeError(f"invalid Irodori-TTS decoder latent shape {shape}")
            self.shape = shape
        output_shape = tuple(self.context.get_tensor_shape("pcm"))
        if len(output_shape) != 3 or output_shape[:2] != (shape[0], 1) or output_shape[2] <= 0:
            raise RuntimeError(f"invalid Irodori-TTS decoder output shape {output_shape}")
        latent = latent.contiguous().float()
        pcm = self.torch.empty(output_shape, dtype=self.torch.float32, device=latent.device)
        if (
            not self.context.set_tensor_address("latent", latent.data_ptr())
            or not self.context.set_tensor_address("pcm", pcm.data_ptr())
            or not self.context.execute_async_v3(self.torch.cuda.current_stream(latent.device).cuda_stream)
        ):
            raise RuntimeError("Irodori-TTS TensorRT waveform decoding failed")
        self.last_backend = "tensor_rt"
        return pcm
