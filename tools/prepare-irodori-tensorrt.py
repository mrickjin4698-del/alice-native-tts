"""Export Irodori's RF-DiT velocity network and build a TensorRT engine."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch
from torch import nn
from torch.nn import functional as F


def rotate_half_heads(q, cos, sin):
    rotated, plain = q.chunk(2, dim=2)
    even, odd = rotated.float().unflatten(-1, (-1, 2)).unbind(-1)
    rotated = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return torch.cat((rotated.flatten(-2).to(q.dtype), plain), dim=2)


class Block(nn.Module):
    def __init__(self, block):
        super().__init__()
        attention = block.attention
        self.heads = attention.heads
        self.head_dim = attention.head_dim
        self.qkvg = nn.Parameter(
            torch.cat([layer.weight for layer in (attention.wq, attention.wk, attention.wv, attention.gate)]),
            requires_grad=False,
        )
        self.q_norm, self.k_norm, self.attention_out = attention.q_norm, attention.k_norm, attention.wo
        self.attention_adaln, self.mlp_adaln = block.attention_adaln, block.mlp_adaln
        self.mlp_in = nn.Parameter(
            torch.cat((block.mlp.w1.weight, block.mlp.w3.weight)), requires_grad=False
        )
        self.mlp_out = block.mlp.w2

    def forward(self, x, condition, cos, sin, mask, context_k, context_v):
        h, gate = self.attention_adaln(x, condition)
        q, k, v, attention_gate = F.linear(h, self.qkvg).chunk(4, dim=-1)
        q = self.q_norm(q.unflatten(-1, (self.heads, self.head_dim)))
        k = self.k_norm(k.unflatten(-1, (self.heads, self.head_dim)))
        q = rotate_half_heads(q, cos, sin).transpose(1, 2)
        k = rotate_half_heads(k, cos, sin)
        v = v.unflatten(-1, (self.heads, self.head_dim))
        y = F.scaled_dot_product_attention(
            q, torch.cat((k, context_k), dim=1).transpose(1, 2),
            torch.cat((v, context_v), dim=1).transpose(1, 2), attn_mask=mask,
        )
        y = y.transpose(1, 2).flatten(2) * torch.sigmoid(attention_gate)
        x = x + gate * self.attention_out(y)
        h, gate = self.mlp_adaln(x, condition)
        activation, value = F.linear(h, self.mlp_in).chunk(2, dim=-1)
        return x + gate * self.mlp_out(F.silu(activation) * value)


class Denoiser(nn.Module):
    def __init__(self, model, max_steps):
        super().__init__()
        from irodori_tts.model import get_timestep_embedding, precompute_freqs_cis

        self.timestep_embedding = get_timestep_embedding
        self.timestep_dim = model.cfg.timestep_embed_dim
        self.projection = model.in_proj
        self.condition = model.cond_module
        self.blocks = nn.ModuleList(Block(block) for block in model.blocks)
        self.output_norm, self.output = model.out_norm, model.out_proj
        freqs = precompute_freqs_cis(model.head_dim, max_steps)
        self.register_buffer("cos", freqs.real.contiguous()[None, :, None, :])
        self.register_buffer("sin", freqs.imag.contiguous()[None, :, None, :])

    def forward(self, latent, timestep, context_mask, context_k, context_v):
        dtype = self.projection.weight.dtype
        condition = self.condition(self.timestep_embedding(timestep, self.timestep_dim).to(dtype))[:, None]
        x = self.projection(latent.to(dtype))
        cos = self.cos[:, :latent.shape[1]]
        sin = self.sin[:, :latent.shape[1]]
        latent_mask = torch.ones_like(latent[:, :, 0], dtype=torch.bool)
        mask = torch.cat((latent_mask, context_mask), dim=-1)[:, None, None, :]
        for index, block in enumerate(self.blocks):
            x = block(x, condition, cos, sin, mask, context_k[index], context_v[index])
        return self.output(self.output_norm(x)).float()


def load_denoiser(runtime_root, checkpoint, max_steps):
    sys.path.insert(0, str(runtime_root))
    from irodori_tts.config import ModelConfig, merge_dataclass_overrides
    from irodori_tts.inference_runtime import _load_checkpoint_for_inference
    from irodori_tts.model import TextToLatentRFDiT

    weights, config, _, text_encoder = _load_checkpoint_for_inference(checkpoint)
    config = merge_dataclass_overrides(ModelConfig(), config, section="checkpoint model_config")
    if config.flow_parameterization != "rf_velocity":
        raise ValueError("the TensorRT denoiser export expects an RF velocity checkpoint")
    model = TextToLatentRFDiT(
        config, pretrained_backbone_config=text_encoder,
        load_pretrained_backbone_weights=not config.use_pretrained_text_encoder,
    )
    model.load_state_dict(weights, assign=True)
    return Denoiser(model, max_steps).eval(), config


INPUTS = ("latent", "timestep", "context_mask", "context_k", "context_v")


def input_shapes(config, batch, steps, context):
    kv = (config.num_layers, batch, context, config.num_heads, config.model_dim // config.num_heads)
    return ((batch, steps, config.patched_latent_dim), (batch,), (batch, context), kv, kv)


def profile(shapes):
    return ",".join(
        name + ":" + "x".join(str(extent) for extent in shape)
        for name, shape in zip(INPUTS, shapes, strict=True)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp16")
    parser.add_argument("--max-latent-steps", type=int, default=750)
    parser.add_argument("--max-context", type=int, default=512)
    args = parser.parse_args()
    if args.max_latent_steps < 1 or args.max_context < 1:
        parser.error("profile limits must be positive")

    checkpoint = args.checkpoint.resolve()
    plan = args.output.resolve()
    plan.parent.mkdir(parents=True, exist_ok=True)
    onnx_path = plan.with_suffix(".onnx")
    denoiser, config = load_denoiser(args.runtime_root.resolve(), checkpoint, args.max_latent_steps)
    dtype = torch.float16 if args.precision == "fp16" else torch.float32
    cos, sin = denoiser.cos, denoiser.sin
    denoiser.to(device="cuda", dtype=dtype)
    # Keep RoPE's tables at the checkpoint's FP32 frequency precision.
    denoiser.cos, denoiser.sin = cos.cuda(), sin.cuda()
    optimum = input_shapes(config, 3, min(126, args.max_latent_steps), min(32, args.max_context))
    example = tuple(
        torch.ones(shape, device="cuda", dtype=kind)
        for shape, kind in zip(optimum, (torch.float32, torch.float32, torch.bool, dtype, dtype), strict=True)
    )
    dynamic_axes = {
        "latent": {0: "batch", 1: "latent_steps"},
        "timestep": {0: "batch"},
        "context_mask": {0: "batch", 1: "context_steps"},
        "context_k": {1: "batch", 2: "context_steps"},
        "context_v": {1: "batch", 2: "context_steps"},
        "velocity": {0: "batch", 1: "latent_steps"},
    }
    with torch.no_grad():
        torch.onnx.export(
            denoiser, example, str(onnx_path), dynamo=False, opset_version=18,
            input_names=list(INPUTS), output_names=["velocity"], dynamic_axes=dynamic_axes,
        )
    del denoiser, example
    torch.cuda.empty_cache()
    print(f"Exported {onnx_path}", flush=True)

    command = [
        str(args.trtexec.resolve()), f"--onnx={onnx_path}", f"--saveEngine={plan}", "--skipInference",
        f"--minShapes={profile(input_shapes(config, 1, 1, 1))}", f"--optShapes={profile(optimum)}",
        f"--maxShapes={profile(input_shapes(config, 4, args.max_latent_steps, args.max_context))}",
        "--builderOptimizationLevel=2", "--memPoolSize=workspace:1536",
    ]
    if args.precision == "fp32":
        command.append("--noTF32")
    subprocess.run(command, check=True)
    metadata = {"schema_version": 1, "precision": args.precision}
    for kind, path in (("checkpoint", checkpoint), ("plan", plan)):
        with path.open("rb") as file:
            metadata[f"{kind}_sha256"] = hashlib.file_digest(file, "sha256").hexdigest()
    plan.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
