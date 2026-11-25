"""Export and check experimental solver graphs with the same ABI-2 contract."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
from unittest.mock import patch

import torch


TOOLS = Path(__file__).resolve().parent


def export_variants(model, exporter, helper, family, variants, output):
    _, steps, guidance, _, maximum = exporter["FAMILIES"][family]
    _, conditioner, serial, _ = exporter["make_graphs"](torch, model, family, steps, guidance)
    device = next(serial.parameters()).device
    generator = torch.Generator(device=device).manual_seed(1234)
    validations = {}
    for variant in variants:
        _, _, graph, _ = exporter["make_graphs"](
            torch, model, family, steps, guidance, step_variant=variant,
        )
        probes = {}
        temperature = 0.875 if family.startswith("v5") else 1.0
        for frames in (400, 592, min(965, maximum)):
            features = torch.randn((1, 512, frames), generator=generator, device=device)
            prompt = torch.randn((1, 100, frames), generator=generator, device=device)
            prompt[..., 327:] = 0
            mask = torch.ones((1, 1, frames), device=device)
            mask[..., :327] = 0
            state = torch.randn(prompt.shape, generator=generator, device=device) * (mask * temperature)
            time = torch.tensor([0.125], dtype=torch.float32, device=device)
            frequencies, scale = serial.dit.rotary_embed.forward_from_seq_len(frames)
            if scale != 1:
                raise ValueError("unexpected rotary scale")
            condition = conditioner(features, prompt)
            conditions = condition if isinstance(condition, tuple) else (condition,)
            values = (state, time, mask, frequencies.float(), *conditions)
            expected, actual = serial(*values), graph(*values)
            if actual.shape != expected.shape or not torch.isfinite(actual).all():
                raise ValueError(f"invalid {variant} step result")
            difference = actual - expected
            probes[str(frames)] = {
                "relative_l2": (torch.linalg.vector_norm(difference)
                                / torch.linalg.vector_norm(expected).clamp_min(1e-8)).item(),
                "max_abs": difference.abs().max().item(),
            }
        destination = output / variant
        destination.mkdir(parents=True, exist_ok=False)
        names = ["state", "time", "generation_mask", "rope", "condition"]
        if guidance:
            names.append("negative")
        dynamic = {
            "state": {2: "frames"}, "generation_mask": {2: "frames"}, "rope": {1: "frames"},
            "condition": {1: "frames"}, "state_next": {2: "frames"},
        }
        if guidance:
            dynamic["negative"] = {1: "frames"}
        with ExitStack() as cleanup:
            for norm in graph.modules():
                if isinstance(norm, torch.nn.LayerNorm):
                    cleanup.enter_context(patch.object(
                        norm, "forward", lambda x, norm=norm: helper["fp32_layer_norm"](torch, x, norm),
                    ))
            exporter["export_graph"](
                torch, destination / "dit_step.onnx", graph, values, names, ["state_next"],
                dynamic, fold_constants=family.startswith("v5"),
            )
        validations[variant] = probes
        print(variant, probes, flush=True)
    (output / "validation.json").write_text(json.dumps(validations, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variant", action="append", required=True, choices=["unmasked", "batched"])
    args = parser.parse_args()
    exporter = runpy.run_path(str(TOOLS / "export-native-diffusion.py"))
    spec = exporter["checkpoint_spec"](args.checkpoint.resolve(), args.runtime.resolve(), "fp16")
    if "batched" in args.variant and spec["family"] != "v5dev":
        parser.error("batched CFG requires a V5 Dev checkpoint")
    helper = runpy.run_path(str(TOOLS / "gpt-sovits-tensorrt.py"))
    worker = runpy.run_path(str(TOOLS / "gpt-sovits-worker.py"))
    model, _, _, _ = worker["load"](spec)
    with torch.inference_mode():
        export_variants(model, exporter, helper, spec["family"], args.variant, args.output.resolve())


if __name__ == "__main__":
    main()
