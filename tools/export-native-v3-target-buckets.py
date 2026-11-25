"""Export fixed-shape CUDA target graphs with explicit valid lengths for v3."""

import argparse
import json
from pathlib import Path
import runpy

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


TOOLS = Path(__file__).resolve().parent
BUCKETS = ((96, 64), (256, 128))


class Bucket(nn.Module):
    def __init__(self, vits):
        super().__init__()
        self.vits = vits

    def forward(self, codes, phones, speaker, code_count, phone_count):
        m = self.vits
        speaker = speaker.to(next(m.parameters()).dtype)
        quantized = F.interpolate(m.quantizer.decode(codes), scale_factor=2.0, mode="nearest")
        hidden = m.enc_p(quantized, code_count * 2, phones, phone_count, speaker, 1)[0]
        features = F.interpolate(m.bridge(hidden), scale_factor=1.875, mode="nearest")
        # Unpadded v3 uses a length slightly greater than the resampled tensor.
        # The mask here keeps that prefix valid and makes padding zero at each WN layer.
        valid = torch.div(code_count * 15, 4, rounding_mode="floor")
        return m.wns1(features, valid, speaker)[0].float()


def export(model, pack_dir, output, exporter):
    target, _, _, _ = exporter["make_graphs"](torch, model, "v3", 32, 0.0)
    bucket = Bucket(model.vits_model)
    device = next(model.vits_model.parameters()).device
    speaker = torch.from_numpy(np.fromfile(pack_dir / "reference_speaker.f32", dtype="<f4")).reshape(1, -1, 1).to(device)
    generator = torch.Generator(device=device).manual_seed(1234)
    symbols = model.vits_model.enc_p.text_embedding.num_embeddings
    validation = {}
    output.mkdir(parents=True, exist_ok=False)
    for tokens, phones in BUCKETS:
        name = f"target_features-{tokens}x{phones}"
        probes = []
        for code_count, phone_count in ((1, 1), (min(tokens, 67), 25), (tokens - 1, phones - 1)):
            codes = torch.randint(0, 1024, (1, 1, code_count), generator=generator, device=device)
            text = torch.randint(0, symbols, (1, phone_count), generator=generator, device=device)
            values = (
                F.pad(codes, (0, tokens - code_count), value=0),
                F.pad(text, (0, phones - phone_count), value=0),
                speaker,
                torch.tensor([code_count], device=device),
                torch.tensor([phone_count], device=device),
            )
            expected = target(codes, text, speaker)
            actual = bucket(*values)[..., :expected.shape[-1]]
            stats = exporter["compare"](torch, name, actual, expected, "fp16")
            probes.append({"tokens": code_count, "phones": phone_count, **stats})
        exporter["export_graph"](
            torch, output / (name + ".onnx"), bucket, values,
            ["codes", "phones", "speaker", "code_count", "phone_count"], ["features"], {},
        )
        validation[name] = probes
    (output / "validation.json").write_text(json.dumps(validation, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    exporter = runpy.run_path(str(TOOLS / "export-native-diffusion.py"))
    spec = exporter["checkpoint_spec"](args.checkpoint.resolve(), args.runtime.resolve(), "fp16")
    pack = args.pack.resolve()
    metadata = json.loads((pack / "pack.json").read_text(encoding="utf-8"))
    if spec["family"] != "v3" or metadata["family"] != "v3":
        parser.error("requires matched v3 checkpoint and ABI-2 pack")
    helper = runpy.run_path(str(TOOLS / "gpt-sovits-tensorrt.py"))
    if helper["checkpoint_digest"](spec["sovits"]) != metadata["sovits_checkpoint_sha256"]:
        parser.error("SoVITS checkpoint differs from the pack")
    worker = runpy.run_path(str(TOOLS / "gpt-sovits-worker.py"))
    model, _, _, _ = worker["load"](spec)
    with torch.inference_mode():
        export(model, pack, args.output.resolve(), exporter)


if __name__ == "__main__":
    main()
