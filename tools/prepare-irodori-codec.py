"""Build a TensorRT FP16 waveform decoder from a local Irodori DACVAE checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import torch
from torch import nn


class Decoder(nn.Module):
    def __init__(self, codec):
        super().__init__()
        self.model = codec.model

    def forward(self, latent):
        return self.model.decode(latent.transpose(1, 2).contiguous().half()).float()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--codec-weights", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-latent-frames", type=int, default=750)
    args = parser.parse_args()
    if args.max_latent_frames < 1 or args.output.suffix.lower() != ".plan":
        parser.error("use a .plan output and a positive frame limit")

    sys.path.insert(0, str(args.runtime_root.resolve()))
    from irodori_tts.codec import DACVAECodec

    weights = args.codec_weights.resolve(strict=True)
    plan = args.output.resolve()
    plan.parent.mkdir(parents=True, exist_ok=True)
    onnx = plan.with_suffix(".onnx")
    codec = DACVAECodec.load(repo_id=str(weights), device="cuda", dtype=torch.float16)
    optimum = min(126, args.max_latent_frames)
    latent_dim = codec.latent_dim
    example = torch.zeros((1, optimum, latent_dim), device="cuda")
    decoder = Decoder(codec).eval()
    metadata = {
        "schema_version": 1, "sample_rate": codec.sample_rate,
        "hop_length": int(codec.model.hop_length), "precision": "fp16",
    }
    with torch.no_grad():
        torch.onnx.export(
            decoder, (example,), str(onnx), dynamo=False, opset_version=18,
            input_names=["latent"], output_names=["pcm"],
            dynamic_axes={"latent": {1: "frames"}, "pcm": {2: "samples"}},
        )
    del decoder, codec, example
    torch.cuda.empty_cache()
    trtexec = str(args.trtexec.resolve(strict=True))
    test_frames = sorted({min(frames, args.max_latent_frames) for frames in (58, 126, 184)})
    with tempfile.TemporaryDirectory(prefix="dacvae-build-", dir=plan.parent) as work:
        for level in (3, 2):
            candidate = Path(work) / f"decoder-opt{level}.plan"
            build = subprocess.run([
                trtexec, f"--onnx={onnx}", f"--saveEngine={candidate}", "--skipInference",
                f"--minShapes=latent:1x1x{latent_dim}", f"--optShapes=latent:1x{optimum}x{latent_dim}",
                f"--maxShapes=latent:1x{args.max_latent_frames}x{latent_dim}",
                f"--builderOptimizationLevel={level}", "--memPoolSize=workspace:1536",
            ])
            if build.returncode == 0 and all(
                subprocess.run([
                    trtexec, f"--loadEngine={candidate}", f"--shapes=latent:1x{frames}x{latent_dim}",
                    "--warmUp=0", "--duration=0", "--iterations=3", "--noDataTransfers", "--noCudaGraph",
                ]).returncode == 0
                for frames in test_frames
            ):
                candidate.replace(plan)
                break
        else:
            raise RuntimeError("no TensorRT DACVAE plan passed the inference smoke tests")
    for name, path in (("weights", weights), ("plan", plan)):
        with path.open("rb") as data:
            metadata[f"{name}_sha256"] = hashlib.file_digest(data, "sha256").hexdigest()
    plan.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
