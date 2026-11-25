"""Build local TensorRT 11 plans from ABI-2 acoustic ONNX graphs."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


PROFILES = {
    "v3": (934, 612),
    "v4": (1000, 612),
    "v5turbo": (1000, 1024),
    "v5dev": (1000, 1024),
}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profiles(pack):
    family = pack["family"]
    max_frames, channels = PROFILES[family]
    defaults = ["--noTF32", "--maxAuxStreams=0"]
    acoustic = ["--builderOptimizationLevel=2", "--memPoolSize=workspace:2048"]
    if family != "v3":
        yield "target_features", [
            f"codes:1x1x{tokens},phones:1x{phones}"
            for tokens, phones in ((1, 1), (64, 24), (511, 511))
        ], defaults + [
            "--builderOptimizationLevel=3",
            "--memPoolSize=workspace:4096",
            "--tacticSources=-EDGE_MASK_CONVOLUTIONS,-JIT_CONVOLUTIONS",
        ]
    yield "vocoder", [
        f"mel:1x100x{frames}" for frames in (1, 256, 2048)
    ], defaults + acoustic
    window = (2, 512, max_frames)
    yield "dit_condition", [
        f"features:1x{pack['feature_channels']}x{frames},prompt:1x100x{frames}"
        for frames in window
    ], defaults + acoustic
    steps = []
    for frames in window:
        shape = (
            f"state:1x100x{frames},generation_mask:1x1x{frames},"
            f"rope:1x{frames}x{pack['rope_channels']},condition:1x{frames}x{channels}"
        )
        if family == "v5dev":
            shape += f",negative:1x{frames}x{channels}"
        steps.append(shape)
    yield "dit_step", steps, defaults + acoustic


def build_pack(trtexec, pack_dir, output, step_root, step_variant):
    pack = json.loads((pack_dir / "pack.json").read_text(encoding="utf-8"))
    if pack.get("abi") != 2 or pack.get("family") not in PROFILES or pack.get("precision") != "fp16":
        raise ValueError(f"expected a supported FP16 ABI-2 pack: {pack_dir}")
    target = output / pack["family"]
    target.mkdir(parents=True, exist_ok=True)
    manifest = {
        "family": pack["family"],
        "sovits_checkpoint_sha256": pack["sovits_checkpoint_sha256"],
        "graphs": {},
    }
    manifest_path = target / "plans.json"
    previous = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    for name, shapes, options in profiles(pack):
        source = pack_dir / f"{name}.onnx"
        if name == "dit_step" and step_root is not None:
            source = step_root / pack["family"] / step_variant / "dit_step.onnx"
        plan = target / f"{name}.plan"
        fingerprint = sha256(source)
        settings = {"onnx_sha256": fingerprint, "shapes": shapes, "options": options}
        before = previous.get("graphs", {}).get(name, {})
        if plan.is_file() and all(before.get(key) == value for key, value in settings.items()):
            if before.get("plan_sha256") == sha256(plan):
                manifest["graphs"][name] = before
                continue
        print(f"Building {pack['family']} {name}", flush=True)
        with tempfile.TemporaryDirectory(prefix=".build-", dir=target) as pending:
            built = Path(pending) / plan.name
            command = [
                str(trtexec), f"--onnx={source}", f"--saveEngine={built}", "--skipInference",
                *options, *(f"--{bound}Shapes={shape}" for bound, shape in zip(("min", "opt", "max"), shapes)),
            ]
            log_path = target / f"{name}.build.log"
            with log_path.open("w", encoding="utf-8") as log:
                if subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False).returncode:
                    raise RuntimeError(f"TensorRT build failed. See {log_path}")
            if not built.is_file() or built.stat().st_size == 0:
                raise RuntimeError(f"TensorRT wrote no plan. See {log_path}")
            settings["plan_sha256"] = sha256(built)
            built.replace(plan)
        manifest["graphs"][name] = settings
    temporary = manifest_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    temporary.replace(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", action="append", type=Path, required=True)
    parser.add_argument("--trtexec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-root", type=Path, help="Solver exports organized by model family")
    parser.add_argument("--step-variant", choices=["unmasked", "batched"], default="unmasked")
    args = parser.parse_args()
    for pack in args.pack:
        build_pack(
            args.trtexec.resolve(), pack.resolve(), args.output.resolve(),
            args.step_root.resolve() if args.step_root else None, args.step_variant,
        )


if __name__ == "__main__":
    main()
