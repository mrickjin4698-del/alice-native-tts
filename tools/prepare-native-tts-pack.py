"""Create an ABI-1 pack from the validated v2Pro exports and a GPT-SoVITS install.
Run with the install's Python. Python is needed only for pack preparation.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--exports", type=Path, required=True)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--transcript", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--streaming-graph", type=Path, help="Optional validated sovits_stream.onnx")
args = parser.parse_args()
streaming = None
if args.streaming_graph:
    report = json.loads(args.streaming_graph.with_suffix(".validation.json").read_text(encoding="utf-8"))
    streaming = {"abi": 1, "graph": "sovits_stream.onnx", "latent_channels": 192, "samples_per_token": 1280}
    if report.get("streaming") != streaming or len(report.get("validation", [])) < 4:
        raise RuntimeError("Missing supported streaming graph validation")
    import hashlib
    if report.get("graph_sha256") != hashlib.sha256(args.streaming_graph.read_bytes()).hexdigest():
        raise RuntimeError("Streaming graph does not match its validation report")
    import math
    for row in report["validation"]:
        if {v["output"] for v in row["outputs"]} != {"audio", "latent"}:
            raise RuntimeError("Incomplete streaming validation outputs")
        for value in row["outputs"]:
            error = value["original_relative_l2"]
            if not math.isfinite(error) or error > 2e-4:
                raise RuntimeError("Streaming graph differs from the original model")
os.environ["NLTK_DATA"] = str(args.source / "runtime/nltk_data")
import numpy as np
from nltk.corpus import cmudict
import g2p_en

out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)

def copy(source, name):
    target = out / name
    if target.exists():
        if target.resolve() == source.resolve():
            return
        raise RuntimeError(f"Refusing to overwrite existing pack file: {target}")
    shutil.copy2(source, target)

for name in ("bert", "ssl", "gpt_encoder", "gpt_step", "sovits", "spectrogram", "sv_embedding", "vq_encoder"):
    copy(args.exports / f"{name}.onnx", f"{name}.onnx")
copy(args.exports / "config.json", "config.json")
copy(args.source / "GPT_SoVITS/text/G2PWModel/g2pW.onnx", "g2pw.onnx")
copy(args.reference, "reference.wav")
if streaming:
    copy(args.streaming_graph, streaming["graph"])
cmu = {word: variants[0] for word, variants in cmudict.dict().items()}
(out / "cmudict.json").write_text(json.dumps(cmu, separators=(",", ":")), encoding="utf-8")
model = g2p_en.G2p()
weights = np.load(Path(g2p_en.__file__).parent / "checkpoint20.npz")
data = {name: weights[name].reshape(-1).tolist() for name in weights.files}
data["phonemes"] = model.phonemes
(out / "g2p-en.json").write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
# Retain actual source license files instead of silently assigning one license
# to all pretrained/custom weights. Voice rights remain with the pack author.
licenses = [
    "Alice GPT-SoVITS v2Pro model pack ABI 1",
    "GPT-SoVITS: MIT — https://github.com/RVC-Boss/GPT-SoVITS",
    "Minimal export wrappers: https://github.com/GPT-SoVITS-Devel/GPT-SoVITS_minimal_inference",
    "G2PW: Apache-2.0 — https://github.com/GitYCC/g2pW",
    "Chinese RoBERTa: https://huggingface.co/hfl/chinese-roberta-wwm-ext-large",
    "Chinese HuBERT: https://huggingface.co/TencentGameMate/chinese-hubert-base",
    "ERes2Net: https://github.com/modelscope/3D-Speaker",
    "g2p_en: Apache-2.0 — https://github.com/Kyubyong/g2p",
    "CMUdict: https://github.com/cmusphinx/cmudict (retained license below)",
    "Custom voice weights: supplied locally by the pack author; do not redistribute without applicable voice/model rights.",
]
for path in [args.source / "LICENSE", args.source / "runtime/nltk_data/corpora/cmudict/LICENSE"]:
    if path.is_file():
        licenses.extend(["\n" + str(path.relative_to(args.source)), path.read_text(encoding="utf-8", errors="replace")])
try:
    licenses.extend(["\nCMUdict corpus license", cmudict.open("LICENSE").read()])
except (OSError, LookupError):
    pass
(out / "LICENSES.txt").write_text("\n".join(licenses), encoding="utf-8")
manifest = {"abi": 1, "architecture": "v2Pro", "sample_rate": 32000, "max_len": 512,
            "reference_audio": "reference.wav", "reference_text": args.transcript}
if streaming:
    manifest["streaming"] = streaming
(out / "pack.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps({"pack": str(out), "files": len(list(out.iterdir())), "bytes": sum(p.stat().st_size for p in out.iterdir())}))
