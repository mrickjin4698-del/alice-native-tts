"""Prepare an experimental mixed-precision GPT pair for ABI-2."""

import argparse
import json
from pathlib import Path

import onnx
from onnxruntime.transformers.float16 import DEFAULT_OP_BLOCK_LIST, convert_float_to_float16
from onnxruntime.transformers.onnx_model import OnnxModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fp16-softmax", action="store_true", help="Experiment with FP16 attention softmax outputs")
    args = parser.parse_args()
    pack = json.loads((args.pack / "pack.json").read_text(encoding="utf-8"))
    if pack.get("abi") != 2:
        parser.error("expected an ABI-2 pack")
    args.output.mkdir(parents=True, exist_ok=False)
    blocked = set(DEFAULT_OP_BLOCK_LIST) | {
        "LayerNormalization", "Softmax", "ReduceMean", "Sqrt", "TopK",
    }
    if args.fp16_softmax:
        blocked.remove("Softmax")
    blocked = sorted(blocked)
    for name in ("gpt_encoder.onnx", "gpt_step_active_len.onnx"):
        source = onnx.load(str(args.pack / name))
        converted = OnnxModel(convert_float_to_float16(
            source, keep_io_types=["bert_feature", "topk_values"], op_block_list=blocked,
        ))
        converted.topological_sort(is_deterministic=True)
        destination = args.output / name
        onnx.save(converted.model, str(destination))
        onnx.checker.check_model(str(destination))
        print(f"Wrote {destination}", flush=True)
    (args.output / "gpt.json").write_text(json.dumps({
        "gpt_checkpoint_sha256": pack["gpt_checkpoint_sha256"],
        "cache": "fp16",
        "blocked_operators": blocked,
    }, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
