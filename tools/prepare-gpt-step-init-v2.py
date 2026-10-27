#!/usr/bin/env python3
"""Create the first-token bridge graph for the opt-in per-layer GPT cache graph.

This is an offline model-pack preparation tool. It uses only Python's standard
library and leaves ABI-1 files untouched. The bridge accepts packed ABI-1 K/V
caches, splits them once on the selected EP, then exposes the per-layer cache
outputs expected by gpt_step_v2.onnx.
"""

from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

LAYERS = 24


def read_varint(data: bytes, pos: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if byte < 0x80:
            return value, pos
        shift += 7
        if shift >= 70:
            raise ValueError("invalid protobuf varint")


def encode_varint(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def encode_int_field(number: int, value: int) -> bytes:
    return encode_varint(number << 3) + encode_varint(value)


def encode_bytes_field(number: int, value: bytes) -> bytes:
    return encode_varint((number << 3) | 2) + encode_varint(len(value)) + value


def parse_fields(data: bytes) -> list[tuple[int, int, bytes, bytes]]:
    fields = []
    pos = 0
    while pos < len(data):
        start = pos
        tag, pos = read_varint(data, pos)
        number, wire = tag >> 3, tag & 7
        if wire == 0:
            value_start = pos
            _, pos = read_varint(data, pos)
            value = data[value_start:pos]
        elif wire == 1:
            value = data[pos : pos + 8]
            pos += 8
        elif wire == 2:
            size, pos = read_varint(data, pos)
            value = data[pos : pos + size]
            pos += size
        elif wire == 5:
            value = data[pos : pos + 4]
            pos += 4
        else:
            raise ValueError(f"unsupported protobuf wire type {wire}")
        fields.append((number, wire, value, data[start:pos]))
    return fields


def field_text(data: bytes, number: int) -> str | None:
    for field_number, wire, value, _ in parse_fields(data):
        if field_number == number and wire == 2:
            return value.decode("utf-8")
    return None


def make_split_node(name: str, packed_name: str, split_name: str, outputs: list[str]) -> bytes:
    attr = b"".join(
        [
            encode_bytes_field(1, b"axis"),
            encode_int_field(3, 0),
            encode_int_field(20, 2),  # AttributeProto.INT
        ]
    )
    parts = [
        encode_bytes_field(1, packed_name.encode()),
        encode_bytes_field(1, split_name.encode()),
        *(encode_bytes_field(2, name.encode()) for name in outputs),
        encode_bytes_field(3, name.encode()),
        encode_bytes_field(4, b"Split"),
        encode_bytes_field(5, attr),
    ]
    return b"".join(parts)


def make_split_sizes(name: str) -> bytes:
    dims = encode_varint(LAYERS)
    raw = struct.pack("<" + "q" * LAYERS, *([1] * LAYERS))
    return b"".join(
        [
            encode_bytes_field(1, dims),
            encode_int_field(2, 7),  # TensorProto.INT64
            encode_bytes_field(8, name.encode()),
            encode_bytes_field(9, raw),
        ]
    )


def main(pack: Path) -> None:
    base_path = pack / "gpt_step.onnx"
    step_v2_path = pack / "gpt_step_v2.onnx"
    output_path = pack / "gpt_step_init_v2.onnx"
    if not base_path.is_file() or not step_v2_path.is_file():
        raise FileNotFoundError("model pack must contain gpt_step.onnx and gpt_step_v2.onnx")

    model = base_path.read_bytes()
    v2 = step_v2_path.read_bytes()
    model_fields = parse_fields(model)
    v2_fields = parse_fields(v2)
    base_graphs = [value for number, wire, value, _ in model_fields if number == 7 and wire == 2]
    v2_graphs = [value for number, wire, value, _ in v2_fields if number == 7 and wire == 2]
    if len(base_graphs) != 1 or len(v2_graphs) != 1:
        raise ValueError("expected exactly one ONNX graph in each GPT Step model")

    base_graph_fields = parse_fields(base_graphs[0])
    packed_inputs = {}
    for number, wire, value, _ in base_graph_fields:
        if number == 11 and wire == 2:
            name = field_text(value, 1)
            if name in ("k_cache", "v_cache"):
                packed_inputs[name] = value
    if set(packed_inputs) != {"k_cache", "v_cache"}:
        raise ValueError("could not read packed K/V input definitions from gpt_step.onnx")

    cache_inputs: dict[str, bytes] = {}
    other_graph_fields = []
    original_nodes = []
    original_initializers = set()
    for number, wire, value, raw in parse_fields(v2_graphs[0]):
        if number == 1 and wire == 2:
            original_nodes.append(value)
        elif number == 5 and wire == 2:
            name = field_text(value, 8)
            if name:
                original_initializers.add(name)
            other_graph_fields.append(raw)
        elif number == 11 and wire == 2:
            name = field_text(value, 1)
            if name and (name.startswith("k_cache_layer_") or name.startswith("v_cache_layer_")):
                cache_inputs[name] = value
            else:
                other_graph_fields.append(raw)
        else:
            other_graph_fields.append(raw)

    expected_inputs = {
        f"{prefix}_cache_layer_{layer:02}"
        for prefix in ("k", "v")
        for layer in range(LAYERS)
    }
    if set(cache_inputs) != expected_inputs:
        raise ValueError(f"gpt_step_v2 cache input mismatch: found {len(cache_inputs)} layer inputs")
    if "/r1_7_k_split_sizes" in original_initializers or "/r1_7_v_split_sizes" in original_initializers:
        raise ValueError("split initializer names already exist in the source graph")

    # Move the former graph inputs to value_info so the new Split outputs retain
    # the exact type and symbolic dimensions expected by the per-layer graph.
    value_infos = b"".join(encode_bytes_field(13, info) for info in cache_inputs.values())
    packed_graph_inputs = b"".join(
        encode_bytes_field(11, packed_inputs[name]) for name in ("k_cache", "v_cache")
    )
    split_sizes = b"".join(
        [
            encode_bytes_field(5, make_split_sizes("/r1_7_k_split_sizes")),
            encode_bytes_field(5, make_split_sizes("/r1_7_v_split_sizes")),
        ]
    )
    split_nodes = b"".join(
        [
            encode_bytes_field(
                1,
                make_split_node(
                    "/R1_7_SplitKCache",
                    "k_cache",
                    "/r1_7_k_split_sizes",
                    [f"k_cache_layer_{layer:02}" for layer in range(LAYERS)],
                ),
            ),
            encode_bytes_field(
                1,
                make_split_node(
                    "/R1_7_SplitVCache",
                    "v_cache",
                    "/r1_7_v_split_sizes",
                    [f"v_cache_layer_{layer:02}" for layer in range(LAYERS)],
                ),
            ),
        ]
    )
    old_nodes = b"".join(encode_bytes_field(1, node) for node in original_nodes)
    if not old_nodes:
        raise ValueError("gpt_step_v2 has no nodes")
    # Repeated field order is preserved for existing nodes; place the bridge
    # splits first so all downstream consumers see their producers first.
    graph = split_nodes + old_nodes + value_infos + packed_graph_inputs + split_sizes + b"".join(
        other_graph_fields
    )

    graph_replacement_done = False
    output_model = bytearray()
    for number, wire, value, raw in v2_fields:
        if number == 7 and wire == 2:
            if graph_replacement_done:
                raise ValueError("unexpected duplicate graph field")
            output_model.extend(encode_bytes_field(7, graph))
            graph_replacement_done = True
        else:
            output_model.extend(raw)
    if not graph_replacement_done:
        raise ValueError("missing ONNX graph field")

    output_path.write_bytes(output_model)
    manifest = {
        "graph_version": "2.1-init-bridge",
        "source_step_sha256": hashlib.sha256(step_v2_path.read_bytes()).hexdigest(),
        "source_packed_step_sha256": hashlib.sha256(base_path.read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
        "input_cache_layout": "ABI-1 packed K/V",
        "output_cache_layout": "v2 per-layer K/V",
        "bridge_nodes": ["Split(k_cache, axis=0)", "Split(v_cache, axis=0)"],
        "layers": LAYERS,
    }
    output_path.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), **manifest}, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: prepare-gpt-step-init-v2.py MODEL_PACK_DIRECTORY")
    main(Path(sys.argv[1]))
