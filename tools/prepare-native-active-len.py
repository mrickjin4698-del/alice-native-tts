#!/usr/bin/env python3
"""Prepare an isolated GPT-step graph with an explicit host-shape candidate input.

Requires active_len == x_len[0] + y_len[0] + idx[0] + 1 on every invocation.
Does not establish TensorRT classification, capture compatibility, or performance.
Uses only standard-library protobuf helpers; never overwrites existing outputs.
"""
from __future__ import annotations

import hashlib
import json
import runpy
import struct
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HELPERS = runpy.run_path(str(Path(__file__).with_name("prepare-gpt-step-init-v2.py")), run_name="protobuf_helpers")
parse = HELPERS["parse_fields"]
text = HELPERS["field_text"]
varint = HELPERS["read_varint"]
ef = HELPERS["encode_bytes_field"]
ei = HELPERS["encode_int_field"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def values(blob, field):
    return [v for n, w, v, raw in parse(blob) if n == field]


def integers(blob, field):
    result = []
    for n, w, v, raw in parse(blob):
        if n != field:
            continue
        pos = 0
        while pos < len(v):
            value, pos = varint(v, pos)
            result.append(value if value < 2**63 else value - 2**64)
    return result


def main(source, destination):
    metadata_path = destination.with_suffix(".json")
    require(source.resolve() != destination.resolve(), "source and destination must differ")
    require(not destination.exists() and not metadata_path.exists(), "output or metadata already exists")
    original = source.read_bytes()
    model_fields = parse(original)
    graphs = values(original, 7)
    require(len(graphs) == 1, "expected one graph")
    graph = graphs[0]
    graph_fields = parse(graph)
    nodes = values(graph, 1)
    by_name = {text(node, 3): node for node in nodes}
    require(len(by_name) == len(nodes), "duplicate node names")
    producers = {}
    for node in nodes:
        for output in values(node, 2):
            name = output.decode()
            require(name not in producers, "duplicate output " + name)
            producers[name] = node
    input_names = [text(v, 1) for v in values(graph, 11)]
    require("active_len" not in input_names and "active_len" not in producers, "active_len already exists")

    def inputs(node):
        return [v.decode() for v in values(node, 1)]

    def attribute(node, name, default=None):
        matches = [v for v in values(node, 5) if text(v, 1) == name]
        require(len(matches) <= 1, "duplicate attribute")
        return integers(matches[0], 3)[0] if matches else default

    def constant(name):
        node = producers[name]
        require(text(node, 4) == "Constant", "expected Constant: " + name)
        attrs = values(node, 5)
        require(len(attrs) == 1 and text(attrs[0], 1) == "value", "unexpected Constant encoding")
        tensors = values(attrs[0], 5)
        require(len(tensors) == 1, "expected tensor constant")
        tensor = tensors[0]
        require(integers(tensor, 2) == [7], "expected INT64 constant")
        dims = tuple(integers(tensor, 1))
        raw = values(tensor, 9)
        data = tuple(struct.unpack("<" + "q" * (len(raw[0]) // 8), raw[0])) if raw else tuple(integers(tensor, 7))
        require(len(data) == 1, "expected singleton constant: " + name)
        require(dims in [(), (1,)], "unexpected constant dimensions")
        return data[0], dims

    # Return affine coefficients (x_len[0], y_len[0], idx[0], constant), tensor shape.
    def expression(name):
        node = producers[name]
        op = text(node, 4)
        ins = inputs(node)
        if op == "Constant":
            value, shape = constant(name)
            return (0, 0, 0, value), shape
        if op == "Gather":
            require(len(ins) == 2 and attribute(node, "axis", 0) == 0, "unexpected Gather")
            require(constant(ins[1]) == (0, ()), "expected scalar zero Gather index")
            if ins[0] in ["x_len", "y_len", "idx"]:
                coefficients = [0, 0, 0, 0]
                coefficients[["x_len", "y_len", "idx"].index(ins[0])] = 1
                return tuple(coefficients), ()
            value, shape = expression(ins[0])
            require(shape == (1,), "Gather source must contain exactly one item")
            return value, ()
        if op == "Add":
            require(len(ins) == 2, "unexpected Add")
            a, sa = expression(ins[0])
            b, sb = expression(ins[1])
            require(sa == sb == (), "only scalar Add supported")
            return tuple(x + y for x, y in zip(a, b)), ()
        if op == "Reshape":
            require(len(ins) == 2 and constant(ins[1]) == (-1, (1,)), "expected singleton flatten")
            value, shape = expression(ins[0])
            require(shape in [(), (1,)], "unexpected Reshape data")
            return value, (1,)
        if op == "Unsqueeze":
            require(len(ins) == 2 and constant(ins[1]) == (0, (1,)), "expected Unsqueeze axis zero")
            value, shape = expression(ins[0])
            require(shape == (), "expected scalar Unsqueeze data")
            return value, (1,)
        raise ValueError("unproved expression operator: " + op)

    for name in ["x_len", "y_len", "idx"]:
        info = next(v for v in values(graph, 11) if text(v, 1) == name)
        tensor_type = values(values(info, 2)[0], 1)[0]
        require(integers(tensor_type, 1) == [7], "length input must be INT64")
        dims = values(values(tensor_type, 2)[0], 1)
        require(len(dims) == 1, "length input must be rank one")

    changes = []
    replacements = {}
    for layer in range(24):
        for kind, offset in [("K", 0), ("V", 1)]:
            name = "/Slice_" + str(3 + layer * 6 + offset)
            node = by_name[name]
            ins = inputs(node)
            require(text(node, 4) == "Slice" and len(ins) == 4, "unexpected Slice " + name)
            scatter_number = layer * 2 + offset
            scatter_name = "/ScatterElements" + ("_" + str(scatter_number) if scatter_number else "")
            require(text(producers[ins[0]], 3) == scatter_name, "unexpected Slice source")
            require(text(producers[ins[0]], 4) == "ScatterElements", "expected ScatterElements")
            require(constant(ins[1]) == (0, (1,)), "Slice must start at zero")
            require(constant(ins[3]) == (1, (1,)), "Slice must target cache axis one")
            coefficients, shape = expression(ins[2])
            require(coefficients == (1, 1, 1, 1) and shape == (1,), "Slice end is not proven T+1")
            parts = []
            index = 0
            for n, w, v, raw in parse(node):
                if n == 1:
                    parts.append(ef(1, b"active_len") if index == 2 else raw)
                    index += 1
                else:
                    parts.append(raw)
            replacement = b"".join(parts)
            require(inputs(replacement) == ins[:2] + ["active_len"] + ins[3:], "unexpected input replacement")
            require([r for n,w,v,r in parse(node) if n != 1] ==
                    [r for n,w,v,r in parse(replacement) if n != 1], "non-input node fields changed")
            replacements[name] = replacement
            changes.append({"node": name, "layer": layer, "cache": kind, "input_index": 2,
                            "old_input": ins[2], "new_input": "active_len",
                            "old_end_coefficients": list(coefficients), "old_end_shape": list(shape)})
    require(len(replacements) == 48, "expected exactly 48 changes")
    # ValueInfo: active_len, tensor(INT64), fixed shape [1].
    shape = ef(1, ei(1, 1))
    tensor_type = ei(1, 7) + ef(2, shape)
    info = ef(1, b"active_len") + ef(2, ef(1, tensor_type))
    new_graph = b"".join(ef(1, replacements[text(v, 3)]) if n == 1 and text(v, 3) in replacements else raw
                         for n, w, v, raw in graph_fields) + ef(11, info)
    derived = b"".join(ef(7, new_graph) if n == 7 else raw for n, w, v, raw in model_fields)

    # Reparse the serialized result and verify preservation, before creating any files.
    verified_graph = values(derived, 7)[0]
    verified_nodes = values(verified_graph, 1)
    require(len(verified_nodes) == len(nodes), "node count changed")
    changed_names = []
    for old, new in zip(nodes, verified_nodes):
        if old != new:
            require(text(old, 3) in replacements and new == replacements[text(old, 3)], "unexpected node change")
            changed_names.append(text(old, 3))
    require(set(changed_names) == set(replacements), "changed node set mismatch")
    require(values(graph, 5) == values(verified_graph, 5), "initializer bytes changed")
    require([r for n,w,v,r in parse(graph) if n != 1] + [ef(11, info)] ==
            [r for n,w,v,r in parse(verified_graph) if n != 1], "non-node graph fields changed")
    require([r for n,w,v,r in model_fields if n != 7] ==
            [r for n,w,v,r in parse(derived) if n != 7], "non-graph model fields changed")
    metadata = {
        "source": str(source), "output": str(destination),
        "source_sha256": hashlib.sha256(original).hexdigest(),
        "output_sha256": hashlib.sha256(derived).hexdigest(),
        "new_input": {"name": "active_len", "dtype": "INT64", "shape": [1]},
        "required_invariant": "active_len[0] == x_len[0] + y_len[0] + idx[0] + 1",
        "proof_scope": "Structural ONNX equivalence for matching active_len, valid original inputs, and no INT64 overflow; not runtime or numerical validation.",
        "valid_cache_position": "For the existing 512-slot cache, 0 <= T <= 511 and active_len == T+1.",
        "changed_nodes": changes,
        "validation": {"changed_node_count": 48, "unchanged_node_count": len(nodes)-48,
                       "all_untouched_nodes_byte_identical": True,
                       "all_initializers_byte_identical": True,
                       "other_graph_and_model_fields_byte_identical": True},
        "tensorrt_status": "Not built or tested; host shape classification and graph capture remain unproven."
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents overwriting even if another process races us.
    with destination.open("xb") as stream:
        stream.write(derived)
    with metadata_path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(destination), "metadata": str(metadata_path),
                      "source_sha256": metadata["source_sha256"], "output_sha256": metadata["output_sha256"],
                      "validation": metadata["validation"]}, indent=2))


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    if len(sys.argv) not in [1, 3]:
        raise SystemExit("usage: prepare-native-active-len.py [SOURCE_ONNX OUTPUT_ONNX]")
    main(Path(sys.argv[1]) if len(sys.argv) == 3 else root / "target/r1-pack/gpt_step.onnx",
         Path(sys.argv[2]) if len(sys.argv) == 3 else root / "target/r2.5-active-len/gpt_step.onnx")
