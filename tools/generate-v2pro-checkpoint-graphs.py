"""Rebuild the embedded v2Pro operator graphs without loading checkpoint weights."""
import argparse
import io
import json
from pathlib import Path
import runpy
import sys
from types import MethodType

PROFILES = Path(__file__).resolve().parent.parent / "src" / "checkpoint_native"


def write_graph(name, module, arguments, inputs, outputs, axes, prefix, destination, torch, onnx):
    buffer = io.BytesIO()
    # The legacy exporter keeps every learned tensor as a named graph input.
    torch.onnx.export(
        module.eval(), tuple(arguments), buffer, input_names=inputs,
        output_names=outputs, dynamic_axes=axes, export_params=False,
        keep_initializers_as_inputs=True, do_constant_folding=False,
        opset_version=20, dynamo=False,
    )
    model = onnx.load_model_from_string(buffer.getvalue())
    graph = model.graph
    if graph.initializer:
        raise ValueError(f"{name} already contains parameters")
    expected_inputs = set(inputs)
    if not expected_inputs <= {value.name for value in graph.input}:
        raise ValueError(f"{name} is missing expected runtime inputs")
    state = module.state_dict()
    original_inputs = list(graph.input)
    graph.ClearField("input")
    for value in original_inputs:
        if value.name in expected_inputs:
            graph.input.append(value)
            continue
        shape = [dim.dim_value for dim in value.type.tensor_type.shape.dim]
        if (not value.name.startswith(prefix) or value.name not in state
                or list(state[value.name].shape) != shape
                or value.type.tensor_type.elem_type != onnx.TensorProto.FLOAT):
            raise ValueError(f"{name} has an unexpected weight input {value.name}")
        tensor = graph.initializer.add(name=value.name, dims=shape, data_type=onnx.TensorProto.FLOAT)
        tensor.data_location = onnx.TensorProto.EXTERNAL
        tensor.external_data.add(key="location", value="checkpoint")
    if not graph.initializer:
        raise ValueError(f"{name} has no external weights")

    tables = 0
    for node in graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.type == onnx.AttributeProto.TENSOR and tuple(attr.t.dims) == (1, 4000, 512):
                # The packed GPT KV cache bounds every requested position.
                table = onnx.numpy_helper.to_array(attr.t)[:, :512, :].copy()
                attr.t.CopyFrom(onnx.numpy_helper.from_array(table))
                tables += 1
    if tables != int(name.startswith("gpt_")):
        raise ValueError(f"{name} has an unexpected positional table layout")
    model.doc_string = "GPT-SoVITS v2Pro inference architecture with externally supplied checkpoint tensors."
    onnx.save(model, destination / f"{name}.onnx")


def write_fp16_graphs(directory):
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16
    from onnxruntime.transformers.onnx_model import OnnxModel

    fp32_io = {
        "gpt_encoder": ["bert_feature", "topk_values"],
        "gpt_step": ["topk_values"],
        "sovits_stream": ["refer_spec", "sv_emb", "overlap_latent", "audio", "latent"],
    }
    for name, keep_io_types in fp32_io.items():
        model = onnx.load(directory / f"{name}.onnx", load_external_data=False)
        # The converter tries to read initializer data. Represent the external
        # parameters as typed inputs during graph conversion, then restore them.
        parameters = {tensor.name: tensor for tensor in model.graph.initializer}
        for tensor in parameters.values():
            if (tensor.data_location != onnx.TensorProto.EXTERNAL
                    or tensor.raw_data or tensor.float_data or tensor.int32_data):
                raise ValueError(f"{name} has an embedded parameter {tensor.name}")
            model.graph.input.append(
                onnx.helper.make_tensor_value_info(tensor.name, tensor.data_type, tensor.dims)
            )
        model.graph.ClearField("initializer")
        fp16 = convert_float_to_float16(model, keep_io_types=keep_io_types)
        OnnxModel(fp16).topological_sort(is_deterministic=True)
        onnx.checker.check_model(fp16)
        inputs = list(fp16.graph.input)
        fp16.graph.ClearField("input")
        for value in inputs:
            if value.name in parameters:
                tensor = fp16.graph.initializer.add()
                tensor.CopyFrom(parameters[value.name])
                tensor.data_type = value.type.tensor_type.elem_type
            else:
                fp16.graph.input.append(value)
        onnx.save(fp16, directory / f"{name}_fp16.onnx")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="GPT-SoVITS minimal-inference source")
    parser.add_argument("--output", type=Path, required=True, help="Operator graph directory")
    parser.add_argument("--fp16-only", action="store_true", help="Read FP32 graph templates from --output")
    args = parser.parse_args()
    if args.fp16_only:
        write_fp16_graphs(args.output)
        return
    if not args.source:
        parser.error("--source is required when generating FP32 graphs")
    source = args.source.resolve()
    sys.path[:0] = [str(source), str(source / "GPT_SoVITS")]

    import onnx
    import torch
    from AR.models.t2s_lightning_module import Text2SemanticLightningModule
    from export_onnx import GPTEncoder, GPTStep, VQEncoder
    from GPT_SoVITS.module.models import SynthesizerTrn

    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    args.output.mkdir(parents=True, exist_ok=True)
    gpt_profile = json.loads((PROFILES / "gpt-profile.json").read_text(encoding="utf-8"))
    gpt_profile["model"].update(dropout=0, random_bert=0)
    gpt = Text2SemanticLightningModule(gpt_profile, args.output, is_train=False).eval()
    encoder = GPTEncoder(gpt, max_len=512)
    phones = torch.randint(0, 100, (1, 25), dtype=torch.long)
    prompts = torch.randint(0, 1024, (1, 16), dtype=torch.long)
    bert = torch.randn(1, 1024, 25)
    _, _, keys, values, text_length, prompt_length = encoder(phones, prompts, bert)
    write_graph(
        "gpt_encoder", encoder, [phones, prompts, bert],
        ["phoneme_ids", "prompts", "bert_feature"],
        ["topk_values", "topk_indices", "k_cache", "v_cache", "x_len", "y_len"],
        {"phoneme_ids": {1: "text_len"}, "prompts": {1: "prompt_len"},
         "bert_feature": {2: "text_len"}, "k_cache": {1: "batch"},
         "v_cache": {1: "batch"}, "x_len": {0: "one"}, "y_len": {0: "one"}},
        "t2s_model.", args.output, torch, onnx,
    )
    write_graph(
        "gpt_step", GPTStep(gpt),
        [torch.ones(1, 1, dtype=torch.long), keys, values, text_length, prompt_length, torch.tensor([0])],
        ["samples", "k_cache", "v_cache", "x_len", "y_len", "idx"],
        ["topk_values", "topk_indices", "k_cache_new", "v_cache_new"],
        {"k_cache": {1: "batch"}, "v_cache": {1: "batch"}, "x_len": {0: "one"},
         "y_len": {0: "one"}, "idx": {0: "one"}},
        "t2s_model.", args.output, torch, onnx,
    )
    del encoder, gpt, keys, values
    voice = json.loads((PROFILES / "sovits-profile.json").read_text(encoding="utf-8"))
    vits = SynthesizerTrn(
        voice["data"]["filter_length"] // 2 + 1, 32, n_speakers=300,
        p_dropout=0.0, n_layers_q=3, use_spectral_norm=False, freeze_quantizer=True,
        **voice["model"],
    ).eval()
    for module in vits.modules():
        if "EuclideanCodebook" in type(module).__name__:
            module.init_embed_ = MethodType(lambda self, data: None, module)
    write_graph(
        "vq_encoder", VQEncoder(vits), [torch.randn(1, 768, 20)],
        ["ssl_content"], ["codes"], {"ssl_content": {2: "time"}, "codes": {2: "time"}},
        "vq_model.", args.output, torch, onnx,
    )
    streaming_export = Path(__file__).with_name("export-native-tts-stream.py")
    streaming_vits = runpy.run_path(str(streaming_export))["streaming_vits"](vits)
    write_graph(
        "sovits_stream", streaming_vits,
        [torch.randint(0, 1024, (1, 1, 48)), torch.randint(0, 100, (1, 32)),
         torch.randn(1, 1025, 163), torch.randn(1, 20480),
         torch.tensor([26]), torch.zeros(1, 192, 4)],
        ["pred_semantic", "text_seq", "refer_spec", "sv_emb", "result_length", "overlap_latent"],
        ["audio", "latent"],
        {"pred_semantic": {2: "sem_len"}, "text_seq": {1: "text_len"},
         "refer_spec": {2: "ref_len"}, "overlap_latent": {2: "overlap_frames"},
         "audio": {2: "audio_len"}, "latent": {2: "latent_frames"}},
        "model.", args.output, torch, onnx,
    )

    write_fp16_graphs(args.output)


if __name__ == "__main__":
    main()
