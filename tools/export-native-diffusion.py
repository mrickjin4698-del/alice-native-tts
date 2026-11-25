"""Export an ABI-2 native GPT-SoVITS v3, v4 or V5 diffusion pack."""

import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import shutil
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parent
FAMILIES = {
    "v3": (24000, 32, 0.0, 468, 934),
    "v4": (48000, 32, 0.0, 500, 1000),
    "v5turbo": (48000, 4, 0.0, 500, 1000),
    "v5dev": (48000, 32, 1.30, 500, 1000),
}
SHARED_ASSETS = ("bert.onnx", "g2pw.onnx", "g2p-en.json", "cmudict.json",
                 "config.json", "gpt_encoder.onnx", "gpt_step.onnx")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def verify_gpt_assets(torch, helper, path, gpt):
    """Check every learned initializer in the ABI-1 export against the checkpoint."""
    import hashlib
    import numpy as np
    import onnx
    from onnx import numpy_helper

    def key(value):
        return tuple(value.shape), hashlib.sha256(np.ascontiguousarray(value)).digest()

    weights = torch.load(gpt, map_location="cpu", weights_only=True)["weight"]
    allowed = set()
    for value in weights.values():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            array = value.detach().float().numpy()
            allowed.add(key(array))
            if array.ndim == 2:
                allowed.add(key(array.T))
    for name in ("gpt_encoder.onnx", "gpt_step.onnx"):
        graph = onnx.load(str(path / name)).graph
        weights = [value for value in graph.initializer
                   if value.name.startswith(("t2s_model.", "onnx::MatMul_"))]
        if not weights or any(key(numpy_helper.to_array(value)) not in allowed for value in weights):
            raise ValueError(f"{name} does not match the checkpoint's ABI-1 GPT parameters")
    return helper["checkpoint_digest"](gpt)


def checkpoint_spec(manifest, runtime, precision):
    spec = read(manifest)
    if spec.get("schema_version") != 1 or spec.get("family") not in FAMILIES:
        raise ValueError("expected a v3, v4, v5turbo or v5dev checkpoint manifest")
    models = runtime / "GPT_SoVITS/pretrained_models"
    for key in ("gpt", "sovits", "reference_audio"):
        spec[key] = str((manifest.parent / spec[key]).resolve())
    for key, default in (("bert", "chinese-roberta-wwm-ext-large"), ("hubert", "chinese-hubert-base")):
        path = manifest.parent / spec[key] if key in spec else models / default
        spec[key] = str(path.resolve())
    spec["runtime_root"] = str(runtime)
    spec["device"] = "cuda" if precision == "fp16" else "cpu"
    spec.setdefault("default_language", "auto")
    return spec


def reference_data(torch, model, spec, reference_limit):
    source = type(model).using_vocoder_synthesis.__globals__
    text = spec["reference_text"].strip()
    language = spec["reference_language"]
    if text[-1] not in source["splits"]:
        text += "." if language == "en" else "。"
    phones, bert, _ = model.text_preprocessor.segment_and_extract_feature_for_text(
        text, language, spec["family"],
    )
    cache = model.prompt_cache
    prompt = cache["prompt_semantic"].reshape(1, 1, -1)
    phone_tensor = torch.tensor([phones], dtype=torch.long, device=prompt.device)
    spec_value = cache["refer_spec"][0]
    if isinstance(spec_value, tuple):
        spec_value = spec_value[0]
    features, speaker = model.vits_model.decode_encp(prompt, phone_tensor, spec_value.to(model.precision))
    audio = cache["raw_audio"].to(device=prompt.device, dtype=torch.float32)
    if audio.shape[0] == 2:
        audio = audio.mean(0, keepdim=True)
    mel_rate = 24000 if spec["family"] == "v3" else 32000
    if cache["raw_sr"] != mel_rate:
        audio = source["resample"](audio, cache["raw_sr"], mel_rate, model.configs.device)
    mel_fn = source["mel_fn"] if spec["family"] == "v3" else source["mel_fn_v4"]
    mel = source["norm_spec"](mel_fn(audio)).to(model.precision)
    frames = min(mel.shape[-1], features.shape[-1])
    if not frames or bert.shape != (1024, len(phones)):
        raise ValueError("unexpected reference feature shapes")
    return phones, bert, prompt, features[..., :frames][..., -reference_limit:], \
        mel[..., :frames][..., -reference_limit:], speaker


def make_graphs(torch, model, family, steps, guidance, *, step_variant="serial"):
    from torch import nn
    from torch.nn import functional as F

    is_v5 = family.startswith("v5")
    rate_factor = 1.875 if family == "v3" else 2.0
    dtype = model.precision

    class TargetFeatures(nn.Module):
        def __init__(self):
            super().__init__()
            self.vits = model.vits_model

        def forward(self, codes, phones, speaker):
            m = self.vits
            speaker = speaker.to(dtype)
            code_frames = torch._shape_as_tensor(codes)[2:3].to(codes.device)
            text_frames = torch._shape_as_tensor(phones)[1:2].to(phones.device)
            quantized = F.interpolate(m.quantizer.decode(codes), scale_factor=2.0, mode="nearest")
            hidden = m.enc_p(quantized, code_frames * 2, phones, text_frames, speaker, 1)[0]
            features = F.interpolate(m.bridge(hidden), scale_factor=rate_factor, mode="nearest")
            # Upstream uses 3.875 * tokens for the v3 mask and 4 * tokens otherwise.
            frames = torch.div(code_frames * 31, 8, rounding_mode="floor") if family == "v3" else code_frames * 4
            return m.wns1(features, frames, speaker)[0].float()

    class Condition(nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = model.vits_model.cfm.estimator

        def forward(self, features, prompt):
            prompt = prompt.transpose(1, 2).to(dtype)
            text = features.transpose(1, 2).to(dtype)
            text_embedding = self.dit.text_embed(text, text.shape[1], drop_text=False)
            pos = torch.cat((prompt, text_embedding), dim=-1)
            if not is_v5:
                return pos.float()
            projection = self.dit.input_embed.proj
            weight = projection.weight[:, prompt.shape[-1]:]
            pos = F.linear(pos, weight, projection.bias).float()
            if not guidance:
                return pos
            neg = torch.cat((torch.zeros_like(prompt), text_embedding), dim=-1)
            return pos, F.linear(neg, weight, projection.bias).float()

    class Step(nn.Module):
        def __init__(self):
            super().__init__()
            self.dit = model.vits_model.cfm.estimator

        def predict(self, x, time, mask, rope, condition):
            m = self.dit
            if is_v5:
                weight = m.input_embed.proj.weight[:, :x.shape[-1]]
                hidden = F.linear(x, weight) + condition.to(dtype)
            else:
                hidden = m.input_embed.proj(torch.cat((x, condition.to(dtype)), dim=-1))
            hidden = m.input_embed.conv_pos_embed(hidden, mask=mask if is_v5 else None) + hidden
            residual = hidden
            for block in m.transformer_blocks:
                hidden = block(hidden, time, mask=mask, rope=rope)
            if m.long_skip_connection is not None:
                hidden = m.long_skip_connection(torch.cat((hidden, residual), dim=-1))
            return m.proj_out(m.norm_out(hidden, time)).transpose(1, 2)

        def forward(self, state, time, generation_mask, freqs, condition, negative=None):
            x = state.to(dtype)
            t = time.to(dtype)
            time_embedding = self.dit.time_embed(t)
            if not is_v5:
                time_embedding = time_embedding + self.dit.d_embed(torch.ones_like(t) * (1.0 / steps))
            mask = torch.ones_like(generation_mask[:, 0], dtype=torch.bool) if step_variant == "serial" else None
            rope = (freqs.to(dtype), 1.0)
            x_frames = x.transpose(1, 2)
            if step_variant == "batched":
                paired = self.predict(
                    torch.cat((x_frames, x_frames), dim=0),
                    torch.cat((time_embedding, time_embedding), dim=0),
                    None, rope, torch.cat((condition, negative), dim=0),
                )
                v, negative_v = paired[:1], paired[1:]
            else:
                v = self.predict(x_frames, time_embedding, mask, rope, condition)
                if guidance:
                    negative_v = self.predict(x_frames, time_embedding, mask, rope, negative)
            if guidance:
                v = v + (v - negative_v) * guidance
            updated = x + (1.0 / steps) * v
            return torch.where(generation_mask > 0, updated, torch.zeros_like(updated)).float()

    class Waveform(nn.Module):
        def __init__(self):
            super().__init__()
            self.vocoder = model.vocoder

        def forward(self, mel):
            mel = (mel.to(dtype) + 1) / 2 * 14 - 12
            return self.vocoder(mel).float()

    return TargetFeatures(), Condition(), Step(), Waveform()


def export_graph(torch, output, module, values, inputs, outputs, dynamic, *, fold_constants=True):
    print(f"Exporting {output.name}", flush=True)
    torch.onnx.export(module.eval(), values, str(output), input_names=inputs,
                      output_names=outputs, dynamic_axes=dynamic, opset_version=20,
                      dynamo=False, do_constant_folding=fold_constants)


def write_tensor(tensor, path):
    import numpy as np

    values = tensor.detach().float().cpu().contiguous().numpy().astype("<f4", copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite reference data: {path.name}")
    values.tofile(path)


def compare(torch, name, actual, expected, precision):
    actual, expected = actual.float(), expected.float()
    if actual.shape != expected.shape or not torch.isfinite(actual).all():
        raise ValueError(f"{name} shape or finite-value check failed")
    delta = actual - expected
    stats = {
        "relative_l2": (torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(expected).clamp_min(1e-8)).item(),
        "max_abs": delta.abs().max().item(),
    }
    limit = 5e-3 if precision == "fp16" else 2e-4
    if not stats["relative_l2"] <= limit:
        raise ValueError(f"{name} differs from upstream: {stats}")
    print(f"Validated {name} against upstream: {stats}", flush=True)
    return stats


def export(model, spec, args, helper, gpt_hash):
    import torch

    family = spec["family"]
    sample_rate, steps, guidance, reference_limit, total_frames = FAMILIES[family]
    if int(model.vocoder_configs["sr"]) != sample_rate:
        raise ValueError("the checkpoint uses an unexpected output sample rate")

    args.output.mkdir(parents=True, exist_ok=False)
    for name in SHARED_ASSETS:
        shutil.copy2(args.gpt_pack / name, args.output / name)
    active_len = runpy.run_path(str(TOOLS / "prepare-native-active-len.py"))
    active_len["main"](args.output / "gpt_step.onnx", args.output / "gpt_step_active_len.onnx")

    with torch.inference_mode():
        phones, bert, prompt, reference, mel, speaker = reference_data(torch, model, spec, reference_limit)
        if speaker.ndim != 3 or speaker.shape[0] != 1 or speaker.shape[2] != 1 or mel.shape[1] != 100:
            raise ValueError("unexpected speaker or mel dimensions")
        target, condition_graph, step, wave = make_graphs(torch, model, family, steps, guidance)
        # Synthetic target lengths exercise the same decoder with the real speaker condition.
        device = speaker.device
        test_codes = torch.arange(45, device=device, dtype=torch.long).reshape(1, 1, 45)
        test_phones = torch.tensor([phones], device=device, dtype=torch.long)
        actual = target(test_codes, test_phones, speaker.float())
        expected, _ = model.vits_model.decode_encp(test_codes, test_phones, None, speaker)
        validation = {"target_features": compare(torch, "target features", actual, expected, args.precision)}

        export_graph(torch, args.output / "target_features.onnx", target,
                     (test_codes, test_phones, speaker.float()),
                     ["codes", "phones", "speaker"], ["features"],
                     {"codes": {2: "tokens"}, "phones": {1: "phones"}, "features": {2: "frames"}})
        value = torch.cat((reference, actual.to(reference.dtype)), dim=-1).float()
        frames = value.shape[-1]
        padded = torch.nn.functional.pad(mel.float(), (0, frames - mel.shape[-1]))
        condition = condition_graph(value, padded)
        conditions = condition if isinstance(condition, tuple) else (condition,)
        condition_outputs = ["condition", "negative"] if guidance else ["condition"]
        export_graph(torch, args.output / "dit_condition.onnx", condition_graph, (value, padded),
                     ["features", "prompt"], condition_outputs,
                     {"features": {2: "frames"}, "prompt": {2: "frames"},
                      **{name: {1: "frames"} for name in condition_outputs}})
        frequencies, scale = model.vits_model.cfm.estimator.rotary_embed.forward_from_seq_len(total_frames)
        if not isinstance(scale, (int, float)) or scale != 1 or frequencies.shape[:2] != (1, total_frames):
            raise ValueError("unsupported rotary position table")
        rope = frequencies[:, :frames].float()
        position_mask = torch.ones((1, 1, frames), device=device)
        position_mask[..., :mel.shape[-1]] = 0
        state = torch.zeros_like(padded)
        time = torch.tensor([0.125], device=device)

        # Compare a solver step against the upstream estimator on the same
        # tensors. This also checks the original serial CFG formula.
        estimator = model.vits_model.cfm.estimator
        lens = torch.tensor([frames], dtype=torch.long, device=device)
        dt = torch.ones((1,), device=device, dtype=model.precision) / steps
        kwargs = {"infer": True}
        if family.startswith("v5"):
            kwargs["static_cache"] = estimator.prepare_static_cache(padded.to(model.precision), lens,
                                                                     value.to(model.precision))
        else:
            kwargs["dt_base_bootstrap"] = dt
        def velocity(drop_audio_cond):
            return estimator(state.to(model.precision), padded.to(model.precision), lens, time.to(model.precision),
                             text0=value.to(model.precision), drop_audio_cond=drop_audio_cond, **kwargs)[0].transpose(1, 2)
        ref_v = velocity(False)
        if guidance:
            ref_v = ref_v + (ref_v - velocity(True)) * guidance
        ref_state = state.to(model.precision) + (1.0 / steps) * ref_v
        ref_state[..., :mel.shape[-1]] = 0
        result = step(state, time, position_mask, rope, *conditions)
        validation["dit_step"] = compare(torch, "denoising step", result, ref_state, args.precision)

        with ExitStack() as cleanup:
            for norm in step.modules():
                if isinstance(norm, torch.nn.LayerNorm):
                    cleanup.enter_context(patch.object(
                        norm, "forward", lambda x, norm=norm: helper["fp32_layer_norm"](torch, x, norm),
                    ))
            export_graph(torch, args.output / "dit_step.onnx", step,
                         (state, time, position_mask, rope, *conditions),
                         ["state", "time", "generation_mask", "rope", *condition_outputs], ["state_next"],
                         {"state": {2: "frames"}, "generation_mask": {2: "frames"}, "rope": {1: "frames"},
                          "state_next": {2: "frames"}, **{name: {1: "frames"} for name in condition_outputs}},
                         # The legacy Torch exporter folds v3/v4 attention constants
                         # across CPU and CUDA. Leave their folding to ONNX Runtime.
                         fold_constants=family.startswith("v5"))
        with ExitStack() as cleanup:
            if type(model.vocoder).__name__ == "BigVGAN":
                helper["patch_bigvgan_resampling"](cleanup, model.vocoder)
            export_graph(torch, args.output / "vocoder.onnx", wave, (mel.float(),),
                         ["mel"], ["waveform"], {"mel": {2: "frames"}, "waveform": {2: "samples"}})

        for name, value in (("reference_bert", bert), ("reference_features", reference),
                            ("reference_mel", mel), ("reference_speaker", speaker), ("rope", frequencies)):
            write_tensor(value, args.output / (name + ".f32"))

    pack = {
        "abi": 2, "family": family, "sample_rate": sample_rate, "precision": args.precision,
        "gpt_checkpoint_sha256": gpt_hash,
        "sovits_checkpoint_sha256": helper["checkpoint_digest"](spec["sovits"]),
        "reference_audio_sha256": helper["checkpoint_digest"](spec["reference_audio"]),
        "reference_phones": list(phones), "reference_codes": prompt.flatten().cpu().tolist(),
        "reference_frames": reference.shape[-1], "feature_channels": reference.shape[1],
        "speaker_channels": speaker.shape[1], "rope_channels": frequencies.shape[-1],
        "export_validation": validation,
    }
    (args.output / "pack.json").write_text(json.dumps(pack, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {family} ABI-2 pack to {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--gpt-pack", required=True, type=Path, help="ABI-1 frontend and GPT graphs from the same checkpoint")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    args = parser.parse_args()
    for name in ("checkpoint", "runtime", "gpt_pack", "output"):
        setattr(args, name, getattr(args, name).resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    spec = checkpoint_spec(args.checkpoint, args.runtime, args.precision)

    import torch
    torch.set_num_threads(4)
    helper = runpy.run_path(str(TOOLS / "gpt-sovits-tensorrt.py"))
    gpt_hash = verify_gpt_assets(torch, helper, args.gpt_pack, spec["gpt"])
    # Use the same checkpoint validation and device/precision settings as the worker.
    worker = runpy.run_path(str(TOOLS / "gpt-sovits-worker.py"))
    model, _, _, _ = worker["load"](spec)
    export(model, spec, args, helper, gpt_hash)


if __name__ == "__main__":
    main()
