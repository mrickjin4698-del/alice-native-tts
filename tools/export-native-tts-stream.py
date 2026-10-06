"""Offline v2Pro streaming graph export. Never imported by the Rust runtime.

Uses the same enc_p latent overlap and zero-noise decoding as GPT-SoVITS'
decode_streaming; replaces in-place latent writes with dynamic concatenation.
The minimal-inference source tree used for the ABI-1 export is required.
"""
import argparse
import json
import math
import os
import sys
import faulthandler
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")


def main():
    faulthandler.dump_traceback_later(45, repeat=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Refusing to overwrite an existing streaming graph")
    sys.path[:0] = [str(args.source), str(args.source / "GPT_SoVITS")]
    print("Loading offline export dependencies", flush=True)
    import numpy as np
    import onnxruntime as ort
    import torch
    from torch import nn
    from torch.nn import functional as F
    from GPT_SoVITS.process_ckpt import load_sovits_new, get_sovits_version_from_path_fast
    from GPT_SoVITS.module.models import SynthesizerTrn

    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    print("Loading v2Pro weights", flush=True)
    saved = load_sovits_new(str(args.weights))
    def as_dict(value):
        if isinstance(value, dict): return {k:as_dict(v) for k,v in value.items()}
        if isinstance(value, (list,tuple)): return [as_dict(v) for v in value]
        if hasattr(value, "__dict__"): return {k:as_dict(v) for k,v in vars(value).items()}
        return value
    config = as_dict(saved["config"])
    _, version, _ = get_sovits_version_from_path_fast(str(args.weights))
    if version != "v2Pro" or config["data"]["sampling_rate"] != 32000:
        raise SystemExit("Requires the ABI-1 v2Pro 32 kHz weights")
    config["model"].update(version=version, semantic_frame_rate="25hz")
    model = SynthesizerTrn(config["data"]["filter_length"] // 2 + 1,
                           config["train"]["segment_size"] // config["data"]["hop_length"],
                           n_speakers=config["data"]["n_speakers"], **config["model"])
    model.load_state_dict(saved["weight"], strict=False)
    model.eval()
    for module in model.modules():
        if "EuclideanCodebook" in type(module).__name__:
            import types
            module.init_embed_ = types.MethodType(lambda self, data: None, module)
    channels = config["model"]["hidden_channels"]
    samples_per_token = math.prod(config["model"]["upsample_rates"]) * 2
    if channels != 192 or samples_per_token != 1280:
        raise SystemExit("Unsupported streaming latent/audio dimensions")

    class StreamingVits(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model

        def forward(self, pred_semantic, text_seq, refer_spec, sv_emb,
                    result_length, overlap_latent):
            m = self.model
            ref_mask = torch.ones_like(refer_spec[:, :1, :])
            ge = m.ref_enc(refer_spec[:, :704, :] * ref_mask, ref_mask)
            ge = m.prelu(ge + m.sv_emb(sv_emb).unsqueeze(-1))
            quantized = F.interpolate(m.quantizer.decode(pred_semantic), scale_factor=2.0, mode="nearest")
            enc = m.enc_p
            mask = torch.ones_like(quantized[:, :1, :])
            y = enc.ssl_proj(quantized * mask) * mask
            y = enc.encoder_ssl(y * mask, mask)
            text = enc.text_embedding(text_seq).transpose(1, 2)
            text_mask = torch.ones_like(text[:, :1, :])
            text = enc.encoder_text(text * text_mask, text_mask)
            y = enc.mrte(y, mask, text, text_mask, m.ge_to512(ge.transpose(2, 1)).transpose(2, 1))
            y = enc.encoder2(y * mask, mask)
            y = y[:, :, -(result_length[0] * 2):]
            overlap = overlap_latent.shape[-1]
            # Empty overlap is a valid first chunk. Clamp only the denominator
            # to avoid division by zero; both blend slices remain empty.
            denominator = torch.clamp(torch.tensor(overlap, dtype=y.dtype), min=1) * 2
            phase = torch.arange(overlap, device=y.device, dtype=y.dtype) * math.pi / denominator
            blend = torch.sin(phase)[None, None, :] * y[:, :, :overlap] + torch.cos(phase)[None, None, :] * overlap_latent
            latent = torch.cat((blend, y[:, :, overlap:]), dim=-1)
            latent_mask = torch.ones_like(latent[:, :1, :])
            mean, _ = torch.split(enc.proj(latent) * latent_mask, m.inter_channels, dim=1)
            z = m.flow(mean, latent_mask, g=ge, reverse=True)
            audio = m.dec(z * latent_mask, g=ge)
            return audio, latent

    wrapper = StreamingVits().eval()
    generator = torch.Generator().manual_seed(20261003)
    codes = torch.randint(0, 1024, (1, 1, 48), generator=generator)
    phones = torch.randint(0, 100, (1, 32), generator=generator)
    ref = torch.randn(1, 1025, 163, generator=generator)
    sv = torch.randn(1, 20480, generator=generator)
    overlap = torch.zeros(1, channels, 4)
    names = ["pred_semantic", "text_seq", "refer_spec", "sv_emb", "result_length", "overlap_latent"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    print("Exporting dynamic streaming graph", flush=True)
    torch.onnx.export(wrapper, (codes, phones, ref, sv, torch.tensor([26]), overlap), str(args.output),
        input_names=names, output_names=["audio", "latent"], opset_version=20, dynamo=False,
        dynamic_axes={"pred_semantic": {2: "sem_len"}, "text_seq": {1: "text_len"},
                      "refer_spec": {2: "ref_len"}, "overlap_latent": {2: "overlap_frames"},
                      "audio": {2: "audio_len"}, "latent": {2: "latent_frames"}})
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(args.output), options, providers=["CPUExecutionProvider"])
    checks = []
    for total, result, overlap_tokens, text_length, ref_length in [(24,24,0,32,163), (48,26,2,32,163), (53,12,7,21,127), (10,10,0,9,88)]:
        c = torch.randint(0, 1024, (1, 1, total), generator=generator)
        p = phones[:, :text_length]
        r = ref[:, :, :ref_length]
        prev = torch.randn(1, channels, overlap_tokens * 2, generator=generator)
        inputs = (c,p,r,sv,torch.tensor([result]),prev)
        expected = wrapper(*inputs)
        # Independent original decode_streaming parity, not only self-parity.
        original = model.decode_streaming(c,p,[r],sv_emb=[sv],noise_scale=0.0,
            result_length=result, overlap_frames=prev if overlap_tokens else None)
        actual = session.run(None, {name: value.numpy() for name,value in zip(names,inputs)})
        diffs = []
        for label, index in [("audio",0),("latent",1)]:
            error = np.linalg.norm(actual[index] - original[index].numpy()) / max(np.linalg.norm(original[index].numpy()),1e-8)
            self_error = np.linalg.norm(actual[index] - expected[index].numpy()) / max(np.linalg.norm(expected[index].numpy()),1e-8)
            if not np.isfinite(actual[index]).all() or error > 2e-4 or self_error > 2e-4:
                raise RuntimeError(f"{label} parity failed: {error}, {self_error}")
            diffs.append({"output":label,"original_relative_l2":float(error),"wrapper_relative_l2":float(self_error),"shape":list(actual[index].shape)})
        if actual[0].shape[-1] != result * samples_per_token:
            raise RuntimeError("Streaming graph retained a fixed traced result length")
        checks.append({"total_tokens":total,"result_tokens":result,"overlap_tokens":overlap_tokens,"outputs":diffs})
    metadata = {"abi":1,"graph":"sovits_stream.onnx","latent_channels":channels,"samples_per_token":samples_per_token}
    import hashlib
    report = {"streaming":metadata,"validation":checks,"graph_sha256":hashlib.sha256(args.output.read_bytes()).hexdigest()}
    args.output.with_suffix(".validation.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
