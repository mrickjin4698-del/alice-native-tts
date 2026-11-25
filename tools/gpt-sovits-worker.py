"""Local GPT-SoVITS checkpoint adapter embedded in the Alice TTS binary."""

import base64
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys
import traceback

# The upstream runtime writes diagnostic output while importing and inferring.
wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
sys.stdin.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def send(kind, **data):
    wire.write(json.dumps({"type": kind, **data}, ensure_ascii=False, allow_nan=False) + "\n")
    wire.flush()


def load(spec):
    root = Path(spec["runtime_root"])
    os.chdir(root)
    sys.path[:0] = [str(root), str(root / "GPT_SoVITS")]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    import torch
    from GPT_SoVITS.TTS_infer_pack.TTS import TTS, TTS_Config

    family = spec["family"]
    if family not in TTS_Config.default_configs:
        raise ValueError(f"this GPT-SoVITS installation does not support {family}")

    class SessionConfig(TTS_Config):
        def save_configs(self, configs_path=None):
            # The worker owns its configuration for the lifetime of this process.
            self.update_configs()

    use_tensorrt = spec["device"] == "tensor_rt"
    device = "cuda" if use_tensorrt else spec["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch runtime cannot use CUDA")

    model_config = {
        "device": device,
        "is_half": device == "cuda",
        "version": family,
        "t2s_weights_path": spec["gpt"],
        "vits_weights_path": spec["sovits"],
        "bert_base_path": spec["bert"],
        "cnhuhbert_base_path": spec["hubert"],
    }
    settings = SessionConfig({"custom": model_config})
    model = TTS(settings)
    if model.configs.version != family:
        raise ValueError(
            f"expected checkpoint family {family} but GPT-SoVITS loaded {model.configs.version}"
        )
    if spec["reference_language"] not in model.configs.languages:
        raise ValueError(f"unsupported reference language {spec['reference_language']}")
    if spec["default_language"] not in model.configs.languages:
        raise ValueError(f"unsupported text language {spec['default_language']}")

    model.set_ref_audio(spec["reference_audio"])
    stages = install_checkpoint_tensorrt(model, spec["gpt"], torch) if use_tensorrt else ()
    fragment = family in ("v3", "v4", "v5turbo", "v5dev")
    rate = int(model.vocoder_configs["sr"] if fragment else model.configs.sampling_rate)
    send("ready", sample_rate=rate, languages=model.configs.languages)
    return model, fragment, rate, stages


def synthesize(model, spec, fragment, sample_rate, stages, message):
    import numpy as np

    language = message.get("language") or spec["default_language"]
    if language not in model.configs.languages:
        raise ValueError(f"unsupported text language {language}")
    inputs = {
        "text": message["text"],
        "text_lang": language,
        "ref_audio_path": spec["reference_audio"],
        "prompt_text": spec["reference_text"],
        "prompt_lang": spec["reference_language"],
        "text_split_method": "cut5",
        "batch_size": 1,
        "split_bucket": False,
        "parallel_infer": False,
        "streaming_mode": not fragment,
        "return_fragment": fragment,
        "super_sampling": False,
        "seed": 1234,
    }
    if stages:
        inputs.update(use_cuda_graph=False, use_flash_attention=False, cache_dit="off")
    calls_before = [stage.calls for stage in stages]
    count = 0
    for rate, audio in model.run(inputs):
        audio = np.asarray(audio)
        if int(rate) != sample_rate or audio.ndim != 1 or audio.dtype.kind != "i" or audio.dtype.itemsize != 2:
            raise ValueError("GPT-SoVITS returned an unexpected PCM format")
        if audio.size:
            count += int(audio.size)
            send("audio", pcm=base64.b64encode(audio.astype("<i2", copy=False).tobytes()).decode("ascii"))
    if count == 0:
        raise ValueError("GPT-SoVITS returned no PCM samples")
    unused = [stage.label for stage, before in zip(stages, calls_before) if stage.calls == before]
    if unused:
        raise RuntimeError(f"GPT-SoVITS did not execute TensorRT {', '.join(unused)}")
    if stages and calls_before[0] == 0:
        counts = ", ".join(f"{stage.label}={stage.calls}" for stage in stages)
        print(f"[alice-tts] TensorRT calls: {counts}", file=sys.stderr, flush=True)
    send("done", samples=count)


def main():
    spec = json.loads(sys.stdin.readline())
    model, fragment, rate, stages = load(spec)
    for line in sys.stdin:
        try:
            with stages[0].backend.stream_context() if stages else nullcontext():
                synthesize(model, spec, fragment, rate, stages, json.loads(line))
        except Exception as error:
            traceback.print_exc()
            send("error", message=str(error))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        send("error", message=str(error))
        sys.exit(1)
