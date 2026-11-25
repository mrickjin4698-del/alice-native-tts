# Irodori-TTS adapter embedded in the Alice TTS binary.

import base64
import json
import os
import sys
import traceback


wire = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", newline="\n")
os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
sys.stdout = sys.stderr
sys.stdin.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def send(kind, **data):
    wire.write(json.dumps({"type": kind, **data}, ensure_ascii=False, allow_nan=False) + "\n")
    wire.flush()


def main():
    # The isolated interpreter runs with the selected checkout as its working directory.
    sys.path.insert(0, os.getcwd())
    import numpy as np
    import torch
    from irodori_tts import inference_runtime

    spec = json.loads(sys.stdin.readline())
    device = spec["device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but this PyTorch environment cannot use CUDA")
    runtime_options = {
        "checkpoint": spec["checkpoint"],
        "model_device": device,
        "codec_device": device,
    }
    if spec["codec_weights"] is not None:
        runtime_options["codec_repo"] = spec["codec_weights"]
    runtime = inference_runtime.InferenceRuntime.from_key(inference_runtime.RuntimeKey(**runtime_options))
    denoiser = None
    decoder = None
    inference_stream = None
    if spec["denoiser_plan"] is not None:
        denoiser = TensorRtDenoiser(runtime.model, spec["denoiser_plan"], spec["checkpoint"])
        runtime.model.encode_conditions = denoiser.encode_conditions
        runtime.model.forward_with_encoded_conditions = denoiser.forward
    if spec["decoder_plan"] is not None:
        decoder = TensorRtDecoder(runtime.codec, spec["decoder_plan"], spec["codec_weights"])
        runtime.codec.decode_latent = decoder
        inference_runtime.find_flattening_point = batched_tail_scan(
            torch, inference_runtime.find_flattening_point,
        )
        if denoiser is not None:
            from irodori_tts.rf import draw_initial_noise, temporal_score_rescale

            inference_runtime.sample_euler_rf_cfg = linear_rf_sampler(
                torch, inference_runtime.sample_euler_rf_cfg, draw_initial_noise, temporal_score_rescale,
            )
    if denoiser is not None or decoder is not None:
        inference_stream = torch.cuda.Stream()
        inference_stream.wait_stream(torch.cuda.current_stream())
    profile = os.environ.get("ALICE_IRODORI_PROFILE") == "1"
    reference = spec["reference_audio"]
    caption = spec["caption"]
    if reference is not None and not runtime.model_cfg.use_speaker_condition_resolved:
        raise ValueError("the Irodori-TTS checkpoint does not support reference audio")
    if caption is not None and not runtime.model_cfg.use_caption_condition:
        raise ValueError("the Irodori-TTS checkpoint does not support captions")

    sample_rate = int(runtime.codec.sample_rate)
    send("ready", sample_rate=sample_rate, languages=spec["languages"])
    for line in sys.stdin:
        try:
            message = json.loads(line)
            if denoiser is not None:
                denoiser.reset_request()
            with torch.cuda.stream(inference_stream):
                result = runtime.synthesize(
                    inference_runtime.SamplingRequest(
                        text=message["text"],
                        caption=caption,
                        ref_wav=reference,
                        no_ref=reference is None,
                        seed=spec["seed"],
                    )
                )
            if profile:
                timings = {name: round(seconds * 1000, 2) for name, seconds in result.stage_timings}
                if denoiser is not None:
                    timings.update(
                        tensorrt_steps=denoiser.executed_steps,
                        pytorch_steps=denoiser.eager_steps,
                        condition_reuses=denoiser.encode_conditions.hits,
                        caption_reuses=denoiser.caption_encoding.hits if denoiser.caption_encoding else 0,
                    )
                if decoder is not None:
                    timings["codec_backend"] = decoder.last_backend
                print("[alice-tts] Irodori " + json.dumps(timings), file=sys.stderr)
            audio = result.audio.detach().float().cpu().numpy()
            if audio.ndim == 2 and audio.shape[0] == 1:
                audio = audio[0]
            if int(result.sample_rate) != sample_rate or audio.ndim != 1 or not audio.size:
                raise ValueError("unexpected Irodori-TTS audio format")
            if not np.isfinite(audio).all():
                raise ValueError("Irodori-TTS returned non-finite audio")
            pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
            send("audio", pcm=base64.b64encode(pcm.tobytes()).decode("ascii"))
            send("done", samples=int(pcm.size))
        except Exception as error:
            traceback.print_exc()
            send("error", message=str(error))
        finally:
            if denoiser is not None:
                denoiser.reset_request()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        traceback.print_exc()
        send("error", message=str(error))
        sys.exit(1)
