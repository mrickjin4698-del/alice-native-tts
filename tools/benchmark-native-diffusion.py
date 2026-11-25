"""Compare native diffusion packs and matched checkpoints through the JSONL worker."""

import argparse
from array import array
import base64
import hashlib
import json
from math import sqrt
import os
from pathlib import Path
import runpy
from statistics import median, quantiles
import subprocess
import sys
import time
import wave


WORKLOADS = {
    "short": "今天测试本地语音。",
    "standard": "今天我们测试第二版语音，后面的内容会继续生成，直到整个回复播放完毕。",
    "rolling": "清晨的列车沿着河岸缓缓驶过田野和村庄车窗里映着远处的山影和刚刚亮起的天空。",
}
PROFILE_PREFIX = "[alice-tts] native diffusion "


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def measure(worker, key, text, sample_rate):
    started = time.perf_counter_ns()
    worker.send({"op": "synthesize", "request_id": key, "text": text})
    chunks = []
    arrivals = []
    first = None
    while True:
        packet = worker.receive()
        arrived = time.perf_counter_ns()
        if packet.get("request_id") != key:
            raise ValueError(f"unexpected response: {packet.get('type')}")
        if packet["type"] == "audio":
            if packet["sample_rate"] != sample_rate:
                raise ValueError("worker changed sample rate")
            if first is None:
                first = arrived
            chunks.append(packet["pcm"])
            arrivals.append((arrived - started) / 1e6)
        elif packet["type"] == "done":
            if first is None:
                raise ValueError("worker produced no audio")
            audio = [base64.b64decode(chunk, validate=True) for chunk in chunks]
            pcm = b"".join(audio)
            samples = sum(len(chunk) // 2 for chunk in audio)
            if any(not chunk or len(chunk) % 2 for chunk in audio) or samples != packet["samples"] or not any(pcm):
                raise ValueError("failed PCM validation")
            duration = samples / sample_rate
            complete = (arrived - started) / 1e6
            timing = {
                "first_ms": (first - started) / 1e6, "total_ms": complete, "rtf": complete / 1000 / duration,
                "audio_seconds": duration, "first_chunk_seconds": len(audio[0]) / 2 / sample_rate,
                "chunks": len(audio), "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
                "chunk_samples": [len(chunk) // 2 for chunk in audio],
                "chunk_pcm_sha256": [hashlib.sha256(chunk).hexdigest() for chunk in audio],
                **playback_metrics(audio, arrivals, sample_rate, pcm),
            }
            return timing, pcm
        else:
            raise ValueError(packet.get("message", str(packet)))


def playback_metrics(chunks, arrivals, sample_rate, pcm):
    """Estimate onset and starvation for immediate playback without a prebuffer."""
    samples = array("h", pcm)
    if sys.byteorder == "big":
        samples.byteswap()
    frame = sample_rate // 100
    threshold = frame * 32768**2 * 10**(-4.5)
    onset = next((index for index in range(0, len(samples) - frame + 1, frame)
                  if sum(value**2 for value in samples[index:index + frame]) >= threshold), None)
    audible = None
    gap = 0.0
    buffer_end = arrivals[0]
    offset = 0
    for chunk, arrival in zip(chunks, arrivals):
        count = len(chunk) // 2
        gap = max(gap, arrival - buffer_end)
        playback = max(arrival, buffer_end)
        if onset is not None and offset <= onset < offset + count:
            audible = playback + (onset - offset) * 1000 / sample_rate
        buffer_end = playback + count * 1000 / sample_rate
        offset += count
    return {"audible_ms": audible, "onset_audio_ms": None if onset is None else onset * 1000 / sample_rate,
            "max_playback_gap_ms": gap}


def pcm_relative_l2(reference, current):
    if reference == current:
        return 0.0
    if len(reference) != len(current):
        return None
    reference_samples = array("h", reference)
    current_samples = array("h", current)
    if sys.byteorder == "big":
        reference_samples.byteswap()
        current_samples.byteswap()
    difference = sum((a - b) ** 2 for a, b in zip(reference_samples, current_samples))
    energy = sum(value ** 2 for value in reference_samples)
    return sqrt(difference / energy)


def summary(samples, names):
    result = {}
    for name in names:
        series = [row[name] for row in samples if row[name] is not None]
        if series:
            p95 = series[0] if len(series) == 1 else quantiles(series, n=100, method="inclusive")[94]
            result[name] = {"p50": median(series), "p95": p95}
    return result


def bench(worker_class, args, model, family, is_native, runtime, sample_rate):
    device = args.native_device if is_native else args.checkpoint_device
    label = family + ("_native" if is_native else "_checkpoint")
    env = os.environ.copy()
    if is_native:
        env["ALICE_NATIVE_TTS_DIFFUSION_PROFILE"] = "1"
    command = [str(args.executable), "worker", str(model), "--device", device]
    if not is_native:
        command += ["--gpt-sovits-root", str(runtime), "--python", str(args.python)]
    print(f"Running {label} ({device})", flush=True)
    log_path = args.output / (label + ".stderr.log")
    result = {"device": device, "sample_rate": sample_rate, "workloads": {}}
    with log_path.open("w", encoding="utf-8") as log:
        started = time.perf_counter_ns()
        worker = worker_class(command, args.output, env, log, args.timeout)
        try:
            ready = worker.receive()
            result["ready_ms"] = (time.perf_counter_ns() - started) / 1e6
            if (ready["type"] != "ready" or ready["device"] != device
                    or ready["model"]["family"] != family or ready["model"]["sample_rate"] != sample_rate
                    or ready["model"]["streaming"] != "text_fragments"):
                raise ValueError(f"unexpected worker handshake: {ready}")
            for name in args.workload:
                text = WORKLOADS[name]
                for index in range(args.warmups):
                    measure(worker, f"{name}-warmup-{index}", text, sample_rate)
                samples = []
                reference_pcm = None
                for index in range(args.samples):
                    measured, pcm = measure(worker, f"{name}-{index}", text, sample_rate)
                    if reference_pcm is None:
                        reference_pcm = pcm
                        if args.save_wav:
                            with wave.open(str(args.output / f"{label}-{name}.wav"), "wb") as wav:
                                wav.setparams((1, 2, sample_rate, 0, "NONE", "not compressed"))
                                wav.writeframes(pcm)
                    measured["pcm_relative_l2_vs_first"] = pcm_relative_l2(reference_pcm, pcm)
                    samples.append(measured)
                timing = summary(samples, ("first_ms", "total_ms", "rtf", "audio_seconds", "first_chunk_seconds",
                                           "audible_ms", "onset_audio_ms", "max_playback_gap_ms"))
                result["workloads"][name] = {"samples": samples, "summary": timing}
                print(f"  {name}: first={timing['first_ms']['p50']:.1f} ms, "
                      f"total={timing['total_ms']['p50']:.1f} ms, RTF={timing['rtf']['p50']:.3f}", flush=True)
            worker.finish()
        finally:
            worker.close()

    if is_native:
        stages = [json.loads(line.split(PROFILE_PREFIX, 1)[1]) for line in log_path.read_text(
            encoding="utf-8", errors="replace").splitlines() if PROFILE_PREFIX in line]
        stride = args.warmups + args.samples
        if len(stages) != stride * len(args.workload):
            raise ValueError("missing native stage timing records")
        for index, name in enumerate(args.workload):
            rows = stages[index * stride + args.warmups:(index + 1) * stride]
            steps = 4 if family == "v5turbo" else 32
            if any(row["denoise_steps"] != steps * row["chunks"] for row in rows):
                raise ValueError(f"unexpected denoising step count for {family} {name}")
            if name == "rolling" and any(row["chunks"] <= row["fragments"] for row in rows):
                raise ValueError(f"{family} rolling workload did not exercise multiple windows")
            result["workloads"][name]["native_stages"] = rows
            result["workloads"][name]["native_stage_summary"] = summary(rows, tuple(rows[0]))
    return label, result


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--executable", type=Path, required=True)
    cli.add_argument("--python", type=Path, required=True)
    cli.add_argument("--model", action="append", nargs=3, required=True, metavar=("PACK", "CHECKPOINT", "RUNTIME"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--native-device", default="tensor_rt_native")
    cli.add_argument("--checkpoint-device", choices=["cuda", "tensor_rt"], default="tensor_rt")
    cli.add_argument("--checkpoint-report", type=Path, help="Reuse matching checkpoint measurements from this benchmark.json")
    cli.add_argument("--samples", type=int, default=8)
    cli.add_argument("--warmups", type=int, default=3)
    cli.add_argument("--timeout", type=int, default=300)
    cli.add_argument("--workload", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    cli.add_argument("--save-wav", action="store_true", help="Save the first measured PCM for each new case")
    args = cli.parse_args()
    if args.samples < 2 or args.warmups < 1 or len(set(args.workload)) != len(args.workload):
        cli.error("requires two or more samples, one or more warmups and unique workloads")
    for name in ("executable", "python", "output"):
        setattr(args, name, getattr(args, name).resolve())
    baseline = read(args.checkpoint_report) if args.checkpoint_report else None
    selected = {name: WORKLOADS[name] for name in args.workload}
    if baseline:
        if (baseline["samples_per_case"] != args.samples or baseline["warmups"] != args.warmups
                or any(baseline["workloads"].get(name) != text for name, text in selected.items())):
            cli.error("checkpoint report has different workloads or sample counts")
    args.output.mkdir(parents=True, exist_ok=False)
    worker = runpy.run_path(str(Path(__file__).with_name("check-standalone.py")))["Worker"]
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    report = {"gpu": gpu, "executable_sha256": sha(args.executable), "warmups": args.warmups,
              "samples_per_case": args.samples, "workloads": selected,
              "case_order": [], "models": {}, "cases": {}}
    if baseline:
        if baseline["gpu"] != gpu or gpu is None:
            cli.error("checkpoint report must come from the same GPU and driver")
        report["checkpoint_source_sha256"] = sha(args.checkpoint_report)
    if args.native_device == "tensor_rt_native":
        report["native_tensorrt_acoustics"] = bool(os.environ.get("ALICE_NATIVE_TTS_DIFFUSION_TRT_ROOT"))
        report["native_tensorrt_target"] = os.environ.get("ALICE_NATIVE_TTS_DIFFUSION_TRT_TARGET", "")
        report["native_gpt_fp16_cache"] = bool(os.environ.get("ALICE_NATIVE_TTS_DIFFUSION_GPT_FP16_DIR"))
        report["native_cuda_target_tuning"] = os.environ.get("ALICE_NATIVE_TTS_DIFFUSION_CUDA_TARGET", "")
        report["native_v3_target_buckets"] = bool(os.environ.get("ALICE_NATIVE_TTS_DIFFUSION_V3_BUCKETS"))
        report["native_rolling_pcm"] = os.environ.get("ALICE_NATIVE_TTS_DIFFUSION_ROLLING_PCM") == "1"
        report["native_v5_first_pcm_tokens"] = os.environ.get("ALICE_NATIVE_TTS_V5_FIRST_PCM_TOKENS")
    path = args.output / "benchmark.json"
    for pack_dir, checkpoint_path, runtime in args.model:
        pack_dir, checkpoint_path, runtime = [Path(value).resolve() for value in (pack_dir, checkpoint_path, runtime)]
        config = read(pack_dir / "pack.json")
        checkpoint = read(checkpoint_path)
        family = config["family"]
        if family in report["models"] or config["abi"] != 2 or checkpoint["family"] != family:
            raise ValueError(f"invalid or duplicate model pair {family}")
        for source, fingerprint in (("gpt", "gpt_checkpoint_sha256"), ("sovits", "sovits_checkpoint_sha256"),
                                    ("reference_audio", "reference_audio_sha256")):
            if sha(checkpoint_path.parent / checkpoint[source]) != config[fingerprint]:
                raise ValueError(f"checkpoint and pack differ in {family} {source}")
        report["models"][family] = {key: config[key] for key in (
            "gpt_checkpoint_sha256", "sovits_checkpoint_sha256", "reference_audio_sha256",
            "precision", "export_validation")}
        for is_native in (True, False):
            if baseline and not is_native:
                label = family + "_checkpoint"
                result = baseline["cases"][label]
                if (baseline["models"][family] != report["models"][family]
                        or result["device"] != args.checkpoint_device or result["sample_rate"] != config["sample_rate"]):
                    raise ValueError(f"checkpoint report differs in {family} weights, sample rate or device")
                result = {**result, "workloads": {name: result["workloads"][name] for name in selected}}
            else:
                label, result = bench(worker, args, pack_dir if is_native else checkpoint_path, family, is_native,
                                      runtime, config["sample_rate"])
            report["case_order"].append(label)
            report["cases"][label] = result
            draft = path.with_suffix(".tmp")
            draft.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            draft.replace(path)
    for workload in args.workload:
        rows = {label: record["workloads"][workload] for label, record in report["cases"].items()}
        (args.output / (workload + ".json")).write_text(
            json.dumps({"text": WORKLOADS[workload], "results": rows}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(f"Saved {path}", flush=True)


if __name__ == "__main__":
    main()
