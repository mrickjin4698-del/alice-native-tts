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
            }
            return timing, pcm
        else:
            raise ValueError(packet.get("message", str(packet)))


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
        series = [row[name] for row in samples]
        result[name] = {"p50": median(series), "p95": quantiles(series, n=100, method="inclusive")[94]}
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
                    measured["pcm_relative_l2_vs_first"] = pcm_relative_l2(reference_pcm, pcm)
                    samples.append(measured)
                timing = summary(samples, ("first_ms", "total_ms", "rtf", "audio_seconds", "first_chunk_seconds"))
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
    cli.add_argument("--samples", type=int, default=8)
    cli.add_argument("--warmups", type=int, default=3)
    cli.add_argument("--timeout", type=int, default=300)
    cli.add_argument("--workload", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    args = cli.parse_args()
    if args.samples < 2 or args.warmups < 1 or len(set(args.workload)) != len(args.workload):
        cli.error("requires two or more samples, one or more warmups and unique workloads")
    for name in ("executable", "python", "output"):
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    worker = runpy.run_path(str(Path(__file__).with_name("check-standalone.py")))["Worker"]
    try:
        gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                       "--format=csv,noheader"], text=True, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        gpu = None
    report = {"gpu": gpu, "executable_sha256": sha(args.executable), "warmups": args.warmups,
              "samples_per_case": args.samples, "workloads": {name: WORKLOADS[name] for name in args.workload},
              "case_order": [], "models": {}, "cases": {}}
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
