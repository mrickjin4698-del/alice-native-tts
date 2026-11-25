"""Benchmark two Drama 3 manifests through the public PCM worker."""

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import threading
import time
import wave


TOOLS = Path(__file__).resolve().parent
WORKLOADS = {
    "standard": "你好，欢迎收听。现在正在测试 Drama 三的本地语音合成。",
    "short": "你好，今天声音清晰吗？",
    "long": "清晨的列车沿着河岸缓缓驶过田野和村庄。我们会继续播报，直到这段本地语音全部播放完毕。",
    "english": "This is a local speech synthesis test. Please keep listening until the message finishes.",
}
METRICS = (
    "first_ms", "audible_ms", "total_ms", "rtf", "audio_seconds", "first_chunk_seconds",
    "max_playback_gap_ms",
)


def device_memory(command):
    try:
        output = subprocess.check_output(command, text=True, timeout=5)
        return int(output.splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def save_wave(path, pcm, rate):
    with wave.open(str(path), "wb") as wav:
        wav.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)


def phase(args, name, manifest, workloads, phase_dir, worker_api, bench, smi):
    phase_dir.mkdir()
    command = [str(args.executable), "worker", str(manifest), "--device", "cuda"]
    if args.python:
        command += ["--python", str(args.python)]
    initial = device_memory(smi) if smi else None
    memory = [initial] if initial is not None else []
    stop = threading.Event()

    def monitor():
        while not stop.wait(0.25):
            usage = device_memory(smi)
            if usage is not None:
                memory.append(usage)

    observer = threading.Thread(target=monitor, daemon=True) if smi else None
    if observer:
        observer.start()
    try:
        with (phase_dir / "worker.log").open("w", encoding="utf-8") as log:
            started = time.perf_counter()
            worker = worker_api["Worker"](command, TOOLS.parent, os.environ.copy(), log, args.timeout)
            try:
                ready = worker.receive()
                startup_ms = (time.perf_counter() - started) * 1000
                expected = {"family": "drama3", "runtime": "audio_cpp", "streaming": "semantic_tokens",
                            "sample_rate": 44100, "languages": ["auto"]}
                if ready.get("type") != "ready" or ready.get("model") != expected:
                    raise ValueError(f"unexpected Drama 3 worker handshake: {ready}")
                print(f"{phase_dir.name} ready in {startup_ms:.0f} ms", flush=True)
                rows = {}
                for key, text in workloads:
                    bench["measure"](worker, f"{key}-warmup", text, 44100)
                    rows[key] = []
                    for index in range(args.samples):
                        timing, pcm = bench["measure"](worker, f"{key}-{index}", text, 44100)
                        rows[key].append(timing)
                        if index == 0:
                            save_wave(phase_dir / f"{key}.wav", pcm, 44100)
                        print(f"{phase_dir.name}/{key} {index + 1}: {timing['first_ms']:.0f} ms first PCM, "
                              f"{timing['total_ms']:.0f} ms complete", flush=True)
            finally:
                try:
                    if worker.process.poll() is None:
                        worker.finish()
                finally:
                    worker.close()
    finally:
        if observer:
            stop.set()
            observer.join()
    return {
        "mode": name, "directory": phase_dir.name, "manifest": str(manifest),
        "startup_ms": startup_ms, "initial_device_mib": initial,
        "peak_device_mib": max(memory) if memory else None, "runs": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--executable", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--python", type=Path)
    parser.add_argument("--workload", nargs="+", choices=WORKLOADS, default=list(WORKLOADS))
    parser.add_argument("--samples", type=int, default=3, help="timed requests per workload in each ABBA phase")
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()
    if args.samples < 1 or args.timeout < 1:
        parser.error("--samples and --timeout must be positive")
    for key in ("executable", "baseline", "candidate", "output", "python"):
        if getattr(args, key) is not None:
            setattr(args, key, getattr(args, key).resolve())
    args.output.mkdir(parents=True)
    worker_api = runpy.run_path(str(TOOLS / "check-standalone.py"))
    bench = runpy.run_path(str(TOOLS / "benchmark-native-diffusion.py"))
    smi_path = shutil.which("nvidia-smi")
    smi = [smi_path, "--query-gpu=memory.used", "--format=csv,noheader,nounits"] if smi_path else None
    workloads = [(key, WORKLOADS[key]) for key in args.workload]
    phases = []
    for index, name in enumerate(("baseline", "candidate", "candidate", "baseline"), 1):
        phases.append(phase(args, name, getattr(args, name), workloads, args.output / f"{index}-{name}",
                            worker_api, bench, smi))
        (args.output / "phases.json").write_text(json.dumps(phases, indent=2) + "\n", encoding="utf-8")
    comparisons = {}
    for key, text in workloads:
        comparisons[key] = {"text": text}
        for name in ("baseline", "candidate"):
            rows = [row for part in phases if part["mode"] == name for row in part["runs"][key]]
            comparisons[key][name] = bench["summary"](rows, METRICS)
    (args.output / "benchmark.json").write_text(
        json.dumps(comparisons, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
