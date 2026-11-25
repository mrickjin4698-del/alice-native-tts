"""Measure gsv.cpp V5 over loopback HTTP on the native diffusion fixtures."""

import argparse
import hashlib
import http.client
import json
from pathlib import Path
import runpy
import struct
import time
import wave


BENCH = runpy.run_path(str(Path(__file__).with_name("benchmark-native-diffusion.py")))
SAMPLE_RATE = 48000


def measure(connection, payload):
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    started = time.perf_counter_ns()
    connection.request("POST", "/tts", body=body, headers={"Content-Type": "application/json"})
    response = connection.getresponse()
    if response.status != 200:
        message = response.read(8192).decode(errors="replace")
        raise RuntimeError(f"HTTP {response.status}: {message}")
    header = response.read(44)
    if (header[:4], header[8:12], header[36:40]) != (b"RIFF", b"WAVE", b"data"):
        raise ValueError("unexpected streaming WAV header")
    if struct.unpack_from("<HHI", header, 20) != (1, 1, SAMPLE_RATE) or header[34:36] != b"\x10\x00":
        raise ValueError("expected mono 48 kHz PCM16")

    chunks = []
    arrivals = []
    carry = b""
    while True:
        data = response.read1(65536)
        arrived = (time.perf_counter_ns() - started) / 1e6
        if not data:
            break
        data = carry + data
        end = len(data) & ~1
        carry = data[end:]
        if end:
            chunks.append(data[:end])
            arrivals.append(arrived)
    pcm = b"".join(chunks)
    if carry or not pcm or not any(pcm):
        raise ValueError("empty, silent or truncated PCM")
    seconds = len(pcm) / (2 * SAMPLE_RATE)
    return {
        "first_ms": arrivals[0], "total_ms": arrived, "rtf": arrived / 1000 / seconds,
        "audio_seconds": seconds, "first_chunk_seconds": len(chunks[0]) / (2 * SAMPLE_RATE),
        "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
        **BENCH["playback_metrics"](chunks, arrivals, SAMPLE_RATE, pcm),
    }, pcm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=9880)
    parser.add_argument("--manifest", type=Path, required=True, help="Alice V5 checkpoint manifest")
    parser.add_argument("--ref-audio", type=Path, help="Identical WAV in the gsv.cpp allowed directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dbcache", choices=("off", "auto"), default="off")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--workload", nargs="+", choices=BENCH["WORKLOADS"], default=list(BENCH["WORKLOADS"]))
    args = parser.parse_args()
    manifest = BENCH["read"](args.manifest)
    family = manifest["family"]
    if family not in ("v5turbo", "v5dev") or args.samples < 2 or args.warmups < 1:
        parser.error("requires a V5 manifest, at least two samples and one warmup")
    steps, cfg = (32, 1.30) if family == "v5dev" else (4, 0.0)
    original_ref = args.manifest.parent / manifest["reference_audio"]
    reference = args.ref_audio or original_ref
    reference_digest = BENCH["sha"](reference)
    if reference_digest != BENCH["sha"](original_ref):
        raise ValueError("peer reference audio and native manifest differ")
    args.output.mkdir(parents=True, exist_ok=False)
    params = {
        "ref_audio_path": str(reference.resolve()), "prompt_text": manifest["reference_text"],
        "prompt_lang": manifest["reference_language"], "text_lang": "zh", "sample_steps": steps,
        "cfg_rate": cfg, "seed": 1234, "top_k": 15, "top_p": 1.0, "temperature": 1.0,
        "repetition_penalty": 1.0, "text_split_method": "cut5", "fragment_interval": 0.3,
        "voice": family, "stream": True, "media_type": "wav", "dbcache": args.dbcache,
    }
    report = {"family": family, "steps": steps, "guidance": cfg, "dbcache": args.dbcache,
              "samples": args.samples, "warmups": args.warmups, "reference_audio_sha256": reference_digest,
              "workloads": {name: BENCH["WORKLOADS"][name] for name in args.workload}, "cases": {}}
    connection = http.client.HTTPConnection("127.0.0.1", args.port, timeout=300)
    try:
        for name, text in report["workloads"].items():
            payload = dict(params, text=text)
            for _ in range(args.warmups):
                measure(connection, payload)
            samples = []
            for index in range(args.samples):
                timing, pcm = measure(connection, payload)
                if index == 0:
                    with wave.open(str(args.output / f"{name}.wav"), "wb") as wav:
                        wav.setparams((1, 2, SAMPLE_RATE, 0, "NONE", "not compressed"))
                        wav.writeframes(pcm)
                samples.append(timing)
            totals = BENCH["summary"](samples, ("first_ms", "audible_ms", "total_ms", "rtf",
                                                "audio_seconds", "first_chunk_seconds", "max_playback_gap_ms"))
            report["cases"][name] = {"samples": samples, "summary": totals}
            print(name, {key: round(value["p50"], 3) for key, value in totals.items()}, flush=True)
            (args.output / "benchmark.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                                      encoding="utf-8")
    finally:
        connection.close()


if __name__ == "__main__":
    main()
