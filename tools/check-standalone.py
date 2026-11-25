"""Exercise the public Alice TTS commands from a separate Python process."""

import argparse
import base64
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import wave

WEIGHTS = {
    "v1": ("s1bert25hz-2kh-longer-epoch=68e-step=50232.ckpt", "s2G488k.pth"),
    "v2": (
        "gsv-v2final-pretrained/s1bert25hz-5kh-longer-epoch=12-step=369668.ckpt",
        "gsv-v2final-pretrained/s2G2333k.pth",
    ),
    "v2Pro": ("s1v3.ckpt", "v2Pro/s2Gv2Pro.pth"),
    "v2ProPlus": ("s1v3.ckpt", "v2Pro/s2Gv2ProPlus.pth"),
    "v3": ("s1v3.ckpt", "s2Gv3.pth"),
    "v4": ("s1v3.ckpt", "gsv-v4-pretrained/s2Gv4.pth"),
}
TEXT = "今天我们测试本地语音，后面的内容会继续生成，直到整个回复播放完毕。"
FRAGMENTS = "这是一条本地语音合成测试。接下来继续播报第二句话，验证分段音频能够完整返回。"


def require(condition, message):
    if not condition:
        raise ValueError(message)


class Worker:
    def __init__(self, command, directory, environment, log, timeout):
        self.timeout = timeout
        self.events = queue.Queue()
        self.process = subprocess.Popen(
            command, cwd=directory, env=environment, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
        )
        threading.Thread(target=self.read, daemon=True).start()

    def read(self):
        try:
            for line in self.process.stdout:
                self.events.put(json.loads(line))
        except (UnicodeError, ValueError) as error:
            self.events.put(error)
        finally:
            self.events.put(EOFError("worker stdout closed"))

    def receive(self):
        event = self.events.get(timeout=self.timeout)
        if isinstance(event, Exception):
            raise event
        return event

    def send(self, *requests):
        self.process.stdin.write("".join(json.dumps(request) + "\n" for request in requests))
        self.process.stdin.flush()

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        self.process.stdin.close()

    def finish(self):
        self.send({"op": "shutdown"})
        require(self.process.wait(timeout=self.timeout) == 0, "worker shutdown failed")


def request(worker, key, text, expected_rate, cancel=False):
    start = time.monotonic()
    worker.send({"op": "synthesize", "request_id": key, "text": text})
    count = 0
    first = None
    wave_bytes = bytearray()
    while True:
        event = worker.receive()
        require(event.get("request_id") == key, "worker returned an event for another request")
        if event["type"] == "audio":
            require(event["sample_rate"] == expected_rate, "PCM sample rate changed")
            chunk = base64.b64decode(event["pcm"], validate=True)
            require(len(chunk) > 0 and len(chunk) % 2 == 0, "PCM chunk has invalid length")
            if first is None:
                first = (time.monotonic() - start) * 1000
                if cancel:
                    worker.send({"op": "cancel", "request_id": key})
            count += 1
            wave_bytes.extend(chunk)
        elif event["type"] == "cancelled" and cancel:
            require(count > 0, "cancellation did not follow an audio chunk")
            return {"terminal": "cancelled", "audio_chunks": count}, None
        elif event["type"] == "done" and not cancel:
            require(first is not None and any(wave_bytes), "successful request has no audible PCM")
            require(event["samples"] == len(wave_bytes) // 2, "terminal sample accounting differs")
            return {
                "terminal": "done", "audio_chunks": count, "samples": event["samples"],
                "first_audio_ms": round(first, 2), "sample_rate": expected_rate,
            }, wave_bytes
        else:
            raise ValueError(f"unexpected terminal event {event}")


def exercise(args, mode, model, device, expected_rate, streaming, check_cancel, env):
    start = time.monotonic()
    options = ["--device", device]
    if mode != "native":
        options += ["--gpt-sovits-root", str(args.gpt_sovits_root)]
        if args.python:
            options += ["--python", str(args.python)]
    cmd = [str(args.executable), "worker", str(model), *options]
    if check_cancel:
        cmd += ["--warmup-text", TEXT]
    with (args.output / (mode + ".log")).open("w", encoding="utf-8") as log:
        worker = Worker(cmd, args.output, env, log, args.timeout)
        try:
            ready = worker.receive()
            require(ready["type"] == "ready" and ready["protocol_version"] == 1, "unsupported worker handshake")
            require(ready["pcm_format"] == "s16le" and ready["channels"] == 1, "unsupported worker PCM")
            require(ready["device"] == device, "worker device differs")
            info = ready["model"]
            require(info["sample_rate"] == expected_rate and info["streaming"] == streaming, "unexpected model capabilities")
            require(info["family"] == ("v2Pro" if mode == "native" else mode), "unexpected model family")
            elapsed = round((time.monotonic() - start) * 1000, 2)
            first, pcm = request(worker, "first", FRAGMENTS if streaming == "text_fragments" else TEXT, expected_rate)
            require(first["audio_chunks"] > 1, "streaming test did not exercise multiple chunks")
            result = {"status": "passed", "model": info, "startup_ms": elapsed, "synthesis": first}
            if check_cancel:
                cancelled, _ = request(worker, "interrupt", FRAGMENTS if mode != "native" else TEXT, expected_rate, cancel=True)
                recovered, _ = request(worker, "after-interrupt", TEXT, expected_rate)
                result.update(cancellation=cancelled, recovery=recovered)
            worker.finish()
        finally:
            worker.close()
    with wave.open(str(args.output / (mode + ".wav")), "wb") as wav:
        wav.setparams((1, 2, expected_rate, 0, "NONE", "not compressed"))
        wav.writeframes(pcm)
    if mode == "v4":
        wav_out = args.output / "v4-cli.wav"
        rendered = subprocess.run(
            [str(args.executable), "synthesize", str(model), "--text", TEXT, "--output", str(wav_out), *options],
            text=True, encoding="utf-8", cwd=args.output, env=env, capture_output=True, timeout=args.timeout,
        )
        require(rendered.returncode == 0, rendered.stderr[-1000:])
        with wave.open(str(wav_out), "rb") as wav:
            require(wav.getframerate() == expected_rate and wav.getnchannels() == 1 and wav.getsampwidth() == 2,
                    "WAV command emitted the wrong format")
            require(wav.getnframes() > 0 and any(wav.readframes(wav.getnframes())), "WAV command emitted silence")
        result["wav_command"] = "passed"
    return result


def main():
    parse = argparse.ArgumentParser(description=__doc__)
    parse.add_argument("--executable", required=True, type=Path)
    parse.add_argument("--native-pack", required=True, type=Path)
    parse.add_argument("--gpt-sovits-root", type=Path)
    parse.add_argument("--python", type=Path)
    parse.add_argument("--output", required=True, type=Path)
    parse.add_argument("--native-device", default="cpu")
    parse.add_argument("--checkpoint-device", choices=("cpu", "cuda"), default="cuda")
    parse.add_argument("--families", nargs="*", choices=WEIGHTS, default=list(WEIGHTS))
    parse.add_argument("--skip-native", action="store_true")
    parse.add_argument("--timeout", default=180, type=int)
    args = parse.parse_args()
    if args.families and args.gpt_sovits_root is None:
        parse.error("--gpt-sovits-root is required for checkpoint tests")
    for key in ("executable", "native_pack", "gpt_sovits_root", "python", "output"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    pack = json.loads((args.native_pack / "pack.json").read_text(encoding="utf-8"))
    env = {key: value for key, value in os.environ.items() if not key.startswith("EVENT_STREAM_")}
    tests = []
    if not args.skip_native:
        tests.append(("native", args.native_pack, args.native_device, 32000, "semantic_tokens", True))
    for family in args.families:
        pre = args.gpt_sovits_root / "GPT_SoVITS" / "pretrained_models"
        gpt, sovits = WEIGHTS[family]
        manifest = args.output / f"{family}.json"
        manifest.write_text(json.dumps({
            "schema_version": 1, "family": family, "gpt": str(pre / gpt), "sovits": str(pre / sovits),
            "reference_audio": str(args.native_pack / pack["reference_audio"]),
            "reference_text": pack["reference_text"], "reference_language": "zh", "default_language": "zh",
        }, ensure_ascii=False), encoding="utf-8")
        rate = {"v3": 24000, "v4": 48000}.get(family, 32000)
        streaming = "text_fragments" if family in ("v3", "v4") else "semantic_tokens"
        tests.append((family, manifest, args.checkpoint_device, rate, streaming, family == "v4"))
    records = {}
    for mode, *options in tests:
        print(f"Testing {mode}", flush=True)
        try:
            records[mode] = exercise(args, mode, *options, env)
        except (OSError, EOFError, ValueError, queue.Empty, subprocess.TimeoutExpired) as error:
            records[mode] = {"status": "failed", "error": str(error) or type(error).__name__}
        print(json.dumps({mode: records[mode]}, ensure_ascii=False), flush=True)
        (args.output / "results.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0 if records and all(item["status"] == "passed" for item in records.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
