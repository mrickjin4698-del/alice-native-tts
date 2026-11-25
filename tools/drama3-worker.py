"""Forward audio.cpp speech events into the Alice JSONL worker protocol."""

import base64
from contextlib import closing
import http.client
import json
import sys
import traceback


def send(kind, **fields):
    print(json.dumps({"type": kind, **fields}, ensure_ascii=False), flush=True)


def connection(spec):
    return closing(http.client.HTTPConnection("127.0.0.1", spec["port"], timeout=180))


def require_success(response):
    if response.status != 200:
        body = response.read(2048).decode("utf-8", errors="replace")
        raise RuntimeError(f"audio.cpp returned HTTP {response.status}: {body}")


def synthesize(spec, text):
    body = {
        "model": spec["model"], "input": text, "response_format": "pcm", "stream_format": "sse",
        "options": spec["options"],
    }
    if spec["reference_audio"] is not None:
        body["voice_ref"] = {"type": "path", "path": spec["reference_audio"]}
    if spec["reference_text"] is not None:
        body["reference_text"] = spec["reference_text"]
    with connection(spec) as client:
        client.request("POST", "/v1/audio/speech", json.dumps(body).encode("utf-8"),
                       {"Content-Type": "application/json", "Accept": "text/event-stream"})
        with client.getresponse() as response:
            require_success(response)
            if response.headers.get_content_type() != "text/event-stream":
                raise RuntimeError("audio.cpp did not return an event stream")
            samples = 0
            done = False
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    if done and samples:
                        return samples
                    break
                event = json.loads(payload)
                kind = event["type"]
                if kind == "speech.audio.delta":
                    if int(event.get("sample_rate", spec["sample_rate"])) != spec["sample_rate"]:
                        raise RuntimeError("audio.cpp sample rate changed during a request")
                    if int(event.get("channels", 1)) != 1:
                        raise RuntimeError("audio.cpp returned non-mono PCM")
                    chunk = base64.b64decode(event["audio"], validate=True)
                    if len(chunk) % 2:
                        raise RuntimeError("audio.cpp returned an incomplete PCM sample")
                    if chunk:
                        send("audio", pcm=event["audio"])
                        samples += len(chunk) // 2
                elif kind == "speech.audio.done":
                    done = True
                elif kind == "error":
                    error = event.get("error")
                    raise RuntimeError(error.get("message", "audio.cpp synthesis failed") if isinstance(error, dict)
                                       else str(error or event))
    raise RuntimeError("audio.cpp stream ended before a complete request")


def main():
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")
    spec = json.loads(sys.stdin.readline())
    with connection(spec) as client:
        client.request("GET", "/v1/models")
        with client.getresponse() as response:
            require_success(response)
            models = json.load(response)["data"]
            if not any(model["id"] == spec["model"] for model in models):
                raise RuntimeError("audio.cpp did not load the configured Drama 3 model")
    send("ready", sample_rate=spec["sample_rate"], languages=["auto"])
    for line in sys.stdin:
        try:
            request = json.loads(line)
            samples = synthesize(spec, request["text"])
            send("done", samples=samples)
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
