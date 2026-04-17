import base64
from openai import OpenAI
import threading
from pathlib import Path
import time
import uuid
from typing import Callable, Optional
import sys
import json
import traceback

LOG_FILE = "C:\\Users\\dariu\\Downloads\\log.log"

def _write_log(text: str) -> None:
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")

_write_log("Started")

def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def _progress(percent: int, label: str) -> None:
    _send({"type": "progress", "percent": int(percent), "label": label})

def _log(message: str) -> None:
    _send({"type": "log", "message": message})
    _write_log(f"[LOG] {message}")

def _readData() -> dict:
    raw = sys.stdin.readline()
    return json.loads(raw)

#{"input":{"text":"A cute cat","nodeId":"generate"},"params":{},"nodeId":"generate","workspaceDir":"C:\\Users\\dariu\\Documents\\Modly\\workspace","tempDir":"C:\\Users\\dariu\\AppData\\Local\\Temp"}

if __name__ == "__main__":

    try:
        data = _readData()
        inputData: str = data.get("input")
        params: dict = data.get("params")
        tmpDir: str = data.get("tempDir")

        _write_log(json.dumps(data))

        width = params.get("width", 64)
        height = params.get("height", 64)
        host = params.get("host", "http://192.168.178.138:11434/v1/")
        apiKey = params.get("apiKey", "ollama")
        model = params.get("model", "x/flux2-klein")
        moderation = params.get("moderation", "low")
        quality = params.get("quality", "medium")
        style = params.get("style", "vivid")
        _log(f"Starting generation — prompt: {inputData!r}, model: {model}, size: {width}x{height}")
        _write_log(f"Params: {params}")

        _write_log("width: " + str(width))
        _write_log("height: " + str(height))
        _write_log("host: " + host)
        _write_log("apiKey: " + apiKey)
        _write_log("model: " + model)
        _write_log("moderation: " + moderation)
        _write_log("quality: " + quality)
        _write_log("style: " + style)

        client = OpenAI(
            base_url=host,
            api_key=apiKey,
        )
        partial_b64 = ""
        with client.images.generate(
            model=model,
            prompt=inputData,
            moderation=moderation,
            quality=quality,
            style=style,
            size=f"{width}x{height}",
            response_format='b64_json',
            stream=True
        ) as stream:
            _write_log(stream)
            for event in stream:
                _write_log(f"Stream event: {event.type}")
                if event.type == "image.generation.progress":
                    partial_b64 = event.b64_json or partial_b64
                    if hasattr(event, 'progress') and event.progress is not None:
                        pct = int(event.progress * 100)
                        _write_log(f"Progress: {pct}%")
                        if _progress:
                            _progress(pct, "Generating...")
                        _progress(pct, "Generating...")
                elif event.type == "image.generation.completed":
                    partial_b64 = event.b64_json
                    _write_log("Generation completed event received")
                    if _progress:
                        _progress(100, "Done")
                    _progress(100, "Done")
        image_data = base64.b64decode(partial_b64)
        out = Path(tmpDir)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            f.write(image_data)
        _write_log(f"Saved to: {out}")
        _send({"type": "done", "result": {"filePath": str(out)}})
    except Exception as e:
        _write_log(e)
