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

        _progress(5, "Connecting...")
        client = OpenAI(
            base_url=host,
            api_key=apiKey,
        )

        _progress(10, "Generating...")
        response = client.images.generate(
            model=model,
            prompt=inputData,
            moderation=moderation,
            quality=quality,
            style=style,
            size=f"{width}x{height}",
            response_format='b64_json',
            stream=True
        )
        _progress(100, "Finished")
        image_data = base64.b64decode(response)
        out = Path(tmpDir)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'wb') as f:
            f.write(image_data)
        _write_log(f"Saved to: {out}")
        _send({"type": "done", "result": {"filePath": str(out)}})
    except Exception as e:
        _write_log(e)
