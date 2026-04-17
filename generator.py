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

def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()

def _progress(percent: int, label: str) -> None:
    _send({"type": "progress", "percent": int(percent), "label": label})

def _log(message: str) -> None:
    _send({"type": "log", "message": message})
    _write_log(f"[LOG] {message}")


class ImageGenerator():
    MODEL_ID     = "text-to-image"
    DISPLAY_NAME = "Text to Image"
    VRAM_GB      = 0

    def __init__(self):
        self.outputs_dir = Path("outputs")
        _write_log("ImageGenerator instantiated")

    def is_downloaded(self) -> bool:
        return True

    def load(self) -> None:
        return

    def unload(self) -> None:
        return

    def generate(
        self,
        image_path: str,
        output_path: str,
        params: dict,
        prompt: str = "A photorealistic object on a white background",
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        try:
            width = params.get("width", 64)
            height = params.get("height", 64)
            host = params.get("host", "http://192.168.178.138:11434")
            apiKey = params.get("apiKey", "ollama")
            model = params.get("model", "x/flux2-klein")
            moderation = params.get("moderation", "low")
            quality = params.get("quality", "medium")
            style = params.get("style", "vivid")

            _log(f"Starting generation — prompt: {prompt!r}, model: {model}, size: {width}x{height}")
            _write_log(f"Params: {params}")

            client = OpenAI(
                base_url=host,
                api_key=apiKey,
            )

            partial_b64 = ""

            try:
                with client.images.generate(
                    model=model,
                    prompt=prompt,
                    moderation=moderation,
                    quality=quality,
                    style=style,
                    size=f"{width}x{height}",
                    response_format='b64_json',
                    stream=True
                ) as stream:
                    for event in stream:
                        _write_log(f"Stream event: {event.type}")
                        if event.type == "image.generation.progress":
                            partial_b64 = event.b64_json or partial_b64
                            if hasattr(event, 'progress') and event.progress is not None:
                                pct = int(event.progress * 100)
                                _write_log(f"Progress: {pct}%")
                                if progress_cb:
                                    progress_cb(pct, "Generating...")
                                _progress(pct, "Generating...")
                        elif event.type == "image.generation.completed":
                            partial_b64 = event.b64_json
                            _write_log("Generation completed event received")
                            if progress_cb:
                                progress_cb(100, "Done")
                            _progress(100, "Done")
            except Exception as e:
                _write_log(f"[ERROR] Stream failed: {e}\n{traceback.format_exc()}")
                raise

            if not partial_b64:
                _write_log("[ERROR] No b64 data received from stream")
                raise ValueError("No image data received from the model")

            try:
                image_data = base64.b64decode(partial_b64)
            except Exception as e:
                _write_log(f"[ERROR] base64 decode failed: {e}\n{traceback.format_exc()}")
                raise

            try:
                out = Path(output_path)
                out.parent.mkdir(parents=True, exist_ok=True)
                with open(out, 'wb') as f:
                    f.write(image_data)
                _write_log(f"Saved to: {out}")
            except Exception as e:
                _write_log(f"[ERROR] Failed to write output file: {e}\n{traceback.format_exc()}")
                raise

            _log(f"Saved to: {out}")
            return out

        except Exception as e:
            _write_log(f"[ERROR] generate() failed: {e}\n{traceback.format_exc()}")
            raise

    @classmethod
    def params_schema(cls) -> list:
        return [
            {
                "id": "model",
                "label": "Image Model",
                "type": "select",
                "default": "x/flux2-klein",
                "options": [
                    {"value": "x/z-image-turbo", "label": "x/z-image-turbo"},
                    {"value": "x/flux2-klein",   "label": "x/flux2-klein"}
                ],
                "tooltip": "The model to use for image generation"
            },
            {
                "id": "style",
                "label": "Style",
                "type": "select",
                "default": "vivid",
                "options": [
                    {"value": "natural", "label": "natural"},
                    {"value": "vivid",   "label": "vivid"}
                ],
                "tooltip": "The style of the image"
            },
            {
                "id": "moderation",
                "label": "Moderation",
                "type": "select",
                "default": "low",
                "options": [
                    {"value": "low",  "label": "low"},
                    {"value": "auto", "label": "auto"}
                ],
                "tooltip": "The moderation level for the image"
            },
            {
                "id": "quality",
                "label": "Quality",
                "type": "select",
                "default": "medium",
                "options": [
                    {"value": "low",      "label": "low"},
                    {"value": "medium",   "label": "medium"},
                    {"value": "high",     "label": "high"},
                    {"value": "standard", "label": "standard"},
                    {"value": "hd",       "label": "hd"},
                    {"value": "auto",     "label": "auto"}
                ],
                "tooltip": "The quality of the image"
            },
            {
                "id": "host",
                "label": "Host:Port",
                "type": "string",
                "default": "http://192.168.178.138:11434",
                "tooltip": "The host for image generation"
            },
            {
                "id": "apiKey",
                "label": "API Key",
                "type": "string",
                "default": "ollama",
                "tooltip": "The API key"
            },
            {
                "id": "width",
                "label": "Width",
                "type": "int",
                "default": 512,
                "min": 16,
                "max": 1024,
                "step": 1,
                "tooltip": "The width of the output image"
            },
            {
                "id": "height",
                "label": "Height",
                "type": "int",
                "default": 512,
                "min": 16,
                "max": 1024,
                "step": 1,
                "tooltip": "The height of the output image"
            },
        ]