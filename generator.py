import base64
from openai import OpenAI
from services.generators.base import BaseGenerator, smooth_progress, GenerationCancelled
import threading
from pathlib import Path
import time
import uuid
from typing import Callable, Optional
import sys
import json


def _send(message: dict) -> None:
    sys.stdout.write(json.dumps(message) + "\n")
    sys.stdout.flush()


def _progress(percent: int, label: str) -> None:
    _send({"type": "progress", "percent": int(percent), "label": label})


def _log(message: str) -> None:
    _send({"type": "log", "message": message})

class ImageGenerator(BaseGenerator):
    MODEL_ID     = "text-to-image"
    DISPLAY_NAME = "Text to Image"
    VRAM_GB      = 0
    
    def is_downloaded(self) -> bool:
        return True
    
    def load(self) -> None:
        return
    
    def unload(self) -> None:
        return
    
    def generate(
        self,
        image_bytes: bytes,
        params: dict,
        progress_cb: Optional[Callable[[int, str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Path:
        width = params.get("width", 64)
        height = params.get("height", 64)
        host = params.get("host", "http://192.168.178.138:11434")
        apiKey = params.get("apiKey", "ollama")
        model = params.get("model", "x/flux2-klein")

        _log(params.__str__())

        client = OpenAI(
            base_url=host,
            api_key=apiKey,  # required but ignored
        )
        moderation = params.get("moderation", "low")
        quality = params.get("quality", "medium")
        style = params.get("style", "vivid")


        partial_b64 = ""
        total_chunks = 0

        # First pass to count isn't possible with streams, so we track by completion event
        with client.images.generate(
            model=model,
            prompt=prompt,
            moderation=moderation,
            quality=quality,
            style=style,
            size=str(width) + "x" + str(height),
            response_format='b64_json',
            stream=True
        ) as stream:
            for event in stream:
                # Event types: 'image.generation.progress' and 'image.generation.completed'
                if event.type == "image.generation.progress":
                    partial_b64 = event.b64_json or partial_b64
                    if hasattr(event, 'progress') and event.progress is not None:
                        if progress_cb:
                            progress_cb(int(event.progress * 100), "Generating...")
                elif event.type == "image.generation.completed":
                    partial_b64 = event.b64_json
                    if progress_cb:
                        progress_cb(100, "Done")

        image_data = base64.b64decode(partial_b64)

        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        name = f"{int(time.time())}_{uuid.uuid4().hex[:8]}.png"
        path = self.outputs_dir / name
        print("generation: ", path)
        with open(path, 'wb') as f:
            f.write(image_data)

        return path

    
    @classmethod
    def params_schema(cls) -> list:
        return [
                {
                    "id": "model",
                    "label": "Image Model",
                    "type": "select",
                    "default": "x/flux2-klein",
                    "options": [
                        {
                            "value": "x/z-image-turbo",
                            "label": "x/z-image-turbo"
                        },
                        {
                            "value": "x/flux2-klein",
                            "label": "x/flux2-klein"
                        }
                    ],
                    "tooltip": "Octree resolution for mesh reconstruction. Higher = more detail but slower and more VRAM."
                },
                {
                    "id": "style",
                    "label": "Style",
                    "type": "select",
                    "default": "vivid",
                    "options": [
                        {
                            "value": "natural",
                            "label": "natural"
                        },
                        {
                            "value": "vivid",
                            "label": "vivid"
                        }
                    ],
                    "tooltip": "The Style of the image"
                },
                {
                    "id": "moderation",
                    "label": "moderation",
                    "type": "select",
                    "default": "low",
                    "options": [
                        {
                            "value": "low",
                            "label": "low"
                        },
                        {
                            "value": "auto",
                            "label": "auto"
                        }
                    ],
                    "tooltip": "The Moderation for the image"
                },
                {
                    "id": "quality",
                    "label": "Quality",
                    "type": "select",
                    "default": "medium",
                    "options": [
                        {
                            "value": "standard",
                            "label": "standard"
                        },
                        {
                            "value": "hd",
                            "label": "hd"
                        },
                        {
                            "value": "low",
                            "label": "low"
                        },
                        {
                            "value": "medium",
                            "label": "medium"
                        },
                        {
                            "value": "auto",
                            "label": "auto"
                        },
                        {
                            "value": "high",
                            "label": "high"
                        }
                    ],
                    "tooltip": "The Quality of the image"
                },
                {
                    "id": "host",
                    "label": "Host:Port",
                    "type": "string",
                    "default": "http://192.168.178.138:11434",
                    "tooltip": "The Host for the image generation"
                },
                {
                    "id": "apiKey",
                    "label": "apiKey",
                    "type": "string",
                    "default": "ollama",
                    "tooltip": "The API Key"
                },
                {
                    "id": "width",
                    "label": "Width",
                    "type": "int",
                    "default": 64,
                    "min": 16,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "The width of the output image"
                },
                {
                    "id": "height",
                    "label": "Height",
                    "type": "int",
                    "default": 64,
                    "min": 16,
                    "max": 1024,
                    "step": 1,
                    "tooltip": "The height of the output image"
                },
        ]