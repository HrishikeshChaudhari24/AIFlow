"""Talk to a local model. Ollama for now; anything with the same `extract` method can plug in."""

import base64
import json
import time

import httpx

from .task import Task


class OllamaModel:
    def __init__(self, name: str, host: str = "http://localhost:11434"):
        self.name = name
        self.client = httpx.Client(base_url=host, timeout=600)

    def extract(self, task: Task, image_path) -> tuple[dict, float]:
        image = base64.b64encode(open(image_path, "rb").read()).decode()
        start = time.perf_counter()
        r = self.client.post("/api/chat", json={
            "model": self.name,
            "messages": [{"role": "user", "content": task.prompt(), "images": [image]}],
            "format": task.json_schema(),
            "stream": False,
            "options": {"temperature": 0},
        })
        r.raise_for_status()
        seconds = time.perf_counter() - start
        try:
            return json.loads(r.json()["message"]["content"]), seconds
        except json.JSONDecodeError:
            return {}, seconds
