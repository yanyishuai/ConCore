"""Unified LLM client supporting local Gemma 3 via Ollama or cloud Gemini."""

from __future__ import annotations

import json
import os
from typing import Iterator, Union

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

DEFAULT_OLLAMA_MODEL = "gemma3:4b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"


def strip_json_fences(text: str) -> str:
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()
    return cleaned


class OllamaClient:
    """Local inference via Ollama (Gemma 3 4B recommended)."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        timeout: int = 120,
    ):
        self.model = model or os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        self.base_url = (base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.timeout = int(os.getenv("OLLAMA_TIMEOUT", timeout))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {"num_predict": max_tokens, "temperature": temperature},
        }

        if stream:
            return self._stream(url, payload)
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict):
            return data.get("response") or json.dumps(data)
        return str(data)

    def _stream(self, url: str, payload: dict) -> Iterator[str]:
        with requests.post(url, json=payload, stream=True, timeout=self.timeout) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    chunk = data.get("response", "")
                    if chunk:
                        yield chunk
                except json.JSONDecodeError:
                    yield line


class GeminiClient:
    """Cloud Gemini fallback when LLM_BACKEND=gemini."""

    def __init__(self, model: str | None = None):
        import google.generativeai as genai

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY required when LLM_BACKEND=gemini")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model_name = model or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stream: bool = False,
    ) -> Union[str, Iterator[str]]:
        if stream:
            raise NotImplementedError("Gemini streaming not implemented in this client")
        model = self._genai.GenerativeModel(self.model_name)
        response = model.generate_content(
            prompt,
            generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
        )
        return getattr(response, "text", str(response)).strip()


def get_llm_client(model: str | None = None) -> OllamaClient | GeminiClient:
    backend = os.getenv("LLM_BACKEND", "ollama").lower()
    if backend == "gemini":
        return GeminiClient(model)
    return OllamaClient(model=model)


def generate_text(
    prompt: str,
    max_tokens: int = 2048,
    temperature: float = 0.0,
    model: str | None = None,
) -> str:
    client = get_llm_client(model)
    result = client.generate(
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
    )
    return str(result).strip()
