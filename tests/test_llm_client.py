import os
import unittest
from unittest.mock import MagicMock, patch

from tools.llm.llm_client import (
    DEFAULT_OLLAMA_MODEL,
    GeminiClient,
    OllamaClient,
    get_llm_client,
    strip_json_fences,
)


class TestStripJsonFences(unittest.TestCase):
    def test_plain_json(self):
        self.assertEqual(strip_json_fences('{"a": 1}'), '{"a": 1}')

    def test_fenced_json(self):
        raw = '```json\n{"a": 1}\n```'
        self.assertEqual(strip_json_fences(raw), '{"a": 1}')


class TestOllamaClient(unittest.TestCase):
    @patch("tools.llm.llm_client.requests.post")
    def test_generate_parses_response_field(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "hello from gemma"}
        mock_resp.raise_for_status = MagicMock()
        mock_post.return_value = mock_resp

        client = OllamaClient(model="gemma3:4b", base_url="http://localhost:11434")
        self.assertEqual(client.generate("test prompt"), "hello from gemma")
        payload = mock_post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "gemma3:4b")
        self.assertFalse(payload["stream"])


class TestGetLlmClient(unittest.TestCase):
    @patch.dict(os.environ, {"LLM_BACKEND": "ollama", "OLLAMA_MODEL": "gemma3:4b"}, clear=False)
    def test_defaults_to_ollama(self):
        client = get_llm_client()
        self.assertIsInstance(client, OllamaClient)
        self.assertEqual(client.model, DEFAULT_OLLAMA_MODEL)

    @patch.dict(os.environ, {"LLM_BACKEND": "gemini"}, clear=False)
    @patch("tools.llm.llm_client.GeminiClient")
    def test_gemini_backend(self, mock_gemini_cls):
        sentinel = object()
        mock_gemini_cls.return_value = sentinel
        client = get_llm_client()
        self.assertIs(client, sentinel)
        mock_gemini_cls.assert_called_once()


if __name__ == "__main__":
    unittest.main()
