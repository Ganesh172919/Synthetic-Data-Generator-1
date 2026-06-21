"""
Ollama Provider
===============
Uses a local Ollama server for model inference.
No API key needed — just a running Ollama instance.
"""

import os
import time
from typing import List

from .base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
    ProviderError,
)
from .factory import register_provider


@register_provider('ollama')
class OllamaProvider(BaseProvider):

    def __init__(self):
        self._host = os.environ.get('OLLAMA_HOST', 'http://localhost:11434')

    @property
    def name(self) -> str:
        return 'ollama'

    @property
    def requires_api_key(self) -> bool:
        return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        try:
            import httpx
        except ImportError:
            # Fall back to urllib
            return self._generate_urllib(request)

        model = request.model or 'llama3'

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            with httpx.Client(timeout=300) as client:
                response = client.post(
                    f"{self._host}/api/chat",
                    json={
                        "model": model,
                        "messages": messages,
                        "stream": False,
                        "options": {
                            "temperature": request.temperature,
                            "num_predict": request.max_new_tokens,
                        },
                    },
                )
                response.raise_for_status()
                data = response.json()

            text = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

            return GenerationResponse(
                text=text,
                tokens_used=tokens,
                model=model,
                finish_reason='stop',
                raw=data,
            )
        except Exception as exc:
            raise ProviderError(str(exc), provider='ollama') from exc

    def _generate_urllib(self, request: GenerationRequest) -> GenerationResponse:
        """Fallback using stdlib urllib when httpx is not available."""
        import json
        import urllib.request
        import urllib.error

        model = request.model or 'llama3'

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        payload = json.dumps({
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
                "num_predict": request.max_new_tokens,
            },
        }).encode()

        try:
            req = urllib.request.Request(
                f"{self._host}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.loads(resp.read())

            text = data.get("message", {}).get("content", "")
            tokens = data.get("eval_count", 0) + data.get("prompt_eval_count", 0)

            return GenerationResponse(
                text=text,
                tokens_used=tokens,
                model=model,
                finish_reason='stop',
                raw=data,
            )
        except Exception as exc:
            raise ProviderError(str(exc), provider='ollama') from exc

    def health_check(self) -> ProviderHealth:
        try:
            import urllib.request
            import json

            req = urllib.request.Request(f"{self._host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())

            models = [m.get("name", "") for m in data.get("models", [])]
            return ProviderHealth(
                status='healthy',
                message=f'Ollama server running at {self._host}',
                models_available=models,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=f'Cannot reach Ollama at {self._host}: {exc}',
            )

    def get_models(self) -> List[str]:
        try:
            import urllib.request
            import json

            req = urllib.request.Request(f"{self._host}/api/tags")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read())
            return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            return ['llama3', 'mistral', 'codellama']
