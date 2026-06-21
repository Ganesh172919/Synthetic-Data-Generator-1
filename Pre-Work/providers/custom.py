"""
Custom Endpoint Provider
========================
Works with any OpenAI-compatible API endpoint:
- vLLM
- text-generation-inference (TGI)
- llama.cpp server
- LocalAI
- Any other OpenAI-compatible server

Configure via CUSTOM_API_BASE environment variable.
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


@register_provider('custom')
class CustomEndpointProvider(BaseProvider):

    def __init__(self):
        self._api_base = os.environ.get('CUSTOM_API_BASE', 'http://localhost:8080/v1')
        self._api_key = os.environ.get('CUSTOM_API_KEY', 'no-key')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self._api_key,
                    base_url=self._api_base,
                )
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Run: pip install openai",
                    provider='custom',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'custom'

    @property
    def requires_api_key(self) -> bool:
        return False  # Many local endpoints don't need keys

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        client = self._get_client()
        model = request.model or 'default'

        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})

        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": request.max_new_tokens,
                "temperature": request.temperature,
            }
            if request.stop_sequences:
                kwargs["stop"] = request.stop_sequences

            response = client.chat.completions.create(**kwargs)

            text = response.choices[0].message.content or ""
            tokens = (response.usage.total_tokens if response.usage else 0)

            return GenerationResponse(
                text=text,
                tokens_used=tokens,
                model=response.model,
                finish_reason=response.choices[0].finish_reason or 'stop',
            )
        except Exception as exc:
            raise ProviderError(str(exc), provider='custom') from exc

    def health_check(self) -> ProviderHealth:
        try:
            start = time.time()
            client = self._get_client()
            models = client.models.list()
            latency = (time.time() - start) * 1000
            model_ids = [m.id for m in models.data] if hasattr(models, 'data') else []
            return ProviderHealth(
                status='healthy',
                latency_ms=round(latency, 1),
                message=f'Custom endpoint at {self._api_base} accessible',
                models_available=model_ids,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=f'Cannot reach {self._api_base}: {exc}',
            )

    def get_models(self) -> List[str]:
        try:
            client = self._get_client()
            models = client.models.list()
            return [m.id for m in models.data] if hasattr(models, 'data') else ['default']
        except Exception:
            return ['default']
