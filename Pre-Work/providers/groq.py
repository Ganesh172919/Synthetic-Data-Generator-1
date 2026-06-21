"""
Groq Provider
=============
Uses the Groq API for ultra-fast inference.
Requires GROQ_API_KEY environment variable.
"""

import os
import time
from typing import List

from .base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderError,
)
from .factory import register_provider


@register_provider('groq')
class GroqProvider(BaseProvider):

    AVAILABLE_MODELS = [
        'llama-3.1-70b-versatile',
        'llama-3.1-8b-instant',
        'mixtral-8x7b-32768',
        'gemma2-9b-it',
    ]

    def __init__(self):
        self._api_key = os.environ.get('GROQ_API_KEY', '')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(api_key=self._api_key)
            except ImportError:
                raise ProviderError(
                    "groq package not installed. Run: pip install groq",
                    provider='groq',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'groq'

    @property
    def requires_api_key(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._api_key:
            raise ProviderAuthError("GROQ_API_KEY not set", provider='groq')

        client = self._get_client()
        model = request.model or 'llama-3.1-70b-versatile'

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
            error_msg = str(exc).lower()
            if 'rate' in error_msg or '429' in error_msg:
                raise ProviderRateLimitError(str(exc), provider='groq') from exc
            if 'auth' in error_msg or '401' in error_msg:
                raise ProviderAuthError(str(exc), provider='groq') from exc
            raise ProviderError(str(exc), provider='groq') from exc

    def health_check(self) -> ProviderHealth:
        if not self._api_key:
            return ProviderHealth(
                status='unconfigured',
                message='GROQ_API_KEY not set',
            )

        try:
            start = time.time()
            client = self._get_client()
            client.models.list()
            latency = (time.time() - start) * 1000
            return ProviderHealth(
                status='healthy',
                latency_ms=round(latency, 1),
                message='Groq API accessible',
                models_available=self.AVAILABLE_MODELS,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=str(exc)[:200],
            )

    def get_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
