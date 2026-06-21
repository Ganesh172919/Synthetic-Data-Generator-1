"""
Anthropic Provider
==================
Uses the Anthropic Claude API.
Requires ANTHROPIC_API_KEY environment variable.
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


@register_provider('anthropic')
class AnthropicProvider(BaseProvider):

    AVAILABLE_MODELS = [
        'claude-sonnet-4-20250514',
        'claude-haiku-4-5-20251001',
    ]

    def __init__(self):
        self._api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=self._api_key)
            except ImportError:
                raise ProviderError(
                    "anthropic package not installed. Run: pip install anthropic",
                    provider='anthropic',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'anthropic'

    @property
    def requires_api_key(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._api_key:
            raise ProviderAuthError("ANTHROPIC_API_KEY not set", provider='anthropic')

        client = self._get_client()
        model = request.model or 'claude-haiku-4-5-20251001'

        try:
            kwargs = {
                "model": model,
                "max_tokens": request.max_new_tokens,
                "temperature": request.temperature,
                "messages": [{"role": "user", "content": request.prompt}],
            }
            if request.system_prompt:
                kwargs["system"] = request.system_prompt
            if request.stop_sequences:
                kwargs["stop_sequences"] = request.stop_sequences

            response = client.messages.create(**kwargs)

            text = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    text += block.text

            tokens = (response.usage.input_tokens + response.usage.output_tokens) if response.usage else 0

            return GenerationResponse(
                text=text,
                tokens_used=tokens,
                model=response.model,
                finish_reason=response.stop_reason or 'stop',
                raw={'id': response.id},
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if 'rate' in error_msg or '429' in error_msg:
                raise ProviderRateLimitError(str(exc), provider='anthropic') from exc
            if 'auth' in error_msg or '401' in error_msg or '403' in error_msg:
                raise ProviderAuthError(str(exc), provider='anthropic') from exc
            raise ProviderError(str(exc), provider='anthropic') from exc

    def health_check(self) -> ProviderHealth:
        if not self._api_key:
            return ProviderHealth(
                status='unconfigured',
                message='ANTHROPIC_API_KEY not set',
            )

        try:
            start = time.time()
            client = self._get_client()
            # Lightweight test — list models or a minimal request
            response = client.messages.create(
                model='claude-haiku-4-5-20251001',
                max_tokens=10,
                messages=[{"role": "user", "content": "Hi"}],
            )
            latency = (time.time() - start) * 1000
            return ProviderHealth(
                status='healthy',
                latency_ms=round(latency, 1),
                message='Anthropic API accessible',
                models_available=self.AVAILABLE_MODELS,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=str(exc)[:200],
            )

    def get_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
