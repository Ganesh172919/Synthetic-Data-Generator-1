"""
Azure OpenAI Provider
=====================
Uses Azure-hosted OpenAI models.
Requires AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT.
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


@register_provider('azure_openai')
class AzureOpenAIProvider(BaseProvider):

    AVAILABLE_MODELS = ['gpt-4o', 'gpt-4o-mini', 'gpt-4', 'gpt-35-turbo']

    def __init__(self):
        self._api_key = os.environ.get('AZURE_OPENAI_API_KEY', '')
        self._endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
        self._api_version = os.environ.get('AZURE_OPENAI_API_VERSION', '2024-02-15-preview')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from openai import AzureOpenAI
                self._client = AzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=self._endpoint,
                    api_version=self._api_version,
                )
            except ImportError:
                raise ProviderError(
                    "openai package not installed. Run: pip install openai",
                    provider='azure_openai',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'azure_openai'

    @property
    def requires_api_key(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._api_key or not self._endpoint:
            raise ProviderAuthError(
                "AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set",
                provider='azure_openai',
            )

        client = self._get_client()
        model = request.model or 'gpt-4o-mini'

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
                raise ProviderRateLimitError(str(exc), provider='azure_openai') from exc
            if 'auth' in error_msg or '401' in error_msg:
                raise ProviderAuthError(str(exc), provider='azure_openai') from exc
            raise ProviderError(str(exc), provider='azure_openai') from exc

    def health_check(self) -> ProviderHealth:
        if not self._api_key or not self._endpoint:
            return ProviderHealth(
                status='unconfigured',
                message='AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT must be set',
            )

        try:
            start = time.time()
            client = self._get_client()
            client.models.list()
            latency = (time.time() - start) * 1000
            return ProviderHealth(
                status='healthy',
                latency_ms=round(latency, 1),
                message='Azure OpenAI API accessible',
                models_available=self.AVAILABLE_MODELS,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=str(exc)[:200],
            )

    def get_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
