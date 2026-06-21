"""
Replicate Provider
==================
Uses the Replicate API for running open-source models in the cloud.
Requires REPLICATE_API_TOKEN environment variable.
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


@register_provider('replicate')
class ReplicateProvider(BaseProvider):

    AVAILABLE_MODELS = [
        'meta/llama-2-70b-chat',
        'mistralai/mixtral-8x7b-instruct-v0.1',
        'meta/llama-3-70b-instruct',
        'google-deepmind/gemma-2-27b-it',
    ]

    def __init__(self):
        self._token = os.environ.get('REPLICATE_API_TOKEN', '')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import replicate
                self._client = replicate
            except ImportError:
                raise ProviderError(
                    "replicate package not installed. Run: pip install replicate",
                    provider='replicate',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'replicate'

    @property
    def requires_api_key(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._token:
            raise ProviderAuthError("REPLICATE_API_TOKEN not set", provider='replicate')

        client = self._get_client()
        model_id = request.model or 'meta/llama-2-70b-chat'

        try:
            # Build input based on model
            input_data = {
                "prompt": request.prompt,
                "max_new_tokens": request.max_new_tokens,
                "temperature": request.temperature,
            }
            if request.system_prompt:
                input_data["system_prompt"] = request.system_prompt

            output = client.run(
                f"{model_id}" if ':' in model_id else f"{model_id}:latest",
                input=input_data,
            )

            # Replicate returns a list of strings or a generator
            if hasattr(output, '__iter__') and not isinstance(output, str):
                text = ''.join(str(chunk) for chunk in output)
            else:
                text = str(output)

            return GenerationResponse(
                text=text.strip(),
                tokens_used=len(text.split()),
                model=model_id,
                finish_reason='stop',
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if 'rate' in error_msg or '429' in error_msg or 'throttl' in error_msg:
                raise ProviderRateLimitError(str(exc), provider='replicate') from exc
            if 'auth' in error_msg or '401' in error_msg or '403' in error_msg or 'token' in error_msg:
                raise ProviderAuthError(str(exc), provider='replicate') from exc
            raise ProviderError(str(exc), provider='replicate') from exc

    def health_check(self) -> ProviderHealth:
        if not self._token:
            return ProviderHealth(
                status='unconfigured',
                message='REPLICATE_API_TOKEN not set',
            )

        try:
            client = self._get_client()
            # Lightweight check — list models
            models = list(client.models.list())
            return ProviderHealth(
                status='healthy',
                message='Replicate API accessible',
                models_available=self.AVAILABLE_MODELS,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=str(exc)[:200],
            )

    def get_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
