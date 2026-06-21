"""
Google Gemini Provider
======================
Uses the Google Generative AI API.
Requires GOOGLE_API_KEY environment variable.
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


@register_provider('google')
class GoogleProvider(BaseProvider):

    AVAILABLE_MODELS = [
        'gemini-2.5-pro',
        'gemini-2.5-flash',
        'gemini-2.0-flash',
    ]

    def __init__(self):
        self._api_key = os.environ.get('GOOGLE_API_KEY', '')
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self._api_key)
            except ImportError:
                raise ProviderError(
                    "google-genai package not installed. Run: pip install google-genai",
                    provider='google',
                )
        return self._client

    @property
    def name(self) -> str:
        return 'google'

    @property
    def requires_api_key(self) -> bool:
        return True

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        if not self._api_key:
            raise ProviderAuthError("GOOGLE_API_KEY not set", provider='google')

        client = self._get_client()
        model = request.model or 'gemini-2.0-flash'

        try:
            from google.genai import types

            contents = request.prompt
            if request.system_prompt:
                contents = f"{request.system_prompt}\n\n{request.prompt}"

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=request.temperature,
                    max_output_tokens=request.max_new_tokens,
                ),
            )

            text = response.text or ""
            tokens = 0
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                tokens = getattr(response.usage_metadata, 'total_token_count', 0) or 0

            return GenerationResponse(
                text=text,
                tokens_used=tokens,
                model=model,
                finish_reason='stop',
            )
        except Exception as exc:
            error_msg = str(exc).lower()
            if 'rate' in error_msg or '429' in error_msg:
                raise ProviderRateLimitError(str(exc), provider='google') from exc
            if 'api_key' in error_msg or 'auth' in error_msg or '401' in error_msg:
                raise ProviderAuthError(str(exc), provider='google') from exc
            raise ProviderError(str(exc), provider='google') from exc

    def health_check(self) -> ProviderHealth:
        if not self._api_key:
            return ProviderHealth(
                status='unconfigured',
                message='GOOGLE_API_KEY not set',
            )

        try:
            start = time.time()
            client = self._get_client()
            response = client.models.generate_content(
                model='gemini-2.0-flash',
                contents='Hi',
            )
            latency = (time.time() - start) * 1000
            return ProviderHealth(
                status='healthy',
                latency_ms=round(latency, 1),
                message='Google Gemini API accessible',
                models_available=self.AVAILABLE_MODELS,
            )
        except Exception as exc:
            return ProviderHealth(
                status='unavailable',
                message=str(exc)[:200],
            )

    def get_models(self) -> List[str]:
        return self.AVAILABLE_MODELS
