"""
LLM Provider Abstraction Layer
===============================
Provides a unified interface for multiple LLM providers.

Usage:
    from providers.factory import get_provider
    from providers.base import GenerationRequest

    provider = get_provider('openai')
    response = provider.generate(GenerationRequest(prompt="Hello"))
    print(response.text)
"""

from .base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
    ProviderError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from .factory import get_provider, list_providers, register_provider

__all__ = [
    'BaseProvider',
    'GenerationRequest',
    'GenerationResponse',
    'ProviderHealth',
    'ProviderError',
    'ProviderAuthError',
    'ProviderRateLimitError',
    'ProviderTimeoutError',
    'get_provider',
    'list_providers',
    'register_provider',
]
