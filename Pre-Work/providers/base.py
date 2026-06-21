"""
Base Provider Abstraction
=========================
Defines the interface that all LLM providers must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GenerationRequest:
    """Standardized generation request sent to any provider."""
    prompt: str
    max_new_tokens: int = 2000
    temperature: float = 0.8
    model: Optional[str] = None
    stop_sequences: Optional[List[str]] = None
    system_prompt: Optional[str] = None


@dataclass
class GenerationResponse:
    """Standardized generation response from any provider."""
    text: str
    tokens_used: int = 0
    model: str = ""
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None


@dataclass
class ProviderHealth:
    """Health check result for a provider."""
    status: str  # "healthy", "degraded", "unavailable", "unconfigured"
    latency_ms: float = 0
    message: str = ""
    models_available: List[str] = field(default_factory=list)


class BaseProvider(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider must implement:
    - generate(): Send a prompt and get a text response
    - health_check(): Report provider availability
    - get_models(): List available models

    Providers should handle their own retry logic, rate limiting,
    and error translation into standard exceptions.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider identifier (e.g., 'openai', 'anthropic')."""
        ...

    @property
    @abstractmethod
    def requires_api_key(self) -> bool:
        """Whether this provider needs an API key to function."""
        ...

    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """
        Generate text from a prompt.

        Raises:
            ProviderError: On any generation failure.
            ProviderAuthError: On authentication failure.
            ProviderRateLimitError: On rate limit exceeded.
        """
        ...

    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """Check if the provider is available and responsive."""
        ...

    @abstractmethod
    def get_models(self) -> List[str]:
        """Return list of available model identifiers."""
        ...

    def generate_batch(self, requests: List[GenerationRequest]) -> List[GenerationResponse]:
        """
        Generate text for multiple prompts. Default implementation calls generate()
        sequentially. Providers can override for parallel/batch processing.
        """
        return [self.generate(req) for req in requests]


# ---------------------------------------------------------------------------
# Provider Exceptions
# ---------------------------------------------------------------------------

class ProviderError(Exception):
    """Base exception for provider errors."""
    def __init__(self, message: str, provider: str = "", retryable: bool = False):
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class ProviderAuthError(ProviderError):
    """Authentication/authorization failure."""
    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider=provider, retryable=False)


class ProviderRateLimitError(ProviderError):
    """Rate limit exceeded."""
    def __init__(self, message: str, provider: str = "", retry_after: float = 0):
        super().__init__(message, provider=provider, retryable=True)
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Request timed out."""
    def __init__(self, message: str, provider: str = ""):
        super().__init__(message, provider=provider, retryable=True)
