"""
Mock Provider
=============
Deterministic test provider that generates fake data without any API calls.
Useful for testing, CI, and pipeline development.
"""

import hashlib
import time
from typing import List

from .base import (
    BaseProvider,
    GenerationRequest,
    GenerationResponse,
    ProviderHealth,
)
from .factory import register_provider


@register_provider('mock')
class MockProvider(BaseProvider):

    @property
    def name(self) -> str:
        return 'mock'

    @property
    def requires_api_key(self) -> bool:
        return False

    def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Generate deterministic content based on prompt hash
        prompt_hash = hashlib.md5(request.prompt.encode()).hexdigest()[:8]

        text = (
            f"Q: What is synthetic data generation? (batch {prompt_hash})\n"
            f"A: Synthetic data generation is the process of creating artificial datasets "
            f"that mimic real-world data patterns using machine learning models. "
            f"This is useful for training AI systems when real data is scarce or sensitive.\n"
            f"---SAMPLE---\n"
            f"Q: How does the mock provider work?\n"
            f"A: The mock provider generates deterministic dummy content for testing "
            f"purposes. It requires no API keys and produces consistent results "
            f"based on the input prompt hash ({prompt_hash}).\n"
            f"---SAMPLE---\n"
            f"Q: Why use synthetic data?\n"
            f"A: Synthetic data helps overcome data privacy concerns, augments small "
            f"datasets, and enables rapid prototyping of ML models without waiting "
            f"for real data collection.\n"
        )

        return GenerationResponse(
            text=text,
            tokens_used=len(text.split()),
            model='mock',
            finish_reason='stop',
        )

    def health_check(self) -> ProviderHealth:
        return ProviderHealth(
            status='healthy',
            latency_ms=0,
            message='Mock provider is always available',
            models_available=['mock'],
        )

    def get_models(self) -> List[str]:
        return ['mock']
