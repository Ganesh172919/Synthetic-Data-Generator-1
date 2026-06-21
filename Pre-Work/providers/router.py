"""
Smart Model Router
==================
Automatically selects the best provider based on:
- Dataset size (small → fast/cheap, large → efficient)
- Domain complexity (simple Q&A → smaller model, complex reasoning → larger)
- Provider availability and configuration
- User budget constraints
"""

import os
from typing import Optional, List, Tuple

from .base import ProviderHealth, ProviderError
from .factory import get_provider, list_providers


# Provider tiers: (speed, quality, cost)
# speed: 1=slow, 5=fast
# quality: 1=low, 5=excellent
# cost: 1=free, 5=expensive
PROVIDER_TIERS = {
    'mock':            {'speed': 5, 'quality': 1, 'cost': 1, 'tier': 'test'},
    'groq':            {'speed': 5, 'quality': 3, 'cost': 2, 'tier': 'fast'},
    'ollama':          {'speed': 3, 'quality': 3, 'cost': 1, 'tier': 'local'},
    'huggingface':     {'speed': 2, 'quality': 3, 'cost': 1, 'tier': 'local'},
    'openai':          {'speed': 4, 'quality': 5, 'cost': 4, 'tier': 'premium'},
    'anthropic':       {'speed': 4, 'quality': 5, 'cost': 4, 'tier': 'premium'},
    'google':          {'speed': 4, 'quality': 5, 'cost': 3, 'tier': 'premium'},
    'azure_openai':    {'speed': 4, 'quality': 5, 'cost': 4, 'tier': 'premium'},
    'together':        {'speed': 3, 'quality': 3, 'cost': 2, 'tier': 'budget'},
    'aws_bedrock':     {'speed': 3, 'quality': 5, 'cost': 4, 'tier': 'enterprise'},
    'replicate':       {'speed': 3, 'quality': 3, 'cost': 2, 'tier': 'budget'},
    'custom':          {'speed': 3, 'quality': 3, 'cost': 2, 'tier': 'custom'},
}

# Domain complexity hints
DOMAIN_COMPLEXITY = {
    'financial': 3,
    'healthcare': 4,
    'legal': 4,
    'technology': 3,
    'science': 4,
    'education': 2,
    'customer_support': 2,
    'ecommerce': 2,
    'realestate': 2,
    'gaming': 2,
    'marketing': 2,
    'hr': 2,
    'news': 3,
    'cybersecurity': 4,
    'travel': 2,
    'food': 1,
    'custom': 3,
}

# Parse mode complexity
PARSE_MODE_COMPLEXITY = {
    'qa': 2,
    'text': 2,
    'json': 3,
    'instruction': 3,
    'conversation': 3,
    'classification': 3,
    'ner': 4,
    'summarization': 3,
    'translation': 3,
    'code': 4,
    'reasoning': 5,
}


def _is_provider_configured(provider_name: str) -> bool:
    """Check if a provider has its required env vars set."""
    required_env = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'google': 'GOOGLE_API_KEY',
        'groq': 'GROQ_API_KEY',
        'together': 'TOGETHER_API_KEY',
        'azure_openai': 'AZURE_OPENAI_API_KEY',
        'aws_bedrock': 'AWS_REGION',
        'replicate': 'REPLICATE_API_TOKEN',
        'custom': 'CUSTOM_API_BASE',
    }

    if provider_name == 'mock':
        return True
    if provider_name in ('ollama', 'huggingface'):
        return True  # No key required, always "available"

    env_var = required_env.get(provider_name)
    if not env_var:
        return False
    return bool(os.environ.get(env_var))


def _score_provider(
    provider_name: str,
    dataset_size: int,
    complexity: int,
    prefer: str = 'balanced',
) -> float:
    """
    Score a provider for a given task. Higher is better.

    Args:
        provider_name: Provider identifier.
        dataset_size: Number of records to generate.
        complexity: Task complexity 1-5.
        prefer: 'speed', 'quality', 'cost', or 'balanced'.
    """
    tiers = PROVIDER_TIERS.get(provider_name)
    if not tiers:
        return 0

    speed = tiers['speed']
    quality = tiers['quality']
    cost = tiers['cost']

    # Base score
    if prefer == 'speed':
        score = speed * 3 + quality * 1
    elif prefer == 'quality':
        score = speed * 1 + quality * 3
    elif prefer == 'cost':
        score = speed * 1 + (5 - cost) * 3
    else:  # balanced
        score = speed * 2 + quality * 2 + (5 - cost) * 1

    # Penalty for mismatched complexity
    # High complexity tasks need quality; low complexity tasks waste money on premium
    if complexity >= 4 and quality < 3:
        score -= 10  # Strong penalty: low quality for complex task
    elif complexity <= 2 and cost >= 4:
        score -= 5   # Mild penalty: expensive model for simple task

    # Bonus for large dataset efficiency
    if dataset_size > 10000:
        if speed >= 4:
            score += 3  # Fast providers are better for large datasets
        if cost <= 2:
            score += 2  # Cheap providers save money on large runs

    return score


def select_provider(
    dataset_size: int = 1000,
    domain: str = 'custom',
    parse_mode: str = 'qa',
    prefer: str = 'balanced',
    exclude: Optional[List[str]] = None,
) -> Tuple[str, str]:
    """
    Select the best provider for a given task.

    Args:
        dataset_size: Number of records to generate.
        domain: Domain category (affects complexity).
        parse_mode: Parse mode (affects complexity).
        prefer: 'speed', 'quality', 'cost', or 'balanced'.
        exclude: Provider names to exclude.

    Returns:
        Tuple of (provider_name, reason_string).

    Raises:
        ProviderError: If no configured provider can be found.
    """
    exclude_set = set(exclude or [])
    exclude_set.add('mock')  # Never auto-select mock

    domain_complexity = DOMAIN_COMPLEXITY.get(domain, 3)
    mode_complexity = PARSE_MODE_COMPLEXITY.get(parse_mode, 2)
    avg_complexity = (domain_complexity + mode_complexity) / 2

    candidates = []
    for name in list_providers():
        if name in exclude_set:
            continue
        if not _is_provider_configured(name):
            continue
        score = _score_provider(name, dataset_size, avg_complexity, prefer)
        candidates.append((name, score))

    if not candidates:
        # Fall back to mock if nothing else is available
        return 'mock', 'No configured providers found, falling back to mock'

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = candidates[0]

    reason_parts = [f'score={best_score:.0f}']
    tiers = PROVIDER_TIERS.get(best_name, {})
    reason_parts.append(f"speed={tiers.get('speed', '?')}, quality={tiers.get('quality', '?')}")
    reason_parts.append(f"complexity={avg_complexity:.1f}, size={dataset_size}")

    return best_name, f"Auto-selected {best_name} ({', '.join(reason_parts)})"


def get_routing_info(
    dataset_size: int = 1000,
    domain: str = 'custom',
    parse_mode: str = 'qa',
) -> dict:
    """
    Get detailed routing information for all providers.

    Returns a dict with scoring details for each configured provider.
    """
    domain_complexity = DOMAIN_COMPLEXITY.get(domain, 3)
    mode_complexity = PARSE_MODE_COMPLEXITY.get(parse_mode, 2)
    avg_complexity = (domain_complexity + mode_complexity) / 2

    results = {}
    for name in list_providers():
        configured = _is_provider_configured(name)
        if name == 'mock' or not configured:
            continue

        for pref in ['speed', 'quality', 'cost', 'balanced']:
            score = _score_provider(name, dataset_size, avg_complexity, pref)
            key = f"{name}:{pref}"
            results[key] = {
                'provider': name,
                'preference': pref,
                'score': round(score, 1),
                'configured': configured,
                'tiers': PROVIDER_TIERS.get(name, {}),
            }

    return results
