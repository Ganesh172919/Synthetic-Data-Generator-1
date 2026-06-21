"""
Provider Factory
================
Registry and factory for creating provider instances by name.
"""

import os
from typing import Dict, List, Optional, Type

from .base import BaseProvider, ProviderError


# Global registry: name -> provider class
_registry: Dict[str, Type[BaseProvider]] = {}


def register_provider(name: str):
    """Decorator to register a provider class by name."""
    def decorator(cls: Type[BaseProvider]):
        _registry[name.lower()] = cls
        return cls
    return decorator


def get_provider(name: str, **kwargs) -> BaseProvider:
    """
    Create a provider instance by name.

    Args:
        name: Provider identifier (e.g., 'openai', 'mock').
        **kwargs: Additional arguments passed to the provider constructor.

    Returns:
        Instantiated provider.

    Raises:
        ProviderError: If provider name is not registered.
    """
    key = name.lower().strip()
    if key not in _registry:
        available = ', '.join(sorted(_registry.keys()))
        raise ProviderError(
            f"Unknown provider '{name}'. Available: {available}",
            provider=name,
        )
    return _registry[key](**kwargs)


def list_providers() -> List[str]:
    """Return sorted list of registered provider names."""
    return sorted(_registry.keys())


def get_provider_class(name: str) -> Optional[Type[BaseProvider]]:
    """Get the provider class without instantiating it."""
    return _registry.get(name.lower().strip())


# ---------------------------------------------------------------------------
# Auto-import all provider modules to trigger registration
# ---------------------------------------------------------------------------

def _auto_register():
    """Import all provider modules in this package to trigger @register_provider."""
    import importlib
    import pkgutil
    import providers as pkg

    for _importer, modname, _ispkg in pkgutil.iter_modules(pkg.__path__):
        if modname not in ('base', 'factory', '__init__'):
            try:
                importlib.import_module(f'.{modname}', package='providers')
            except ImportError:
                # Provider may have missing dependencies (e.g., anthropic SDK)
                pass


# Run auto-registration on first import
try:
    _auto_register()
except Exception:
    # If running outside the providers package context, registration
    # must be done manually or via explicit imports
    pass
