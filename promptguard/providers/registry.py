"""Provider registry for dynamic provider loading."""

from __future__ import annotations

from typing import Any, Dict, Optional, Type

from promptguard.core.exceptions import ProviderNotFoundError
from promptguard.providers.base import LLMProvider


class ProviderRegistry:
    """Registry for dynamically loading and managing LLM providers.

    The registry uses lazy loading to avoid importing provider-specific
    dependencies until they are actually needed. Providers can be
    registered using the `@ProviderRegistry.register()` decorator.

    Example:
        @ProviderRegistry.register("my_provider")
        class MyProvider(LLMProvider):
            ...

        # Get provider instance
        provider = ProviderRegistry.get("my_provider")
    """

    _providers: Dict[str, Type[LLMProvider]] = {}
    _instances: Dict[str, LLMProvider] = {}
    _loaded_modules: set[str] = set()

    @classmethod
    def register(cls, name: str):
        """Decorator to register a provider class.

        Args:
            name: Unique identifier for the provider.

        Returns:
            Decorator function that registers the provider class.

        Example:
            @ProviderRegistry.register("openai")
            class OpenAIProvider(LLMProvider):
                ...
        """
        def decorator(provider_cls: Type[LLMProvider]) -> Type[LLMProvider]:
            cls._providers[name] = provider_cls
            return provider_cls
        return decorator

    @classmethod
    def get(cls, name: str, **kwargs: Any) -> LLMProvider:
        """Get or create a provider instance.

        If the provider has already been instantiated, returns the
        cached instance. Otherwise, creates a new instance with the
        provided kwargs.

        Args:
            name: Provider identifier.
            **kwargs: Arguments to pass to the provider constructor.

        Returns:
            LLMProvider instance.

        Raises:
            ProviderNotFoundError: If the provider is not registered
                and cannot be loaded.
        """
        # Check cache first (only if no kwargs, as kwargs might change config)
        cache_key = name if not kwargs else None
        if cache_key and cache_key in cls._instances:
            return cls._instances[cache_key]

        # Try to load if not registered
        if name not in cls._providers:
            cls._try_load_provider(name)

        if name not in cls._providers:
            raise ProviderNotFoundError(provider=name)

        # Create instance
        instance = cls._providers[name](**kwargs)

        # Cache if no custom kwargs
        if cache_key:
            cls._instances[cache_key] = instance

        return instance

    @classmethod
    def _try_load_provider(cls, name: str) -> None:
        """Attempt to import and register a provider module.

        This enables lazy loading of providers - they are only imported
        when first requested.

        Args:
            name: Provider name to try loading.
        """
        if name in cls._loaded_modules:
            return

        cls._loaded_modules.add(name)

        try:
            if name == "openai":
                from promptguard.providers import openai_provider  # noqa: F401
            elif name == "anthropic":
                from promptguard.providers import anthropic_provider  # noqa: F401
            elif name == "gemini":
                from promptguard.providers import gemini_provider  # noqa: F401
            elif name == "local":
                from promptguard.providers import local_provider  # noqa: F401
        except ImportError:
            # Provider module exists but dependencies not installed
            pass

    @classmethod
    def available_providers(cls) -> list[str]:
        """List all registered providers.

        Note: This only returns providers that have been explicitly
        loaded or registered. Use `all_providers()` to get all
        potentially available providers.

        Returns:
            List of registered provider names.
        """
        return list(cls._providers.keys())

    @classmethod
    def all_providers(cls) -> list[str]:
        """List all potentially available providers.

        Returns provider names for all built-in providers, regardless
        of whether they are currently loadable.

        Returns:
            List of all provider names.
        """
        return ["openai", "anthropic", "gemini", "local"]

    @classmethod
    def is_available(cls, name: str) -> bool:
        """Check if a provider is available (loadable).

        Args:
            name: Provider name to check.

        Returns:
            True if the provider can be loaded.
        """
        if name in cls._providers:
            return True

        try:
            cls._try_load_provider(name)
            return name in cls._providers
        except Exception:
            return False

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the provider instance cache.

        Useful for testing or when provider configuration changes.
        """
        cls._instances.clear()

    @classmethod
    def unregister(cls, name: str) -> None:
        """Remove a provider from the registry.

        Args:
            name: Provider name to unregister.
        """
        cls._providers.pop(name, None)
        cls._instances.pop(name, None)
        cls._loaded_modules.discard(name)
