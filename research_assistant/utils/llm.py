"""
LLM utilities for initializing and managing language models.

Provides abstraction for LLM initialization with support for:
- Ollama (local models)
- Future: OpenAI, Anthropic, custom providers
"""

from typing import Optional, Literal
from langchain_community.llms import Ollama
from langchain.llms.base import BaseLLM

from ..config import settings


# Type aliases
ProviderType = Literal["ollama", "openai", "anthropic"]


def get_llm(
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    provider: ProviderType = "ollama",
    **kwargs
) -> BaseLLM:
    """
    Get configured LLM instance.

    Provides abstraction layer for different LLM providers.
    Currently supports Ollama, designed for future extensibility.

    Args:
        model: Model name (default: from settings)
        temperature: Temperature 0.0-2.0 (default: from settings)
        provider: LLM provider ("ollama", "openai", "anthropic")
        **kwargs: Additional provider-specific arguments

    Returns:
        Configured LLM instance

    Raises:
        ValueError: If provider is not supported
        ConnectionError: If cannot connect to LLM service

    Example:
        >>> # Use default settings
        >>> llm = get_llm()

        >>> # Override model
        >>> llm = get_llm(model="llama3.1:8b")

        >>> # Custom temperature for more creative responses
        >>> llm = get_llm(temperature=0.9)

        >>> # Future: OpenAI support
        >>> llm = get_llm(provider="openai", model="gpt-4")
    """
    # Use settings defaults if not specified
    model = model or settings.ollama_model
    temperature = temperature if temperature is not None else settings.ollama_temperature

    if provider == "ollama":
        return _get_ollama_llm(model, temperature, **kwargs)
    elif provider == "openai":
        raise NotImplementedError("OpenAI support coming in future update")
    elif provider == "anthropic":
        raise NotImplementedError("Anthropic support coming in future update")
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def _get_ollama_llm(
    model: str,
    temperature: float,
    **kwargs
) -> Ollama:
    """
    Get Ollama LLM instance.

    Args:
        model: Ollama model name (e.g., "llama3.2:3b")
        temperature: Temperature for generation
        **kwargs: Additional Ollama arguments

    Returns:
        Configured Ollama instance

    Example:
        >>> llm = _get_ollama_llm("llama3.2:3b", 0.7)
        >>> response = llm.invoke("What is LangGraph?")
    """
    return Ollama(
        model=model,
        temperature=temperature,
        base_url=settings.ollama_base_url,
        **kwargs
    )


def test_llm_connection(llm: Optional[BaseLLM] = None) -> bool:
    """
    Test if LLM is accessible and working.

    Args:
        llm: LLM instance to test (default: creates new instance)

    Returns:
        True if connection successful, False otherwise

    Example:
        >>> if test_llm_connection():
        ...     print("LLM is ready!")
        ... else:
        ...     print("LLM connection failed")
    """
    try:
        if llm is None:
            llm = get_llm()

        # Simple test prompt
        response = llm.invoke("Say 'OK' if you can read this.")

        # Check if we got a response
        return len(response.strip()) > 0

    except Exception as e:
        print(f"LLM connection test failed: {e}")
        return False


def get_llm_info(llm: Optional[BaseLLM] = None) -> dict:
    """
    Get information about the LLM configuration.

    Args:
        llm: LLM instance (default: creates new instance)

    Returns:
        Dictionary with LLM configuration details

    Example:
        >>> info = get_llm_info()
        >>> print(f"Using model: {info['model']}")
    """
    if llm is None:
        llm = get_llm()

    info = {
        "provider": "ollama",  # Currently only Ollama
        "model": settings.ollama_model,
        "temperature": settings.ollama_temperature,
        "base_url": settings.ollama_base_url,
    }

    # Add Ollama-specific info if available
    if isinstance(llm, Ollama):
        info.update({
            "ollama_model": llm.model,
            "ollama_base_url": llm.base_url,
        })

    return info


# Future extension point for multiple providers
class LLMFactory:
    """
    Factory for creating LLM instances with different providers.

    Future implementation will support:
    - Ollama (local models)
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude)
    - Custom providers
    """

    @staticmethod
    def create(
        provider: ProviderType = "ollama",
        model: Optional[str] = None,
        **kwargs
    ) -> BaseLLM:
        """
        Create LLM instance using factory pattern.

        Args:
            provider: Provider type
            model: Model name
            **kwargs: Provider-specific arguments

        Returns:
            Configured LLM instance
        """
        return get_llm(model=model, provider=provider, **kwargs)

    @staticmethod
    def create_ollama(model: Optional[str] = None, **kwargs) -> Ollama:
        """Create Ollama instance"""
        return get_llm(model=model, provider="ollama", **kwargs)

    # Future methods:
    # @staticmethod
    # def create_openai(model: str = "gpt-3.5-turbo", **kwargs) -> ChatOpenAI:
    #     """Create OpenAI instance"""
    #     from langchain_openai import ChatOpenAI
    #     return ChatOpenAI(model=model, **kwargs)
    #
    # @staticmethod
    # def create_anthropic(model: str = "claude-3-sonnet", **kwargs) -> ChatAnthropic:
    #     """Create Anthropic instance"""
    #     from langchain_anthropic import ChatAnthropic
    #     return ChatAnthropic(model=model, **kwargs)
