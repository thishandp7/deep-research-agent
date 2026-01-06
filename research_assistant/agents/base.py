"""
Base agent class for research assistant.

Provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from langchain_core.language_models import BaseLLM

from ..utils.llm import get_llm


class BaseAgent(ABC):
    """
    Abstract base class for all agents.

    Provides common initialization, LLM access, and error handling.
    """

    def __init__(
        self,
        llm: Optional[BaseLLM] = None,
        temperature: Optional[float] = None,
        verbose: bool = False,
    ):
        """
        Initialize base agent.

        Args:
            llm: Language model instance (defaults to settings)
            temperature: LLM temperature (defaults to settings)
            verbose: Enable verbose logging
        """
        self.llm = llm or get_llm(temperature=temperature)
        self.verbose = verbose
        self.name = self.__class__.__name__

    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the agent's primary function.

        Args:
            **kwargs: Agent-specific parameters

        Returns:
            Dictionary with results and any errors

        Raises:
            NotImplementedError: Must be implemented by subclasses
        """
        raise NotImplementedError("Subclasses must implement run()")

    def log(self, message: str, level: str = "INFO") -> None:
        """
        Log message if verbose mode is enabled.

        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        if self.verbose:
            print(f"[{level}] {self.name}: {message}")

    def handle_error(self, error: Exception, context: str = "") -> Dict[str, Any]:
        """
        Handle errors consistently across agents.

        Args:
            error: Exception that occurred
            context: Additional context about the error

        Returns:
            Error dictionary with standardized format
        """
        error_msg = f"{context}: {str(error)}" if context else str(error)
        self.log(error_msg, level="ERROR")

        return {"success": False, "error": error_msg, "error_type": type(error).__name__}

    def create_success_result(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create standardized success result.

        Args:
            data: Result data to return

        Returns:
            Success dictionary with data
        """
        return {"success": True, "error": None, **data}

    def __repr__(self) -> str:
        """String representation."""
        return f"{self.name}(verbose={self.verbose})"
