"""
Configuration management for the research assistant.

Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


# Load .env file if it exists
load_dotenv()


# Type aliases
ModelType = Literal["llama3.2:3b", "llama3.1:8b", "llama3.1:70b", "mistral:7b"]


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    Settings can be configured via:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values (lowest priority)

    Example .env file:
        OLLAMA_BASE_URL=http://localhost:11434
        OLLAMA_MODEL=llama3.2:3b
        MAX_SOURCES=10
        TRUSTWORTHINESS_THRESHOLD=85
    """

    # Ollama Configuration
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Ollama API base URL"
    )

    ollama_model: str = Field(
        default="llama3.2:3b",
        description="Default Ollama model to use"
    )

    ollama_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature (0.0 = deterministic, 2.0 = very random)"
    )

    # Research Configuration
    max_sources: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of sources to discover"
    )

    trustworthiness_threshold: float = Field(
        default=85.0,
        ge=0.0,
        le=100.0,
        description="Minimum trustworthiness score to store sources"
    )

    max_search_results_per_query: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results per query"
    )

    # Paths
    vector_db_path: Path = Field(
        default=Path("./data/vector_db"),
        description="ChromaDB persistence directory"
    )

    reports_path: Path = Field(
        default=Path("./data/reports"),
        description="Directory for generated reports"
    )

    # Scraping Configuration
    scraper_timeout: int = Field(
        default=10,
        ge=1,
        le=60,
        description="Timeout for web scraping requests (seconds)"
    )

    scraper_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retries for failed scrapes"
    )

    # Embedding Configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        case_sensitive = False  # Allow lowercase env vars
    )

    def ensure_directories(self) -> None:
        """
        Ensure required directories exist.

        Creates vector_db_path and reports_path if they don't exist.
        """
        self.vector_db_path.mkdir(parents=True, exist_ok=True)
        self.reports_path.mkdir(parents=True, exist_ok=True)

    def get_report_path(self, topic: str) -> Path:
        """
        Get path for a research report.

        Args:
            topic: Research topic

        Returns:
            Path to HTML report file

        Example:
            >>> settings = Settings()
            >>> settings.get_report_path("AI Safety")
            PosixPath('data/reports/ai_safety.html')
        """
        # Sanitize topic for filename
        safe_topic = topic.lower().replace(" ", "_")
        safe_topic = "".join(c for c in safe_topic if c.isalnum() or c == "_")
        return self.reports_path / f"{safe_topic}.html"

    def __str__(self) -> str:
        """String representation"""
        return (
            f"Settings(model={self.ollama_model}, "
            f"max_sources={self.max_sources}, "
            f"threshold={self.trustworthiness_threshold})"
        )


# Global settings instance
# Import this in other modules: from research_assistant.config import settings
settings = Settings()


# Ensure directories exist on import
settings.ensure_directories()


def get_settings() -> Settings:
    """
    Get settings instance.

    Returns:
        Configured Settings instance

    Example:
        >>> from research_assistant.config import get_settings
        >>> config = get_settings()
        >>> print(config.ollama_model)
        llama3.2:3b
    """
    return settings


def reload_settings() -> Settings:
    """
    Reload settings from environment.

    Useful for testing or dynamic configuration changes.

    Returns:
        Fresh Settings instance
    """
    load_dotenv(override=True)
    return Settings()
