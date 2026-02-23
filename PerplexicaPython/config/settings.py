"""
Application settings and configuration management.
Loads environment variables and provides typed configuration objects.
"""

from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Main application settings loaded from environment variables."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )
    
    # Groq API Configuration
    groq_api_key: str = Field(..., description="Groq API key for LLM access")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile",
        description="Default Groq model to use"
    )
    
    # SearXNG Configuration
    searxng_url: str = Field(
        default="https://rameshrathod-ai-private-search-engine.hf.space/search",
        description="SearXNG instance URL"
    )
    
    # Weather API (Optional)
    openweather_api_key: str | None = Field(
        default=None,
        description="OpenWeather API key for weather widget"
    )
    
    # Database Configuration
    database_url: str = Field(
        default="sqlite:///./perplexica.db",
        description="Database connection URL"
    )
    
    # Application Settings
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Logging level"
    )
    
    max_search_results: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum number of search results to retrieve"
    )
    
    # Research Iteration Limits
    max_research_iterations_speed: int = Field(default=2, ge=1, le=5)
    max_research_iterations_balanced: int = Field(default=6, ge=1, le=15)
    max_research_iterations_quality: int = Field(default=25, ge=1, le=50)
    
    # LLM Settings
    llm_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="LLM temperature for generation"
    )
    llm_max_tokens: int = Field(
        default=4096,
        ge=256,
        le=32768,
        description="Maximum tokens for LLM generation"
    )
    llm_streaming: bool = Field(
        default=True,
        description="Enable streaming responses"
    )
    
    # Search Settings
    search_timeout: int = Field(
        default=30,
        ge=5,
        le=120,
        description="Search request timeout in seconds"
    )
    search_language: str = Field(
        default="en",
        description="Default search language code"
    )
    
    def get_max_iterations(self, mode: Literal["speed", "balanced", "quality"]) -> int:
        """Get max research iterations based on optimization mode."""
        if mode == "speed":
            return self.max_research_iterations_speed
        elif mode == "balanced":
            return self.max_research_iterations_balanced
        else:
            return self.max_research_iterations_quality


# Global settings instance
_settings: Settings | None = None


def get_settings() -> Settings:
    """Get or create the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Reload settings from environment (useful for testing)."""
    global _settings
    _settings = Settings()
    return _settings
