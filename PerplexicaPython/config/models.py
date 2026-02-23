"""
Model configuration and metadata.
Defines available LLM models and their properties.
"""

from typing import Literal
from pydantic import BaseModel, Field


class ModelConfig(BaseModel):
    """Configuration for a specific LLM model."""
    
    provider: Literal["groq", "ollama", "openai"] = Field(
        description="LLM provider name"
    )
    model_id: str = Field(description="Model identifier")
    display_name: str = Field(description="Human-readable model name")
    context_window: int = Field(description="Maximum context window size in tokens")
    supports_streaming: bool = Field(default=True, description="Supports streaming")
    supports_tools: bool = Field(default=True, description="Supports tool/function calling")
    supports_json_mode: bool = Field(default=True, description="Supports JSON output mode")
    max_output_tokens: int = Field(description="Maximum output tokens")
    
    # Cost information (optional, for future usage tracking)
    cost_per_1k_input_tokens: float | None = Field(default=None)
    cost_per_1k_output_tokens: float | None = Field(default=None)


# Groq Models
GROQ_MODELS = {
    "llama-3.3-70b-versatile": ModelConfig(
        provider="groq",
        model_id="llama-3.3-70b-versatile",
        display_name="Llama 3.3 70B",
        context_window=128000,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "llama-3.1-70b-versatile": ModelConfig(
        provider="groq",
        model_id="llama-3.1-70b-versatile",
        display_name="Llama 3.1 70B",
        context_window=128000,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "mixtral-8x7b-32768": ModelConfig(
        provider="groq",
        model_id="mixtral-8x7b-32768",
        display_name="Mixtral 8x7B",
        context_window=32768,
        max_output_tokens=32768,
        supports_streaming=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
    "gemma2-9b-it": ModelConfig(
        provider="groq",
        model_id="gemma2-9b-it",
        display_name="Gemma 2 9B",
        context_window=8192,
        max_output_tokens=8192,
        supports_streaming=True,
        supports_tools=True,
        supports_json_mode=True,
    ),
}


def get_model_config(model_id: str, provider: str = "groq") -> ModelConfig:
    """
    Get configuration for a specific model.
    
    Args:
        model_id: Model identifier
        provider: Provider name (default: groq)
        
    Returns:
        ModelConfig object
        
    Raises:
        ValueError: If model not found
    """
    if provider == "groq":
        if model_id not in GROQ_MODELS:
            raise ValueError(
                f"Unknown Groq model: {model_id}. "
                f"Available models: {', '.join(GROQ_MODELS.keys())}"
            )
        return GROQ_MODELS[model_id]
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def list_available_models(provider: str = "groq") -> list[str]:
    """
    List all available models for a provider.
    
    Args:
        provider: Provider name (default: groq)
        
    Returns:
        List of model IDs
    """
    if provider == "groq":
        return list(GROQ_MODELS.keys())
    else:
        return []
