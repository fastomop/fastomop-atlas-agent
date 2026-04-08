"""Model factory for creating different LLM providers."""
from agno.models.anthropic import Claude
from agno.models.azure import AzureOpenAI
from agno.models.openai import OpenAIChat
from agno.models.ollama import Ollama
from typing import Dict, Any
import os


def create_model(config: Dict[str, Any]) -> Any:
    """
    Create a model instance based on provider settings.

    Args:
        config: Configuration dictionary containing:
            - MODEL_TYPE: Provider type (anthropic, azure, openai, ollama)
            - MODEL_ID: Model identifier
            - api_version: (Azure only) API version
            - temperature: (Optional) Model temperature

    Returns:
        Model instance for the specified provider

    Raises:
        ValueError: If unknown model type is specified
    """
    model_type = config.get("MODEL_TYPE", "ollama")
    model_id = config.get("MODEL_ID", "gpt-oss:20b")

    if model_type == "anthropic":
        return Claude(id=model_id)

    elif model_type == "azure":
        return AzureOpenAI(
            id=model_id,
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT"),
            api_version=config.get("api_version", "2025-01-01-preview"),
            temperature=config.get("temperature")
        )

    elif model_type == "openai":
        return OpenAIChat(id=model_id)

    elif model_type == "ollama":
        from .config import OLLAMA_BASE_URL
        return Ollama(id=model_id, host=OLLAMA_BASE_URL)

    else:
        raise ValueError(f"Unknown model type: {model_type}")
