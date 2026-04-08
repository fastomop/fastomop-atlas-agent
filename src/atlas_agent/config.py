"""Configuration for ATLAS agent."""
import os
from pathlib import Path
from dotenv import load_dotenv
import tomli
from typing import Dict, Any

# Load environment variables
load_dotenv()

# Load config.toml
CONFIG_DIR = Path(__file__).parent.parent.parent
CONFIG_PATH = CONFIG_DIR / "config.toml"
LOCAL_CONFIG_PATH = CONFIG_DIR / "config.local.toml"


def deep_merge(base: dict, override: dict) -> dict:
    """Deep merge override dict into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config() -> dict:
    """Load config with hierarchy: .env > config.local.toml > config.toml"""
    if not CONFIG_PATH.exists():
        return {}

    # Load base config
    with open(CONFIG_PATH, "rb") as f:
        config = tomli.load(f)

    # Override with local config if it exists
    if LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH, "rb") as f:
            local_config = tomli.load(f)
        config = deep_merge(config, local_config)

    return config


config = load_config()


def get_agent_config(agent_name: str) -> Dict[str, Any]:
    """
    Get agent configuration with model provider settings.

    Args:
        agent_name: Name of the agent (e.g., 'orchestrator', 'concept_finder')

    Returns:
        Dictionary with complete agent configuration including model settings
    """
    agents_config = config.get("agents", {})
    if agent_name not in agents_config:
        # Fallback to default provider if agent not configured
        default_provider = config.get("models", {}).get("default_provider", "ollama")
        provider_config = config.get("models", {}).get("providers", {}).get(default_provider, {})

        return {
            "name": agent_name,
            "MODEL_TYPE": provider_config.get("provider", "ollama"),
            "MODEL_ID": provider_config.get("model_id", "gpt-oss:20b"),
        }

    agent_config = agents_config[agent_name].copy()

    # Get provider
    provider = agent_config.get("model_provider", config["models"]["default_provider"])
    provider_config = config["models"]["providers"][provider].copy()

    complete_config = {
        **agent_config,
        "MODEL_TYPE": provider_config["provider"],
        "MODEL_ID": provider_config["model_id"],
    }

    # Add Azure-specific settings
    if provider == "azure":
        complete_config["api_version"] = provider_config.get("api_version", "2025-01-01-preview")
        complete_config["temperature"] = provider_config.get("temperature", 0.0)

    return complete_config


# Legacy configuration (for backwards compatibility)
# Milvus
MILVUS_HOST = os.getenv("MILVUS_HOST", config.get("milvus", {}).get("host", "localhost"))
MILVUS_PORT = int(os.getenv("MILVUS_PORT", config.get("milvus", {}).get("port", "19530")))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", config.get("milvus", {}).get("collection_name", "omop_clinical_concepts"))

# Embedding
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", config.get("embedding", {}).get("model", "abhinand/MedEmbed-large-v0.1"))

# LLM (legacy - use get_agent_config instead)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
default_provider = config.get("models", {}).get("providers", {}).get("ollama", {})
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", default_provider.get("model_id", "gpt-oss:20b"))

# Output
OUTPUT_DIRECTORY = Path(os.getenv("OUTPUT_DIRECTORY", config.get("output", {}).get("export_directory", "./output")))
