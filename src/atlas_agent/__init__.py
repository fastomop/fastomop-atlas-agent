"""OMOP ATLAS Concept Set Agent."""
__version__ = "0.1.0"

from .config import get_agent_config, config
from .config import init as _config_init
from .model_factory import create_model, set_factory


def init(config_dict, model_factory=None, agent_prefix="atlas_"):
    """Initialize atlas_agent with external configuration.

    Call this BEFORE creating any agent instances (e.g. OrchestratorAgent).
    When not called, the module works standalone with file-based config.

    Args:
        config_dict: Unified config dict from the host application
        model_factory: Optional custom model factory function. When provided,
                       create_model() delegates to this instead of built-in logic.
        agent_prefix: Prefix for agent config keys to avoid collisions
    """
    _config_init(config_dict, agent_prefix=agent_prefix)
    if model_factory:
        set_factory(model_factory)


__all__ = ["init", "get_agent_config", "config", "create_model"]
