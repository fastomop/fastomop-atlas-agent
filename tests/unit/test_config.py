"""Unit tests for `atlas_agent.config` — pure-Python, no external services."""

import pytest

from atlas_agent.config import deep_merge, get_agent_config


class TestDeepMerge:
    def test_overrides_top_level_scalars(self):
        assert deep_merge({"a": 1, "b": 2}, {"b": 3}) == {"a": 1, "b": 3}

    def test_recursively_merges_nested_dicts(self):
        base = {"models": {"providers": {"openai": {"id": "gpt-4"}}}}
        override = {"models": {"providers": {"openai": {"id": "gpt-4-turbo"}}}}
        result = deep_merge(base, override)
        assert result["models"]["providers"]["openai"]["id"] == "gpt-4-turbo"

    def test_keeps_unrelated_nested_keys(self):
        base = {"a": {"x": 1, "y": 2}}
        override = {"a": {"y": 3}}
        assert deep_merge(base, override) == {"a": {"x": 1, "y": 3}}

    def test_does_not_mutate_base(self):
        base = {"a": 1, "nested": {"b": 2}}
        deep_merge(base, {"a": 99, "nested": {"b": 99}})
        assert base == {"a": 1, "nested": {"b": 2}}

    def test_override_replaces_scalar_with_dict(self):
        assert deep_merge({"a": 1}, {"a": {"x": 1}}) == {"a": {"x": 1}}


class TestGetAgentConfig:
    """`get_agent_config` reads the bundled `config.toml` and merges in any
    `config.local.toml` overrides. We only assert structural invariants
    here — the exact provider/model strings come from the toml files and
    are not part of this contract."""

    def test_known_agent_returns_complete_config(self):
        # `orchestrator` is one of the agents defined in config.toml.
        cfg = get_agent_config("orchestrator")
        assert "MODEL_TYPE" in cfg
        assert "MODEL_ID" in cfg
        assert isinstance(cfg["MODEL_TYPE"], str)
        assert isinstance(cfg["MODEL_ID"], str)

    def test_unknown_agent_falls_back_to_default_provider(self):
        cfg = get_agent_config("definitely_not_a_real_agent_name")
        # Falls back to the default provider; still returns a usable shape.
        assert "MODEL_TYPE" in cfg
        assert "MODEL_ID" in cfg
        assert cfg["name"] == "definitely_not_a_real_agent_name"

    @pytest.mark.parametrize(
        "agent_name",
        ["clinical_parser", "concept_finder", "relationship_reasoner", "set_builder", "validator", "corrector"],
    )
    def test_all_pipeline_agents_resolvable(self, agent_name):
        cfg = get_agent_config(agent_name)
        assert cfg["MODEL_TYPE"]
        assert cfg["MODEL_ID"]
