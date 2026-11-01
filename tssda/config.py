"""Shared runtime configuration for TSSDA."""

from __future__ import annotations

import os
from pathlib import Path


EXAMPLE_SCHEMA_ROOT = "root.example"
SHARED_KNOWLEDGE_ROOT = (
    Path("runtime") / "knowledge"
)
DEFAULT_MEMORY_PATH = (
    SHARED_KNOWLEDGE_ROOT / "state_memory.json"
)
DEFAULT_DECISION_LEARNING_PATH = (
    SHARED_KNOWLEDGE_ROOT / "decision_learning.json"
)
DEFAULT_SKILL_ROUTING_PATH = SHARED_KNOWLEDGE_ROOT / "skill_routing.json"
DEFAULT_EXECUTION_KNOWLEDGE_PATH = (
    SHARED_KNOWLEDGE_ROOT / "execution_errors.json"
)


def get_default_execution_knowledge_path() -> Path:
    """Return the local execution-error knowledge-base file."""

    return Path(os.getenv("TSSDA_EXECUTION_KNOWLEDGE_PATH")
                or DEFAULT_EXECUTION_KNOWLEDGE_PATH).expanduser()


def get_default_schema_root() -> str:
    """Return the configured tree-model database root."""

    return (
        os.getenv("TSSDA_SCHEMA_ROOT")
        or os.getenv("IOTDB_SCHEMA_ROOT")
        or EXAMPLE_SCHEMA_ROOT
    )


def get_default_memory_path() -> Path:
    """Return the local state-memory file configured for this run."""

    configured = os.getenv("TSSDA_MEMORY_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_MEMORY_PATH


def get_default_decision_learning_path() -> Path:
    """Return the persisted distribution-aware online statistics file."""

    configured = os.getenv("TSSDA_DECISION_LEARNING_PATH")
    if configured:
        return Path(configured).expanduser()
    return DEFAULT_DECISION_LEARNING_PATH


def get_default_skill_routing_path() -> Path:
    """Return the semantic Skill routing statistics file."""
    return Path(os.getenv("TSSDA_SKILL_ROUTING_PATH") or DEFAULT_SKILL_ROUTING_PATH).expanduser()
