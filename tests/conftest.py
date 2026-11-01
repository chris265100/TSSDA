"""Keep every offline test's runtime state outside the source tree."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in ("TSSDA_MEMORY_PATH", "TSSDA_DECISION_LEARNING_PATH",
                 "TSSDA_EXECUTION_KNOWLEDGE_PATH", "TSSDA_SKILL_ROUTING_PATH"):
        monkeypatch.delenv(name, raising=False)
