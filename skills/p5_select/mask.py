"""Action mask scaffold for this TSSDA skill package."""

from __future__ import annotations


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    """Return currently valid candidates for an action-level skill.

    The scaffold keeps all candidates enabled. Phase-specific grammar and
    semantic dependency constraints should be added during detailed phase work.
    """

    return list(action_space)
