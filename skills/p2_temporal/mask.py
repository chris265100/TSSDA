"""Executable temporal constraints from the shared curated rule set."""

from __future__ import annotations

from tssda.constraints import mask_actions


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    return mask_actions(action_id, state, action_space)
