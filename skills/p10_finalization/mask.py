"""P10 currently has no action masking rules."""

from __future__ import annotations


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    return list(action_space)
