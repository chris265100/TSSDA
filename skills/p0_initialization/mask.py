"""Keep task initialization within the read-only generation workflow."""


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    if action_id == "P0A1":
        return [action for action in action_space if action != "materialization"]
    return list(action_space)
