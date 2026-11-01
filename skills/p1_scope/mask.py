"""Schema-scope actions are refined by schema compatibility in each skill."""


def build_mask(action_id: str, state: dict, action_space: list[str]) -> list[str]:
    return list(action_space)
