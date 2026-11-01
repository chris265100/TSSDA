"""Apply the grammar constraints declared by each phase package."""
from typing import Any
from .package_loader import SkillPackage


class MaskRunner:
    def apply(self, package: SkillPackage, skill_id: str, action_id: str,
              state_view: dict[str, Any], action_space: list[str]) -> list[str]:
        masked = list(action_space)
        if package.mask_module is not None and hasattr(package.mask_module, "build_mask"):
            masked = package.mask_module.build_mask(
                action_id=action_id, state=state_view, action_space=masked,
            )
        if not masked:
            raise ValueError(f"No grammar-compatible candidates for {skill_id} ({action_id})")
        return list(masked)
