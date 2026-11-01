"""Skill package loading utilities for the TSSDA runner."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml


PHASE_PACKAGE_HINTS = {
    "P0": "p0_initialization",
    "P1": "p1_scope",
    "P2": "p2_temporal",
    "P3": "p3_group",
    "P4": "p4_operator",
    "P5": "p5_select",
    "P6": "p6_predicate",
    "P7": "p7_fill",
    "P8": "p8_into",
    "P9": "p9_result",
    "P10": "p10_finalization",
}


@dataclass
class SkillPackage:
    """Loaded skill package runtime bundle."""

    phase_id: str
    package_name: str
    package_dir: Path
    package_config: dict[str, Any]
    package_context: str
    skills_module: ModuleType
    outputs_module: ModuleType | None = None
    mask_module: ModuleType | None = None
    tool_module: ModuleType | None = None

    @property
    def skill_registry(self) -> dict[str, type]:
        return getattr(self.skills_module, "SKILL_REGISTRY", {})


class SkillPackageLoader:
    """Load package.yaml, context.md, skills.py, outputs.py, and mask.py."""

    def __init__(self, skills_root: str | Path | None = None):
        default_root = Path(__file__).resolve().parents[1] / "skills"
        self.skills_root = Path(skills_root) if skills_root is not None else default_root

    def load(self, phase_id: str) -> SkillPackage:
        """Load a skill package by phase id."""

        package_dir = self._resolve_package_dir(phase_id)
        package_config = self._load_yaml(package_dir / "package.yaml")
        package_name = package_config.get("package_name") or package_dir.name
        package_context = self._read_optional_text(package_dir / "context.md")

        module_prefix = f"skills.{package_dir.name}"
        skills_module = importlib.import_module(f"{module_prefix}.skills")
        outputs_module = self._optional_import(f"{module_prefix}.outputs")
        mask_module = self._optional_import(f"{module_prefix}.mask")
        tool_module = self._optional_import(f"{module_prefix}.tool")

        package = SkillPackage(
            phase_id=phase_id,
            package_name=package_name,
            package_dir=package_dir,
            package_config=package_config,
            package_context=package_context,
            skills_module=skills_module,
            outputs_module=outputs_module,
            mask_module=mask_module,
            tool_module=tool_module,
        )
        self._validate_package(package)
        return package

    def _validate_package(self, package: SkillPackage) -> None:
        """Validate runtime registration and package metadata consistency."""

        config = package.package_config
        configured_phase = config.get("phase_id")
        if configured_phase != package.phase_id:
            raise ValueError(
                f"{package.package_dir.name} declares phase_id="
                f"{configured_phase!r}, expected {package.phase_id!r}"
            )

        registry = package.skill_registry
        declared_skills = config.get("skills", []) or []
        if set(declared_skills) != set(registry):
            raise ValueError(
                f"{package.package_name} skill registration mismatch: "
                f"declared={declared_skills}, registered={list(registry)}"
            )

        execution_policy = config.get("execution_policy", {}) or {}
        entry_skill = execution_policy.get("entry_skill")
        if entry_skill not in registry:
            raise ValueError(
                f"{package.package_name} entry skill {entry_skill!r} is not registered"
            )
        for target in self._conditional_skill_targets(execution_policy):
            if target not in registry:
                raise ValueError(
                    f"{package.package_name} routes to unregistered skill {target!r}"
                )

        mask_policy = config.get("mask_policy", {}) or {}
        if mask_policy.get("enabled") and (
            package.mask_module is None
            or not hasattr(package.mask_module, "build_mask")
        ):
            raise ValueError(
                f"{package.package_name} enables masking without mask.build_mask"
            )

        outputs = config.get("outputs", {}) or {}
        if package.outputs_module is None:
            if outputs.get("spo_keys") or outputs.get("flag_keys"):
                raise ValueError(
                    f"{package.package_name} declares outputs without outputs.py"
                )
            return
        configured_spo = set(outputs.get("spo_keys", []) or [])
        configured_flags = set(outputs.get("flag_keys", []) or [])
        module_spo = set(getattr(package.outputs_module, "SPO_SCHEMA", {}))
        module_flags = set(getattr(package.outputs_module, "FLAG_SCHEMA", {}))
        if configured_spo != module_spo or configured_flags != module_flags:
            raise ValueError(
                f"{package.package_name} output registration mismatch: "
                f"SPO {sorted(configured_spo)} != {sorted(module_spo)}, "
                f"Flags {sorted(configured_flags)} != {sorted(module_flags)}"
            )

    @staticmethod
    def _conditional_skill_targets(execution_policy: dict[str, Any]) -> list[str]:
        conditional = execution_policy.get("conditional_skills", {}) or {}
        values = conditional.values() if isinstance(conditional, dict) else conditional
        targets: list[str] = []
        for value in values:
            if isinstance(value, str):
                targets.append(value)
            elif isinstance(value, list):
                targets.extend(str(item) for item in value)
        return targets

    def _resolve_package_dir(self, phase_id: str) -> Path:
        hinted_name = PHASE_PACKAGE_HINTS.get(phase_id)
        if hinted_name:
            package_dir = self.skills_root / hinted_name
            if package_dir.exists():
                return package_dir

        for package_dir in self.skills_root.iterdir():
            package_yaml = package_dir / "package.yaml"
            if not package_yaml.exists():
                continue
            config = self._load_yaml(package_yaml)
            if config.get("phase_id") == phase_id:
                return package_dir

        raise FileNotFoundError(f"No skill package found for phase {phase_id}")

    def _load_yaml(self, path: Path) -> dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            return yaml.safe_load(handle) or {}

    def _read_optional_text(self, path: Path) -> str:
        if not path.exists():
            return ""
        return path.read_text(encoding="utf-8")

    def _optional_import(self, module_name: str) -> ModuleType | None:
        try:
            return importlib.import_module(module_name)
        except ModuleNotFoundError:
            return None
