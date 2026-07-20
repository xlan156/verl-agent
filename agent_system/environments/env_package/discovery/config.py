from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


def slugify(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def build_frames_dir(env_kwargs: Dict[str, Any], seed: int, is_train: bool) -> str:
    model_name = env_kwargs.get("model_name") or os.environ.get("MODEL_NAME")
    job_id = slugify(os.environ.get("SLURM_JOB_ID"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    split = "train" if is_train else "eval"
    return os.path.join(
        "outputs",
        "discoveryworld_frames",
        f"{model_name}__seed{seed}__{job_id}__{timestamp}__{split}",
    )


def coerce_max_chemical_n(env_kwargs: Dict[str, Any], default: int = 2) -> int:
    """Read the canonical chemical amount while accepting legacy config keys."""
    return int(
        env_kwargs.get(
            "max_chemical_n",
            env_kwargs.get("max_chemical_N", env_kwargs.get("chemical_N", default)),
        )
    )


def coerce_bool(value: Any, default: bool = False) -> bool:
    """Parse bool-like config values without treating "False" as true."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off", ""}:
            return False
    return bool(value)


def remove_legacy_chemical_keys(env_kwargs: Dict[str, Any]) -> None:
    """Drop legacy chemical amount aliases after canonicalization."""
    env_kwargs.pop("max_chemical_n", None)
    env_kwargs.pop("max_chemical_N", None)
    env_kwargs.pop("chemical_N", None)


@dataclass
class DiscoveryWorkerConfig:
    scenario_name: Optional[str] = None
    difficulty: Optional[str] = None
    max_steps: int = 50
    save_frames: bool = False
    frames_dir: Optional[str] = None
    max_chemical_n: int = 2
    teacher_skill_reward_coef: float = 0.1
    env_variant: str = "original"

    @classmethod
    def from_env_kwargs(
        cls,
        env_kwargs: Optional[Dict[str, Any]],
    ) -> "DiscoveryWorkerConfig":
        kwargs = dict(env_kwargs or {})
        max_chemical_n = coerce_max_chemical_n(kwargs)
        remove_legacy_chemical_keys(kwargs)

        return cls(
            scenario_name=kwargs.pop("scenario_name", None),
            difficulty=kwargs.pop("difficulty", None),
            max_steps=int(kwargs.pop("max_steps", 50)),
            save_frames=coerce_bool(kwargs.pop("save_frames", False)),
            frames_dir=kwargs.pop("frames_dir", None),
            max_chemical_n=max_chemical_n,
            teacher_skill_reward_coef=float(kwargs.pop("teacher_skill_reward_coef", 0.1)),
            env_variant=str(kwargs.pop("env_variant", "original")),  # original, pickupjar, derustmoderate
        )
