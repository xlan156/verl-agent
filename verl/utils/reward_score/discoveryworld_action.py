import re
from typing import Any


ACTION_RE = re.compile(r"<action>\s*([^<\s]+)\s*</action>", flags=re.IGNORECASE)


def is_use_dispenser(skill_name: str) -> bool:
    return skill_name.startswith("use_dispenser_")


def is_remove_chemical(skill_name: str) -> bool:
    return skill_name.startswith("remove_chemical_")


def extract_action(solution_str: str) -> str | None:
    """Extract the last action enclosed by <action>...</action>."""
    if not isinstance(solution_str, str):
        return None

    matches = ACTION_RE.findall(solution_str)
    if not matches:
        return None
    return matches[-1].strip()


def _normalize_ground_truth(ground_truth: Any) -> str:
    if isinstance(ground_truth, dict):
        ground_truth = ground_truth.get("action", "")
    return str(ground_truth).strip()


def compute_score(solution_str, ground_truth, format_score=0.0, score=1.0):
    """Reward +1 when the generated <action> matches the target action."""
    pred_action = extract_action(solution_str)
    target_action = _normalize_ground_truth(ground_truth)
    if pred_action is None:
        return 0.0
    
    both_use = is_use_dispenser(pred_action) and is_use_dispenser(target_action)
    both_remove = is_remove_chemical(pred_action) and is_remove_chemical(target_action)
    if pred_action == target_action:
        return float(score)
    return float(format_score)
