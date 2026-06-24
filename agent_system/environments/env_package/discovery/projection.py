"""Project a model response into one DiscoveryWorld skill.

The rollout policy is trained to emit a ReAct-style response.  Keeping the
format check here (rather than silently recovering an action from malformed
text) is important for GRPO: a malformed response must be distinguishable
from a correctly formatted response so the invalid-action penalty can teach
both the ``think`` and ``action`` portions of the policy.
"""

from typing import Any, Dict, List, Optional, Tuple
import re

from agent_system.environments.env_package.discovery.utils import SKILL_NAMES


# The anchors deliberately reject extra prose, a second action, or an action
# before reasoning.  Whitespace around tags is harmless, while an empty think
# block is not accepted: this is an initialization task that should learn to
# produce actual reasoning as well as an executable action.
_THINK_ACTION_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*"
    r"<action>\s*(?P<action>[A-Za-z0-9_]+)\s*</action>\s*$",
    re.IGNORECASE | re.DOTALL,
)


def _extract_skill(response: Any) -> Optional[str]:
    """Return the exact allowed skill from a well-formed model response."""
    if not isinstance(response, str):
        return None

    match = _THINK_ACTION_RE.fullmatch(response)
    if match is None or not match.group("think").strip():
        return None

    # Skill identifiers are case-sensitive because they are names in the
    # environment's skill map.  The tags themselves stay case-insensitive.
    skill = match.group("action").strip()
    return skill if skill in SKILL_NAMES else None


def discoveryworld_projection(
    actions: List[str],
    infos: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Optional[str]], List[int]]:
    """Extract one skill and report whether the complete response is valid.

    ``infos`` remains an optional argument for compatibility with the
    environment-manager projection interface; action availability is encoded
    by ``SKILL_NAMES`` for this task.
    """
    del infos  # Kept in the signature so callers can share a projection API.

    # ``None`` is intentional.  It says no environment skill was extracted;
    # the environment then produces an invalid/no-op transition and its
    # associated reward.  A magic fallback string can accidentally become a
    # real action in a future environment implementation.
    processed: List[Optional[str]] = []
    valids: List[int] = []
    for response in actions:
        skill = _extract_skill(response)
        processed.append(skill)
        valids.append(int(skill is not None))
    return processed, valids
