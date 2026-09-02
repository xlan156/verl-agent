"""Project a model response into one DiscoveryWorld skill.

The rollout policy is trained to emit a ReAct-style response.  Keeping the
format check here (rather than silently recovering an action from malformed
text) is important for GRPO: a malformed response must be distinguishable
from a correctly formatted response so the invalid-action penalty can teach
both the ``think`` and ``action`` portions of the policy.
"""

from typing import Any, Dict, List, Optional, Tuple
import re

# The anchors deliberately reject extra prose, a second action, or an action
# before reasoning.  Whitespace around tags is harmless, while an empty think
# block is not accepted: this is an initialization task that should learn to
# produce actual reasoning as well as an executable action.
_THINK_ACTION_RE = re.compile(
    r"^\s*<think>\s*(?P<think>.*?)\s*</think>\s*"
    r"<action>\s*(?P<action>[A-Za-z0-9_]+)\s*</action>\s*$",
    re.IGNORECASE | re.DOTALL,
)

_THINK_BLOCK_RE = re.compile(r"<think>\s*(?P<think>.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
_ACTION_BLOCK_RE = re.compile(r"<action>\s*(?P<action>.*?)\s*</action>", re.IGNORECASE | re.DOTALL)


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
    return skill


def response_format_score(response: Any) -> float:
    """Return one syntax-only diagnostic for the response schema.

    Only a score of 1.0 is executable. Lower scores describe how close the
    response is to the schema but do not contribute to environment reward.
    """
    if not isinstance(response, str):
        return 0.0

    strict_match = _THINK_ACTION_RE.fullmatch(response)
    if strict_match is not None:
        skill = strict_match.group("action").strip()
        if strict_match.group("think").strip() and skill:
            return 1.0

    think_blocks = list(_THINK_BLOCK_RE.finditer(response))
    action_blocks = list(_ACTION_BLOCK_RE.finditer(response))

    # The model has emitted one usable action in an action block, but has not
    # yet produced the complete, clean two-block response.
    if len(action_blocks) == 1:
        action = action_blocks[0].group("action").strip()
        if re.fullmatch(r"[A-Za-z0-9_]+", action):
            if len(think_blocks) == 1 and think_blocks[0].group("think").strip():
                return 0.75
            return 0.50

    # Both block types exist, which is a useful precursor to the target
    # protocol, but the action block is malformed or names an invalid skill.
    if len(think_blocks) == 1 and think_blocks[0].group("think").strip() and len(action_blocks) == 1:
        return 0.25

    # Reward a non-empty thought block slightly: it gives malformed groups a
    # direction toward the required order without making it executable.
    if len(think_blocks) == 1 and think_blocks[0].group("think").strip():
        return 0.10

    return 0.0


def discoveryworld_projection(
    actions: List[str],
    infos: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[List[Optional[str]], List[int]]:
    """Extract one skill and report whether the complete response is valid.

    ``infos`` is used when available to reject skills that are globally valid
    but unavailable in the current high-level task phase.
    """
    # ``None`` is intentional.  It says no environment skill was extracted;
    # the environment then produces an invalid/no-op transition and its
    # associated reward.  A magic fallback string can accidentally become a
    # real action in a future environment implementation.
    processed: List[Optional[str]] = []
    valids: List[int] = []
    infos = list(infos or [])
    if len(infos) < len(actions):
        infos.extend([None] * (len(actions) - len(infos)))
        
    for response, info in zip(actions, infos):
        skill = _extract_skill(response)
        
        if skill is not None and info is not None:
            prompt_valid_skills = (info or {}).get("valid_skills")
            valid_skills = set(prompt_valid_skills or [])
            
            if skill not in valid_skills:
                dynamic_reactor = bool((info or {}).get("reactor_dynamic_frequency"))
                dynamic_match = re.fullmatch(r"set_reactor_([3-5])_frequency_(\d{1,5})", skill)
                if not (dynamic_reactor and dynamic_match is not None
                        and int(dynamic_match.group(2)) <= 10000
                        and "set_reactor_{}_frequency_<number>".format(dynamic_match.group(1)) in valid_skills):
                    skill = None
                
        processed.append(skill)
        valids.append(int(skill is not None))
        
    return processed, valids
