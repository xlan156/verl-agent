# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple
import json
import re

AVAILABLE_ACTIONS = ["MOVE_DIRECTION", "ROTATE_DIRECTION", "PICKUP", "USE", "TALK", "PUT"]

def discoveryworld_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """Parse model outputs into JSON action strings for DiscoveryWorld.

    Expected model pattern (recommended but not strictly required):

        <think> ... reasoning ... </think>
        <action>{"action": ..., "arg1": ..., "arg2": ...}</action>

    This function:
      1. Extracts the first <action>...</action> block (case-insensitive).
      2. Tries to parse its inner text as JSON. If parsing fails, marks as invalid
         but still returns the *raw* inner text so the env can decide what to do.
      3. Validity flag is additionally set to 0 if:
           - No <think>...</think> is present, or
           - The original string contains any Chinese character.

    Returns:
      - processed_actions: list of JSON strings or raw inner texts.
      - valids: list[int], 1 for valid, 0 otherwise.
    """

    processed: List[str] = []
    valids: List[int] = [0] * len(actions)

    # Regex patterns
    re_action_block = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
    re_think_block = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
    re_chinese = re.compile(r"[\u4e00-\u9fff]")

    for i, s in enumerate(actions):
        original = s

        # ---- extract inner of <action>...</action> ----
        m = re_action_block.search(s)
        if not m:
            # no explicit <action> block; fall back to entire string (last ~256 chars)
            candidate = s.strip()[-256:]
        else:
            candidate = m.group(1).strip()

        # try JSON parse; if ok, re-dump to a clean canonical JSON string
        try:
            parsed = json.loads(candidate)
            candidate_out = json.dumps(parsed, separators=(",", ":"))
            # Action field must be present and in AVAILABLE_ACTIONS
            action = parsed.get("action") if isinstance(parsed, dict) else None
            if isinstance(action, str) and action in AVAILABLE_ACTIONS:
                valids[i] = 1
            else:
                valids[i] = 0
        except Exception:
            # keep raw inner text; env will attempt json.loads again
            candidate_out = candidate
            # keep valids[i] at 0 for now

        # ---- think tag check ----
        if not re_think_block.search(original):
            valids[i] = 0

        # ---- Chinese character check ----
        if re_chinese.search(original):
            valids[i] = 0

        processed.append(candidate_out)

    return processed, valids
