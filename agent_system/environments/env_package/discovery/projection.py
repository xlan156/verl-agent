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

from typing import Any, Dict, List, Optional, Tuple
import json
import re
from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI

AVAILABLE_ACTIONS = DiscoveryWorldAPI.listKnownActionsStatic()

# 用于解析 LLM 输出中的代码块
_CODE_BLOCK_JSON_RE = re.compile(r"```json(.*?)```", re.DOTALL | re.IGNORECASE)
_CODE_BLOCK_GENERIC_RE = re.compile(r"```(.*?)```", re.DOTALL)

# 记录每个 env 上一次通过校验的合法 action，
# 在本次 action 非法时用于回退，避免直接把垃圾字符串丢给环境。
_ACTION_MEMORY: List[Optional[str]] = []

_DEFAULT_SAFE_ACTION = json.dumps({"action": "MOVE_DIRECTION", "arg1": "east"}, separators=(",", ":"))


def _extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """Robustly extract a JSON object from LLM output.

    Priority:
      1) Last ```json ... ``` block.
      2) Else last ``` ... ``` block.
      3) Else whole text.
    Returns parsed dict or None.
    """
    candidate: Optional[str] = None

    # 1) ```json ... ```
    matches = _CODE_BLOCK_JSON_RE.findall(text)
    if matches:
        candidate = matches[-1].strip()
    else:
        # 2) 任意 ``` ... ```
        matches = _CODE_BLOCK_GENERIC_RE.findall(text)
        if matches:
            candidate = matches[-1].strip()

    # 3) 没有代码块，就用原文
    if candidate is None:
        candidate = text.strip()

    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except Exception:
        return None
    

def discoveryworld_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """

    """

    processed: List[str] = []
    valids: List[int] = [0] * len(actions)

    # 确保 action memory 长度与当前 batch 对齐
    global _ACTION_MEMORY
    if len(_ACTION_MEMORY) < len(actions):
        _ACTION_MEMORY.extend([None] * (len(actions) - len(_ACTION_MEMORY)))

    # Regex patterns
    re_action_block = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)
    re_think_block = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
    re_chinese = re.compile(r"[\u4e00-\u9fff]")

    for i, s in enumerate(actions):
        original = s

        # ---- 优先从 <action>...</action> 中抽 inner 文本 ----
        m = re_action_block.search(s)
        if not m:
            # 没有显式 <action>，退回到整串的最后一段
            candidate_text = s.strip()[-512:]
        else:
            candidate_text = m.group(1).strip()

        # ---- 先直接 json.loads，失败则用代码块提取 ----
        parsed: Optional[Dict[str, Any]] = None
        try:
            tmp = json.loads(candidate_text)
            if isinstance(tmp, dict):
                parsed = tmp
        except Exception:
            parsed = _extract_json_from_text(candidate_text)

        # 先基于 action 字段做一次合法性判断
        candidate_out: str
        is_valid = False
        if parsed is not None:
            candidate_out = json.dumps(parsed, separators=(",", ":"))
            action_name = parsed.get("action")
            if isinstance(action_name, str) and action_name in AVAILABLE_ACTIONS:
                is_valid = True
        else:
            # 仍然没法 parse 成 dict，就把原始片段作为候选
            candidate_out = candidate_text

        # ---- think / 中文 检查并更新 valid 标记 ----
        if not re_think_block.search(original):
            is_valid = False
        if re_chinese.search(original):
            is_valid = False

        if is_valid:
            valids[i] = 1
            _ACTION_MEMORY[i] = candidate_out
        else:
            valids[i] = 0
            # 本次无效：若有历史合法动作，则回退到历史动作；否则给一个安全的默认动作
            if _ACTION_MEMORY[i] is not None:
                candidate_out = _ACTION_MEMORY[i]  # 让环境继续前进
            else:
                candidate_out = _DEFAULT_SAFE_ACTION

        processed.append(candidate_out)

    return processed, valids
