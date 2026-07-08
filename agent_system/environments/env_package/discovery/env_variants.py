from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Optional, Tuple

from agent_system.environments.env_package.discovery.envs import DiscoveryWorldEnv
from agent_system.environments.env_package.discovery.utils import RUST_LABEL_TO_LEVEL, extract_detailed_status


class CCEnvPickJar(DiscoveryWorldEnv):
    """DiscoveryWorld variant that ends once the key is inside a carried jar."""

    def _is_task_complete(self, info: Optional[Dict[str, Any]] = None) -> bool:
        info = info or {}
        return bool(info.get("is_key_in_jar")) and bool(info.get("has_jar"))


class CCEnvDerustToModerate(DiscoveryWorldEnv):
    """DiscoveryWorld variant that ends once the key is moderately rusted or better."""

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        text_obs, info = super().reset(kwargs=kwargs)
        self._prepare_key_in_carried_jar()
        self._last_action_result = None

        text_obs, info = self._format_obs_and_info()
        self._update_state_from_info(info)
        info["won"] = bool(self._is_task_complete(info))
        self._last_info = deepcopy(info)
        self._prev_score = float(info.get("score_normalized", 0.0))
        return text_obs, info

    def _prepare_key_in_carried_jar(self) -> None:
        self._skill_runner._ensure_key_and_jar_ready()
        self._skill_runner.update_ui_and_location()
        _, has_jar, is_key_in_jar, _, _ = extract_detailed_status(self._skill_runner.ui)
        if is_key_in_jar and not has_jar:
            self._skill_runner.move_to_jar()
            self._skill_runner.pick_up_jar()

    def _is_task_complete(self, info: Optional[Dict[str, Any]] = None) -> bool:
        info = info or {}
        rust_status = str(info.get("key_rust_status") or "").strip().lower()
        rust_level = RUST_LABEL_TO_LEVEL.get(rust_status)
        return rust_level is not None and rust_level <= RUST_LABEL_TO_LEVEL["moderately rusted"]
