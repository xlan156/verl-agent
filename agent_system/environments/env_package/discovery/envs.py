from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import time
from collections import defaultdict
import ray
import torch
import math

from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import(
    DiscoveryWorldAPI,
)
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistryEasySkill


def _slugify(value: Optional[str]) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return text.strip("-") or "unknown"


def _build_frames_dir(env_kwargs: Dict[str, Any], seed: int, is_train: bool) -> str:
    scenario = _slugify(env_kwargs.get("scenario_name"))
    difficulty = _slugify(env_kwargs.get("difficulty"))
    model_name = env_kwargs.get("model_name") or os.environ.get("MODEL_NAME")
    job_id = _slugify(os.environ.get("SLURM_JOB_ID"))
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    split = "train" if is_train else "eval"
    return os.path.join(
        "outputs",
        "discoveryworld_frames",
        f"{model_name}__seed{seed}__{job_id}__{timestamp}__{split}",
    )


def _strip_uuid(text: str) -> str:
    if not text:
        return text
    return re.sub(r"\s*\[uuid:\s*[^\]]+\]", "", text).strip()


def _extract_action_and_meta(action: Any) -> Tuple[Optional[str], Dict[str, Any]]:
    """Extract action string plus optional metadata.

    Supports JSON string payloads like:
      {"action": "move_to_key", "__meta": {"has_multiple_skills": true}}
    """

    meta: Dict[str, Any] = {}

    if isinstance(action, dict):
        raw_meta = action.get("__meta")
        if isinstance(raw_meta, dict):
            meta = dict(raw_meta)
        inner = (
            action.get("skill")
            or action.get("action")
            or action.get("raw_action")
            or action.get("raw")
        )
        return (inner if isinstance(inner, str) else None), meta

    if isinstance(action, str):
        stripped = action.lstrip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(action)
            except Exception:
                return action, {}
            if isinstance(payload, dict):
                raw_meta = payload.get("__meta")
                if isinstance(raw_meta, dict):
                    meta = dict(raw_meta)
                inner = (
                    payload.get("skill")
                    or payload.get("action")
                    or payload.get("raw_action")
                    or payload.get("raw")
                )
                return (inner if isinstance(inner, str) else None), meta
        return action, {}

    return None, {}


RUSTED_KEY = "rusted key (heavily rusted)"
JAR = "jar"
IGNORED_OBJECT_NAMES = {"floor", "wall", "grass", "path", "table"}
INVALID_MESSAGE = "Invalid action or argument"
SKILL_TO_CATEGORY = {
    "move_to_key": "MOVE",
    "move_to_jar": "MOVE",
    "move_to_dispensers_A": "MOVE",
    "move_to_dispensers_B": "MOVE",
    "move_to_dispensers_C": "MOVE",
    "move_to_dispensers_D": "MOVE",
    "pick_up_key": "PICKUP_KEY",
    "put_key_in_jar": "PUT",
    "pick_up_jar": "PICKUP_JAR",
    "use_dispenser_A_on_jar": "USE",
    "use_dispenser_B_on_jar": "USE",
    "use_dispenser_C_on_jar": "USE",
    "use_dispenser_D_on_jar": "USE",
    "wash_jar": "USE",
    "open_door": "OPEN",
}


class DiscoveryWorldEnv:

    def __init__(
        self,
        seed: int,
        scenario_name: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_steps: int = 50,
        thread_id: int = 0,
        save_frames: bool = False,
        frames_dir: Optional[str] = None,
    ) -> None:
        self._seed = seed
        self._scenario_name = scenario_name
        self._difficulty = difficulty
        self._max_steps = max_steps
        self._thread_id = thread_id
        self._save_frames = bool(save_frames)
        self._frames_dir = frames_dir

        self._api: Optional[DiscoveryWorldAPI] = None
        self._steps: int = 0
        self._prev_score: float = 0.0
        self._last_action_result: Optional[Dict[str, Any]] = None
        
        # reward shaping
        self.init_reward_shaping()

    def _init_api(self) -> None:
        self._api = DiscoveryWorldAPI(threadID=self._thread_id)
        self._api.save_frames = self._save_frames
        if self._frames_dir:
            self._api.FRAME_DIR = os.path.join(self._frames_dir, f"thread-{self._thread_id}")
        ok = self._api.loadScenario(
            scenarioName=self._scenario_name,
            difficultyStr=self._difficulty,
            randomSeed=self._seed,
            numUserAgents=1,
        )
        if not ok:
            raise RuntimeError(
                f"Failed to load DiscoveryWorld scenario='{self._scenario_name}' "
                f"difficulty='{self._difficulty}'"
            )
    
    def init_reward_shaping(self):
        self.action_history: List[Optional[str]] = []
        self.action_counter = defaultdict(int)
        for action in {"MOVE", "PICKUP", "OPEN", "USE", "PUT"}:
            self.action_counter[action] = 0
        
        self.object_seen: Dict[str, str] = {}
        self._object_name_counts = defaultdict(int)
        self.location_history: List[Tuple[Optional[int], Optional[int]]] = []
        self._skill_runner: Optional[CombinatorialChemistryEasySkill] = None
        
        self.teacher = RulebasedAgentSkill(self)
        self._last_teacher_skill: Optional[str] = None
        self.has_key = False
        self.has_jar = False
        self.is_key_in_jar = False
        self.used_dispensers = {
            "A": False,
            "B": False,
            "C": False,
            "D": False,
        }

    def _record_object_seen(self, name: str, uuid: Any) -> None:
        """Record an object by name with unique suffixes for duplicates."""
        if not name:
            return

        name = str(name)
        uuid_str = str(uuid)

        if name not in self.object_seen:
            self.object_seen[name] = uuid_str
            self._object_name_counts[name] = 1
            return

        # Skip if this UUID is already recorded under any name variant
        if uuid_str in self.object_seen.values():
            return

        base = name
        count = self._object_name_counts.get(base, 1) + 1
        key = f"{base}{count}"
        while key in self.object_seen:
            count += 1
            key = f"{base}{count}"

        self.object_seen[key] = uuid_str
        self._object_name_counts[base] = count

    def _score_normalized(self) -> float:
        assert self._api is not None
        scorecard = self._api.getTaskScorecard() or []
        if not scorecard:
            return 0.0
        return float(scorecard[0].get("scoreNormalized", 0.0))

    def _format_obs_and_info(self) -> Tuple[str, Dict[str, Any]]:
        assert self._api is not None
        observation = self._api.getAgentObservation(agentIdx=0)
        ui = observation.get("ui", {})

        text_obs = json.dumps(self.compress_ui_observation(ui), indent=2, sort_keys=True)

        task_desc = ui.get("taskProgress")[0].get("description")

        info: Dict[str, Any] = {
            "raw_observation": observation,
            "task_description": task_desc,
            "teleport_locations": self._api.listTeleportLocationsDict(),
            "last_action_result": self._last_action_result,
            "score_normalized": self._score_normalized(),
        }

        info["won"] = bool(self._api.areTasksComplete())
        return text_obs, info

    def _update_object_seen_from_ui(self, ui: Dict[str, Any]) -> None:
        for obj in ui.get("inventoryObjects", []) or []:
            if obj.get("name") not in IGNORED_OBJECT_NAMES:
                self._record_object_seen(obj.get("name"), obj.get("uuid"))
        for obj in ui.get("accessibleEnvironmentObjects", []) or []:
            if obj.get("name") not in IGNORED_OBJECT_NAMES:
                self._record_object_seen(obj.get("name"), obj.get("uuid"))
        for direction, objects in ui.get("nearbyObjects", {}).get("objects", {}).items():
            for obj in objects or []:
                if obj.get("name") not in IGNORED_OBJECT_NAMES:
                    self._record_object_seen(obj.get("name"), obj.get("uuid"))
    
    def _update_location_history(self, ui: Dict[str, Any]) -> None:
        location = (ui.get("agentLocation", {}).get("x"), ui.get("agentLocation", {}).get("y"))
        self.location_history.append(location)
    
    def _update_key_jar_status(self, info: Dict[str, Any]) -> None:
        inventory = info.get("raw_observation", {}).get("ui", {}).get("inventoryObjects", [])
        accessible = info.get("raw_observation", {}).get("ui", {}).get("accessibleEnvironmentObjects", [])
        for obj in inventory + accessible:
            if obj.get("name") == RUSTED_KEY:
                description = obj.get("description", "")
                if "in jar" in description:
                    self.is_key_in_jar = True
                else:
                    self.is_key_in_jar = False
        
        inv_objects = {obj.get("name"): obj for obj in inventory or []}
        accessible_objects = {obj.get("name"): obj for obj in accessible or []}
        self.has_key = RUSTED_KEY in inv_objects
        self.has_jar = JAR in inv_objects

    def _update_state_from_info(self, info: Dict[str, Any]) -> None:
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_object_seen_from_ui(ui)
        self._update_location_history(ui)
        self._update_key_jar_status(info)

    def _finalize_info(self, info: Dict[str, Any], is_valid: Optional[bool] = None) -> None:
        info["object_seen"] = dict(self.object_seen)
        if is_valid is not None:
            info["is_valid"] = int(is_valid)

    @staticmethod
    def compress_ui_observation(ui_obs: dict) -> str:
        """
        Compress UI observation from ~4000 tokens to <500 tokens.
        Convert verbose UI JSON into a compact structural text representation.
        """
        lines = []
        
        # 1. Agent Location (concise)
        loc = ui_obs.get("agentLocation", {})
        if loc:
            facing = loc.get("faceDirection", "unknown")
            can_move = ", ".join(loc.get("directions_you_can_move", []))
            blocked = ", ".join(loc.get("directions_blocked", []))
            lines.append(f"Location: ({loc.get('x', '?')}, {loc.get('y', '?')}), facing {facing}")
            if can_move:
                lines.append(f"Can move: {can_move}")
            if blocked:
                lines.append(f"Blocked: {blocked}")
        
        # 2. Inventory
        inventory = ui_obs.get("inventoryObjects", [])
        if inventory:
            items = [_strip_uuid(f"{obj.get('description', '')}")
                     for obj in inventory if obj.get("name") not in IGNORED_OBJECT_NAMES]
            lines.append(f"Inventory: {', '.join(items)}")
        else:
            lines.append("Inventory: empty")
        
        # 3. Accessible Objects
        accessible = ui_obs.get("accessibleEnvironmentObjects", [])
        accessible_objects = [_strip_uuid(f"{obj.get('description', '')}")
                              for obj in accessible if obj.get("name") not in IGNORED_OBJECT_NAMES]
        if accessible_objects:
            lines.append(f"Accessible: {', '.join(accessible_objects)}")
        else:
            lines.append("Accessible: no object is accessible in current location and facing direction")
        
        # 4. Nearby Objects (only interesting objects within certain steps, grouped by direction)
        nearby = ui_obs.get("nearbyObjects", {}).get("objects", {})
        lines.append("Nearby objects:")
        for direction, objects in nearby.items():
            for obj in objects:
                distance = obj.get("distance", 99)
                if distance <= 2 and obj.get("name") not in IGNORED_OBJECT_NAMES:
                    desc = _strip_uuid(f"{obj.get('description', '')}")
                    lines.append(f"- {direction} ({distance} tile(s) away): {desc}")
        
        # 5. Nearby Agents (only if non-empty and has actions)
        nearby_agents = ui_obs.get("nearbyAgents", {}).get("list_of_agents", {})
        if nearby_agents:
            agent_names = [name for name, actions in nearby_agents.items() if actions]
            if agent_names:
                lines.append(f"Agents nearby: {', '.join(agent_names)}")
        
        # 6. Discovery Feed (only recent non-trivial posts)
        feed = ui_obs.get("discoveryFeed", {})
        posts = feed.get("posts", [])
        articles = feed.get("scientific_articles", [])
        
        if len(posts) > 1:  # More than just welcome message
            recent_posts = [f"{p.get('author', 'Unknown')}: {p.get('content', '')}" 
                        for p in posts[-3:]]  # Last 3 posts
            lines.append(f"\nRecent posts: {'; '.join(recent_posts)}")
        
        if articles:
            lines.append(f"Scientific articles available: {len(articles)}")
        
        # 7. Dialog (only if active)
        dialog = ui_obs.get("dialog_box", {})
        if dialog.get("is_in_dialog", False):
            lines.append("\nIN DIALOG")
        
        # 8. Action messages (only if non-empty)
        last_msg = ui_obs.get("lastActionMessage", "")
        extended_msg = ui_obs.get("extended_action_message", "")
        if last_msg:
            lines.append(f"\nLast action: {last_msg}")
        if extended_msg:
            lines.append(f"Extended info: {extended_msg}")
        
        # 9. Task Progress (concise)
        task_progress = ui_obs.get("taskProgress", [])[0]
        success = task_progress.get("completed", False)
        lines.append(f"\nTask completed: {success}")
        
        return "\n".join(lines)

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        self._init_api()
        self._steps = 0
        self._prev_score = 0.0
        self._last_action_result = None
        self.init_reward_shaping()
        self._skill_runner = CombinatorialChemistryEasySkill(self)

        text_obs, info = self._format_obs_and_info()
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_object_seen_from_ui(ui)
        self._finalize_info(info)
        self._last_info = info
        self._prev_score = float(info.get("score_normalized", 0.0))
        
        if not self.location_history:
            self._update_location_history(ui)
        return text_obs, info

    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        assert self._api is not None

        action_text, action_meta = _extract_action_and_meta(action)
        has_multiple_skills = bool(action_meta.get("has_multiple_skills", False))

        skill_name = action_text if isinstance(action_text, str) and action_text in SKILL_TO_CATEGORY else None
        is_valid = bool(skill_name)

        if not is_valid:
            self._last_action_result = {"success": False, "message": INVALID_MESSAGE}
            self.action_history.append("INVALID")
            self._steps += 1

            text_obs, info = self._format_obs_and_info()
            info["has_multiple_skills"] = has_multiple_skills
            self._update_state_from_info(info)
            self._finalize_info(info, is_valid=False)
            self._last_info = info

            done = bool(self._api.areTasksComplete() or self._steps >= self._max_steps)
            info["won"] = bool(self._api.areTasksComplete())
            return text_obs, -1.0, done, info

        if self._skill_runner is None:
            self._skill_runner = CombinatorialChemistryEasySkill(self)

        try:
            skill_fn = self._skill_runner.skill_mapping.get(skill_name)
            if skill_fn is None:
                raise ValueError(f"Unknown skill: {skill_name}")
            skill_fn()
            self.action_history.append(skill_name)
        except Exception as e:
            self._last_action_result = {"success": False, "message": "Invalid action"}
            self.action_history.append("INVALID")
            is_valid = False

        self._steps += 1

        # reward shaping
        text_obs, info = self._format_obs_and_info()
        info["has_multiple_skills"] = has_multiple_skills
        self._update_state_from_info(info)
        self._finalize_info(info, is_valid=is_valid)

        reward, done, reward_info = self._compute_step_reward(skill_name, info)
        
        self._last_info = info
        print("rewards:")
        print(json.dumps(reward_info, indent=2))
        return text_obs, reward, done, info

    def close(self) -> None:
        return None

    def _game_progress_reward(self, cur_score: float) -> float:
        reward = cur_score - self._prev_score
        self._prev_score = cur_score
        if reward > 0:
            return 0.5
        else:
            return 0.0
    
    def _teacher_skill_reward(self, skill_name: Optional[str]) -> float:
        teacher_skill = self.teacher.select_skill(self._last_info)
        self._last_teacher_skill = teacher_skill
        if not teacher_skill:
            with open("teacher_skill.txt", "a") as f:
                ui = (self._last_info.get("raw_observation") or {}).get("ui", {})
                compressed_obs = self.compress_ui_observation(ui)
                f.write(f"Observation:\n{compressed_obs}\n\n")
            return 0.0
        
        print(f" - Teacher skill: {teacher_skill}, Agent skill: {skill_name}")
        if skill_name == teacher_skill:
            skill_cnt = self.teacher.skill_counter[teacher_skill]
            decay = math.exp(- 0.2 * skill_cnt)
            return 1.0 * decay
        else:
            return -0.5

    def _compute_stalling_penalty(self, skill_name: Optional[str]) -> float:
        penalty = 0.0
        if skill_name in {"move_to_key", "move_to_jar", "move_to_dispensers_A", "move_to_dispensers_B", "move_to_dispensers_C", "move_to_dispensers_D"}:
            if len(self.location_history) >= 3 and set(self.location_history[-3:]) == {self.location_history[-1]}:
                penalty -= 0.2
        if len(self.location_history) >= 5 and set(self.location_history[-5:]) == {self.location_history[-1]}:
            penalty -= 0.2
        return penalty

    def _compute_repetition_penalty(self) -> float:
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            return -1.0
        return 0.0
    
    def _recent_action_entropy(self):
        if not self.action_history:
            return 0.0
        
        n = len(self.action_history)
        n = max(n, 6)
        counts = defaultdict(int)
        for action in self.action_history[-n:]:
            counts[action] += 1
        probs = torch.tensor([count / n for count in counts.values()])
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        max_entropy = torch.log(torch.tensor(len(counts), dtype=torch.float))
        
        decay = max(0.0, 1.0 - self._steps / self._max_steps)
        norm_entropy = entropy / (max_entropy + 1e-8) * decay
        
        return norm_entropy.item()
    
    def _compute_moving_error_penalty(self, skill_name: Optional[str], result: Optional[Dict[str, Any]]) -> float:
        if skill_name and skill_name.startswith("move_to"):
            if result and not result.get("success"):
                return -0.1
        return 0.0
    
    def _rare_action_reward(self, skill_name: Optional[str]) -> float:
        category = SKILL_TO_CATEGORY.get(skill_name or "")
        if not category:
            return 0.0
        reward = 0.5 if self.action_counter[category] == 0 else 0.0
        self.action_counter[category] += 1
        return reward
    
    def _invalid_action_penalty(self, is_valid: bool) -> float:
        penalty = 0.0
        if not is_valid:
            penalty = -0.5
        if self._last_action_result.get("success") is False:
            penalty -= 0.5
        return penalty
    
    def _multiple_skills_penalty(self, has_multiple_skills: bool) -> float:
        if has_multiple_skills:
            return -0.5
        return 0.0
    
    def _subgoal_switching_reward(self, skill_name: Optional[str]) -> float:
        reward = 0.0
        if self.has_key and not self.has_jar:
            if skill_name in {"move_to_key", "pick_up_key"}:
                reward = -2.0
        
        if self.has_jar and not self.has_key:
            if skill_name in {"move_to_jar", "pick_up_jar"}:
                reward = -1.0
        
        if self.has_jar and not self.is_key_in_jar:
            if skill_name in {"move_to_jar", "pick_up_jar"}:
                reward = -1.0
            
        if self.is_key_in_jar:
            if skill_name in {"put_key_in_jar"}:
                reward = -1.0
                
        if not self.has_key and not self.has_jar and self.is_key_in_jar:
            if skill_name in {"put_key_in_jar", "pick_up_key"}:
                reward = -2.0
        
        return reward
    
    def _stage_reward(self) -> float:
        reward = 0.0
        if self.has_key:
            reward += 0.2
        
        if self.has_jar:
            reward += 0.2
        
        if self.is_key_in_jar:
            reward += 0.2
        
        used_any_dispenser = self.used_dispensers["A"] or self.used_dispensers["B"] or self.used_dispensers["C"] or self.used_dispensers["D"]
        if not self.has_jar and used_any_dispenser:
            reward -= 0.2
        
        elif self.has_jar and self.used_dispensers["B"]:
            reward += 0.2
        
        used_other_dispensers = any(self.used_dispensers[d] for d in ["A", "C", "D"])
        if self.has_jar and used_other_dispensers:
            reward += 0.1
            
        return reward

    def clip_reward(self, reward, upper, lower):
        clipped = max(lower, min(upper, reward))
        return clipped
    
    def _compute_step_reward(
        self,
        skill_name: Optional[str],
        info: Dict[str, Any],
    ) -> Tuple[float, bool, Dict[str, float]]:
        cur_score = float(info.get("score_normalized", 0.0))
        game_progress_reward = self._game_progress_reward(cur_score)
        teacher_skill_reward = self._teacher_skill_reward(skill_name)
        rare_action_reward = self._rare_action_reward(skill_name)
        stage_reward = self._stage_reward()
        subgoal_switching_reward = self._subgoal_switching_reward(skill_name)
        
        multiple_skills_penalty = self._multiple_skills_penalty(bool(info.get("has_multiple_skills", False)))
        stalling_penalty = self._compute_stalling_penalty(skill_name)
        repetition_penalty = self._compute_repetition_penalty()
        hit_wall_penalty = self._compute_moving_error_penalty(skill_name, self._last_action_result)
        invalid_penalty = self._invalid_action_penalty(bool(info.get("is_valid", 0)))
        info["teacher_skill"] = self._last_teacher_skill

        done = bool(self._api.areTasksComplete() or self._steps >= self._max_steps)
        info["won"] = bool(self._api.areTasksComplete())
        won_reward = 30.0 if info["won"] else 0.0

        time_penalty = -0.02  # Small penalty to encourage faster solutions
        reward_terms = {
            "game_progress_reward": (10.0, game_progress_reward),
            "rare_action_reward": (0.0, rare_action_reward),
            "teacher_skill_reward": (1.5, teacher_skill_reward),
            "stage_reward": (1.0, stage_reward),
            "multiple_skills_penalty": (0.8, multiple_skills_penalty),
            "repetition_penalty": (0.2, repetition_penalty),
            "stalling_penalty": (0.25, stalling_penalty),
            "subgoal_switching_reward": (0.25, subgoal_switching_reward),
            "invalid_penalty": (0.8, invalid_penalty),
        }
        reward = won_reward + sum(weight * value for weight, value in reward_terms.values())
        
        if not done:
            reward = self.clip_reward(reward, upper=6.0, lower=-1.5)

        reward_info = {
            "step_reward": reward,
            "game_progress_reward": game_progress_reward,
            "teacher_skill_reward": teacher_skill_reward,
            "rare_action_reward": rare_action_reward,
            "multiple_skills_penalty": multiple_skills_penalty,
            "repetition_penalty": repetition_penalty,
            "stalling_penalty": stalling_penalty,
            "invalid_penalty": invalid_penalty,
        }

        return reward, done, reward_info


class DiscoveryWorldWorker:
    """Ray remote worker that wraps a single DiscoveryWorldEnv instance."""

    def __init__(
        self,
        seed: int,
        env_kwargs: Optional[Dict[str, Any]] = None,
        thread_id: int = 0,
    ) -> None:
        env_kwargs = env_kwargs or {}
        scenario_name = env_kwargs.get("scenario_name")
        difficulty = env_kwargs.get("difficulty")
        max_steps = int(env_kwargs.get("max_steps", 50))
        save_frames = bool(env_kwargs.get("save_frames", False))
        frames_dir = env_kwargs.get("frames_dir")

        self._env = DiscoveryWorldEnv(
            seed=seed,
            scenario_name=scenario_name,
            difficulty=difficulty,
            max_steps=max_steps,
            thread_id=thread_id,
            save_frames=save_frames,
            frames_dir=frames_dir,
        )

    def reset(self) -> Tuple[str, Dict[str, Any]]:
        return self._env.reset()

    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()


class DiscoveryWorldVectorEnv:

    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        is_train: bool,
        env_kwargs: Optional[Dict[str, Any]] = None,
        resources_per_worker: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Allow env_num to be None (e.g. when val_batch_size is None)
        # In that case we simply create zero environments and return
        self.env_num = int(env_num) if env_num is not None else 0
        self.group_n = int(group_n)
        self.num_processes = self.env_num * self.group_n
        self.is_train = is_train

        self._workers: List[Any] = []

        if self.num_processes == 0:
            # No envs to build (e.g. no validation envs configured)
            return

        if resources_per_worker is None:
            # Reasonable default: light CPU-only envs
            resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0}

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        env_kwargs = env_kwargs or {}
        if "save_frames" not in env_kwargs:
            env_kwargs["save_frames"] = (not is_train)
        if env_kwargs.get("save_frames") and "frames_dir" not in env_kwargs:
            env_kwargs["frames_dir"] = _build_frames_dir(env_kwargs, seed, is_train)

        env_worker = ray.remote(**resources_per_worker)(DiscoveryWorldWorker)
        print(f"DEBUG: Starting to launch {self.num_processes} Ray actors...", flush=True)
        for i in range(self.num_processes):
            # Share seed across group_n replicas
            worker_seed = seed + (i // self.group_n)
            worker = env_worker.remote(worker_seed, env_kwargs, i)
            self._workers.append(worker)


    def reset(self) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.num_processes == 0:
            return [], []

        obs_list: List[str] = []
        info_list: List[Dict[str, Any]] = []

        futures = [worker.reset.remote() for worker in self._workers]
        print(f"DEBUG: Waiting for {len(futures)} workers to reset via ray.get...", flush=True)
        results = ray.get(futures)
        print("DEBUG: All workers reset successfully.", flush=True)

        for obs, info in results:
            self._append_result(obs_list, info_list, obs, info)

        return obs_list, info_list

    def step(self, actions: List[Any]) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
        if self.num_processes == 0:
            if len(actions) not in (0, None):
                raise ValueError(
                    f"No environments available but got {len(actions)} actions.",
                )
            return [], [], [], []

        if len(actions) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} actions, got {len(actions)}",
            )

        obs_list: List[str] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        futures = []
        for worker, act in zip(self._workers, actions):
            future = worker.step.remote(act)
            futures.append(future)

        results = ray.get(futures)
        for obs, rew, done, info in results:
            self._append_result(obs_list, info_list, obs, info)
            reward_list.append(float(rew))
            done_list.append(bool(done))

        return obs_list, reward_list, done_list, info_list

    def close(self) -> None:
        if not getattr(self, "_workers", None):
            return

        # Gracefully close all remote workers
        close_futures = []
        for worker in self._workers:
            close_futures.append(worker.close.remote())
        ray.get(close_futures)

        # Then kill the actors
        for worker in self._workers:
            ray.kill(worker)

    @staticmethod
    def _append_result(
        obs_list: List[str],
        info_list: List[Dict[str, Any]],
        obs: str,
        info: Optional[Dict[str, Any]],
    ) -> None:
        normalized = dict(info or {})
        normalized.setdefault("won", False)
        obs_list.append(obs)
        info_list.append(normalized)


def build_discoveryworld_envs(
    seed: int,
    env_num: int,
    group_n: int,
    is_train: bool,
    env_kwargs: Optional[Dict[str, Any]] = None,
    resources_per_worker: Optional[Dict[str, Any]] = None,
) -> DiscoveryWorldVectorEnv:

    return DiscoveryWorldVectorEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        is_train=is_train,
        env_kwargs=env_kwargs,
        resources_per_worker=resources_per_worker,
    )
