from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional
import json
from collections import Counter, defaultdict, deque

import numpy as np
import ray

from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import(
    DiscoveryWorldAPI,
)
from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.ScenarioMaker import(
    SCENARIOS,
    SCENARIO_DIFFICULTY_OPTIONS,
)


class DiscoveryWorldEnv:

    def __init__(
        self,
        seed: int,
        scenario_name: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_steps: int = 50,
        thread_id: int = 0,
    ) -> None:
        self._seed = seed
        self._scenario_name = scenario_name
        self._difficulty = difficulty
        self._max_steps = max_steps
        self._thread_id = thread_id

        self._api: Optional[DiscoveryWorldAPI] = None
        self._steps: int = 0
        self._prev_score: float = 0.0
        self._last_action_result: Optional[Dict[str, Any]] = None
        
        self.recent_history = {
            "observations": deque(maxlen=3),
            "actions": deque(maxlen=3),
            }
        
        # reward shaping
        self.init_reward_shaping()

    def _init_api(self) -> None:

        self._api = DiscoveryWorldAPI(threadID=self._thread_id)
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
        self.action_history: List[Dict[str, Any]] = []
        self.alpha = 10.0
        self.beta = 1.0
        self.action_counter = defaultdict(int)
        self.object_seen = defaultdict(str)
        self._object_name_counts = defaultdict(int)

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
            if obj.get("name") not in ["floor", "wall", "grass", "path"]:
                self._record_object_seen(obj.get("name"), obj.get("uuid"))
        for obj in ui.get("accessibleEnvironmentObjects", []) or []:
            if obj.get("name") not in ["floor", "wall", "grass", "path"]:
                self._record_object_seen(obj.get("name"), obj.get("uuid"))
        for direction, objects in ui.get("nearbyObjects", {}).get("objects", {}).items():
            for obj in objects or []:
                if obj.get("name") not in ["floor", "wall", "grass", "path"]:
                    self._record_object_seen(obj.get("name"), obj.get("uuid"))

    @staticmethod
    def compress_ui_observation(ui_obs: dict) -> str:
        """
        Compress UI observation from ~4000 tokens to <500 tokens.
        Converts verbose JSON to concise natural language summary while preserving UUIDs.
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
        
        # 2. Inventory (only if non-empty)
        inventory = ui_obs.get("inventoryObjects", [])
        if inventory:
            items = [f"{obj.get('name', 'unknown')}" for obj in inventory if obj.get("name") not in ["floor", "wall", "grass", "path"]]
            lines.append(f"Inventory: {', '.join(items)}")
        else:
            lines.append("Inventory: empty")
        
        # 3. Accessible Objects
        accessible = ui_obs.get("accessibleEnvironmentObjects", [])
        accessible_objects = [f"{obj.get('name', 'unknown')}" for obj in accessible if obj.get("name") not in ["floor", "wall", "grass", "path"]]
        if accessible_objects:
            lines.append(f"Accessible: {', '.join(accessible_objects)}")
        else:
            lines.append("Accessible: no object is accessible in current location and facing direction")
        
        # 4. Nearby Objects (only interesting objects within certain steps, grouped by direction)
        nearby = ui_obs.get("nearbyObjects", {}).get("objects", {})
        nearby_summary = {}
        
        for direction, objects in nearby.items():
            # Only include objects within distance 1 and skip floor/wall/grass
            nearby_objects = []
            for obj in objects:
                distance = obj.get("distance", 99)
                if distance <= 2 and obj.get("name") not in ["floor", "wall", "grass", "path"] and direction in ["north", "south", "east", "west"]:
                    nearby_objects.append(f"{obj.get('name', 'unknown')} (distance={distance+1})")
            if nearby_objects:
                nearby_summary[direction] = nearby_objects
        
        if nearby_summary:
            # Keep concise while exposing compass directions
            lines.append("\nNearby (at most 2 tiles away):")
            for direction, objects in sorted(nearby_summary.items()):
                lines.append(f"{direction}: {', '.join(objects)}")
        
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

        text_obs, info = self._format_obs_and_info()
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_object_seen_from_ui(ui)
        info["object_seen"] = dict(self.object_seen)
        self._prev_score = float(info.get("score_normalized", 0.0))
        return text_obs, info

    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        assert self._api is not None

        if isinstance(action, str):
            try:
                action_json = json.loads(action)
            except Exception:
                # 模型输出非法 JSON 时，给一个空 action，让环境自行报错
                action_json = {}
        elif isinstance(action, dict):
            action_json = action
        else:
            raise ValueError(f"Unsupported action type: {type(action)}")

        # Metadata injected by the projection layer (see DiscoveryWorldEnvironmentManager.step).
        # Use it for reward shaping, but strip before calling the underlying API.
        meta = {}
        if isinstance(action_json, dict):
            meta = action_json.pop("__meta", {}) or {}
        contain_think_block = int(meta.get("contain_think_block", 0)) if isinstance(meta, dict) else 0
        are_json_format = int(meta.get("are_json_format", 0)) if isinstance(meta, dict) else 0

        # Pre-check the structured action for obviously invalid arguments
        invalid_arg_penalty = self._compute_invalid_action_penalty(action_json)

        result = self._api.performAgentAction(agentIdx=0, actionJSON=action_json)
        self._last_action_result = result
        self.action_history.append(action_json.get("action"))
        self._api.tick()
        self._steps += 1

        # reward shaping
        text_obs, info = self._format_obs_and_info()
        info["contain_think_block"] = contain_think_block
        info["are_json_format"] = are_json_format
        
        # Bonus for encountering new objects
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_object_seen_from_ui(ui)
        info["object_seen"] = dict(self.object_seen)
                
        bonus_exploring = self.add_bonus_for_exploring(action_json.get("action", ""), info)
        
        # increment
        cur_score = float(info.get("score_normalized", 0.0))
        ingame_process_reward = cur_score - self._prev_score
        self._prev_score = cur_score
        
        # format reward
        if contain_think_block and are_json_format:
            format_reward = 1
        else:            
            format_reward = 0
            
        
        # error penalty
        penalty_for_error = self.penalize_error_action(result)

        done = bool(self._api.areTasksComplete() or self._steps >= self._max_steps)
        info["won"] = bool(self._api.areTasksComplete())
        won_reward = 100.0 if info["won"] else 0.0
        
        reward = (
            + 0.3 * penalty_for_error
            + 0.2 * format_reward
            + 0.5 * (ingame_process_reward + won_reward)
        )

        return text_obs, reward, done, info

    def close(self) -> None:
        return None
    
    def add_bonus_for_exploring(self, action_str: str, info: Dict[str, Any]) -> float:
        """Return a small bonus reward for taking the first action, to encourage shorter solutions."""
        if self._steps <= 10 and action_str in {"MOVE_DIRECTION", "ROTATE_DIRECTION"}:
            return 0.1
        return 0.0
    
    def penalize_error_action(self, result: Dict[str, Any]) -> float:
        """Return a penalty if the last action resulted in an error, to encourage valid actions."""
        if len(result.get("errors", [])) > 0:
            return 0
        return 1

    def _compute_invalid_action_penalty(self, action_json: Dict[str, Any]) -> float:
        """Return an extra penalty when the agent uses invalid directions or UUIDs.

        - MOVE_DIRECTION / ROTATE_DIRECTION must use one of {north, east, south, west}.
        - For actions whose arguments are expected to be objects/agents, all arg* values
          must be among the currently interactable object UUIDs.
        """

        if self._api is None:
            return 0.0

        if not isinstance(action_json, dict) or not action_json:
            # Completely malformed or empty action
            return -0.05

        action_name = action_json.get("action", "")
        if not action_name:
            return -0.05

        # Direction-only actions: check that the direction is valid
        if action_name in {"MOVE_DIRECTION", "ROTATE_DIRECTION"}:
            direction = str(action_json.get("arg1", "")).lower()
            if direction not in {"north", "south", "east", "west"}:
                return -0.05
            return 0.0

        # Actions that do not take UUID/object arguments: skip UUID checks
        if action_name in {"TELEPORT_TO_LOCATION", "DISCOVERY_FEED_GET_UPDATES", "DISCOVERY_FEED_GET_POST_BY_ID"}:
            return 0.0

        # Build the current set of interactable UUIDs from the UI observation
        try:
            observation = self._api.getAgentObservation(agentIdx=0) or {}
            ui = observation.get("ui", {}) or {}
        except Exception:
            return 0.0

        valid_uuids_inventory = set()
        valid_uuids_accessible = set()

        for obj in ui.get("inventoryObjects", []) or []:
            uuid = obj.get("uuid")
            if uuid is not None:
                valid_uuids_inventory.add(str(uuid))
    
        for obj in ui.get("accessibleEnvironmentObjects", []) or []:
            uuid = obj.get("uuid")
            if uuid is not None:
                valid_uuids_accessible.add(str(uuid))

        valid_uuids_nearby = set()
        nearby = ui.get("nearbyObjects", {}).get("objects", {}) or {}
        for objects in nearby.values():
            for obj in objects or []:
                uuid = obj.get("uuid")
                if uuid is not None:
                    valid_uuids_nearby.add(str(uuid))

        # Some actions (e.g., TALK) may use nearby agents as targets
        valid_uuids_agents = set()
        nearby_agents = ui.get("nearbyAgents", {}).get("list_of_agents", {}) or {}
        for agent_info in nearby_agents.values():
            if isinstance(agent_info, dict):
                uuid = agent_info.get("uuid")
                if uuid is not None:
                    valid_uuids_agents.add(str(uuid))

        # For all remaining actions, treat arg* fields as expected object/agent UUIDs.
        # If any provided UUID is not in the interactable set, apply a penalty.
        has_args = False
        all_valid_uuids = valid_uuids_inventory | valid_uuids_accessible | valid_uuids_nearby | valid_uuids_agents
        for key, value in action_json.items():
            if not key.startswith("arg"):
                continue
            has_args = True
            if value is None:
                return -0.05
            if str(value) not in all_valid_uuids:
                return -0.05
            if key == "arg1" and str(value) not in valid_uuids_inventory:
                return -0.05
            if key == "arg2" and str(value) not in valid_uuids_accessible:
                return -0.05

        # If an action that should reference objects has no args at all, penalize
        if not has_args and action_name in {"PICKUP", "DROP", "PUT", "OPEN", "CLOSE", "ACTIVATE", "DEACTIVATE", "TALK", "EAT", "READ", "USE", "TELEPORT_TO_OBJECT"}:
            return -0.05

        return 0.0
        


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

        self._env = DiscoveryWorldEnv(
            seed=seed,
            scenario_name=scenario_name,
            difficulty=difficulty,
            max_steps=max_steps,
            thread_id=thread_id,
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
            ray.init()

        env_kwargs = env_kwargs or {}

        env_worker = ray.remote(**resources_per_worker)(DiscoveryWorldWorker)
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
        results = ray.get(futures)

        for obs, info in results:
            info = dict(info or {})
            info.setdefault("won", False)
            obs_list.append(obs)
            info_list.append(info)

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
            info = dict(info or {})
            info.setdefault("won", False)
            obs_list.append(obs)
            reward_list.append(float(rew))
            done_list.append(bool(done))
            info_list.append(info)

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
