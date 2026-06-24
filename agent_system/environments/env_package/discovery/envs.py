from __future__ import annotations

import json
import logging
import os
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray

from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import(
    DiscoveryWorldAPI,
)
from agent_system.environments.env_package.discovery.curriculum import (
    build_stage_pools,
    normalize_chemical_state,
    plan_curriculum_goal_stages,
    sample_solution_init_state,
    solution_dict_to_state,
)
from agent_system.environments.env_package.discovery.seed import build_fixed_seed_pools_by_amount
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.env_package.discovery.utils import (
    SKILL_NAMES,
    build_frames_dir,
    coerce_max_chemical_n,
    compress_ui_observation,
    extract_detailed_status,
    format_rust_level,
    is_dispenser_skill,
)


logger = logging.getLogger(__name__)


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
        max_chemical_n: Optional[int] = None,
        curriculum_enabled: bool = False,
        curriculum_train_fraction: float = 0.7,
        curriculum_mix_ratios: Tuple[float, float, float] = (0.7, 0.2, 0.1),
        curriculum_seed: Optional[int] = None,
        is_train: bool = True,
    ) -> None:
        self._seed = seed
        self._scenario_name = scenario_name
        self._difficulty = difficulty
        self._max_steps = max_steps
        self._thread_id = thread_id
        self._save_frames = bool(save_frames)
        self._frames_dir = frames_dir
        self._max_chemical_n = int(max_chemical_n) if max_chemical_n is not None else 2
        self._curriculum_enabled = bool(curriculum_enabled)
        self._curriculum_train_fraction = float(curriculum_train_fraction)
        self._curriculum_mix_ratios = tuple(curriculum_mix_ratios)
        self._curriculum_seed = int(curriculum_seed) if curriculum_seed is not None else int(seed)
        self._is_train = bool(is_train)
        self._curriculum_state: Optional[Tuple[int, ...]] = None
        self._chemical_solution_state: Optional[Tuple[int, ...]] = None

        self._api: Optional[DiscoveryWorldAPI] = None
        self._steps: int = 0
        self._prev_score: float = 0.0
        self._last_action_result: Optional[Dict[str, Any]] = None
        self.train_epoch: Optional[int] = None

    def _init_api(self) -> None:
        self._api = DiscoveryWorldAPI(threadID=self._thread_id)
        self._api.save_frames = self._save_frames
        if self._frames_dir:
            self._api.FRAME_DIR = os.path.join(self._frames_dir, f"thread-{self._thread_id}")
        
        scenario_args = {
            "numChemicals": 4,
            "minChemicals": 1,
            "chemicalMinAmount": self._max_chemical_n,
            "chemicalMaxAmount": self._max_chemical_n,
        }
        ok = self._api.loadScenario(
            scenarioName=self._scenario_name,
            difficultyStr=self._difficulty,
            randomSeed=self._seed,
            numUserAgents=1,
            **scenario_args,
        )
        if not ok:
            raise RuntimeError(
                f"Failed to load DiscoveryWorld scenario='{self._scenario_name}' "
                f"difficulty='{self._difficulty}'"
            )

    def _get_chemical_solution_state(self) -> Optional[Tuple[int, ...]]:
        assert self._api is not None
        world = getattr(self._api, "world", None)
        if world is None:
            return None

        task_scorer = getattr(world, "taskScorer", None)
        tasks = getattr(task_scorer, "tasks", None) or []
        for task in tasks:
            scoring_info = getattr(task, "scoringInfo", None) or {}
            chemical_solution = scoring_info.get("chemicalSolutionDict")
            if chemical_solution is not None:
                return solution_dict_to_state(chemical_solution, num_chemicals=4)
        return None

    def init_reward_shaping(self):

        self.action_history: List[Optional[str]] = []
        self.location_history: List[Tuple[Optional[int], Optional[int]]] = []

        self._skill_runner = CombinatorialChemistrySkill(self)
        self.teacher = RulebasedAgentSkill(self)
        self._last_teacher_skill: Optional[str] = None
        self._last_info: Optional[Dict[str, Any]] = None

        self.used_dispensers = {"A": False, "B": False, "C": False, "D": False}

    def _score_normalized(self) -> float:
        assert self._api is not None
        scorecard = self._api.getTaskScorecard() or []
        if not scorecard:
            return 0.0
        return float(scorecard[0].get("scoreNormalized", 0.0))

    def _format_obs_and_info(self) -> Tuple[str, Dict[str, Any]]:
        """Create text observation and info dict (without state flags, which are computed separately)."""
        assert self._api is not None
        observation = self._api.getAgentObservation(agentIdx=0)
        ui = observation.get("ui", {})

        text_obs = json.dumps(compress_ui_observation(ui), indent=2, sort_keys=True)

        task_desc = ui.get("taskProgress", [])[0].get("description") if ui.get("taskProgress") else ""

        info: Dict[str, Any] = {
            "raw_observation": observation,
            "task_description": task_desc,
            "last_action_result": self._last_action_result,
            "score_normalized": self._score_normalized(),
            "train_epoch": self.train_epoch,
            "curriculum_state": self._curriculum_state,
            "chemical_solution_state": self._chemical_solution_state,
            "max_chemical_n": self._max_chemical_n,
        }

        info["won"] = bool(self._api.areTasksComplete())
        # include env-level non-UI state
        info["used_dispensers"] = dict(self.used_dispensers)

        return text_obs, info

    def _update_location_history(self, ui: Dict[str, Any]) -> None:
        """Record current location."""
        location = (ui.get("agentLocation", {}).get("x"), ui.get("agentLocation", {}).get("y"))
        self.location_history.append(location)

    def _update_state_from_info(self, info: Dict[str, Any]) -> None:
        """Update location history and compute+inject key/jar state from raw UI."""
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_location_history(ui)

        has_key, has_jar, is_key_in_jar, chemical_dict, key_rust_level = extract_detailed_status(ui)
        info["has_key"] = has_key
        info["has_jar"] = has_jar
        info["is_key_in_jar"] = is_key_in_jar
        info["chemical_dict"] = deepcopy(chemical_dict)
        info["key_rust_level"] = key_rust_level
        info["key_rust_status"] = format_rust_level(key_rust_level)
        info["key_is_rusted"] = (
            None if key_rust_level is None else key_rust_level != "no rust"
        )

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and return initial observation and info."""
        self._init_api()
        self._steps = 0
        self._prev_score = 0.0
        self._last_action_result = None
        self._curriculum_state = None
        
        self.init_reward_shaping()
        self._chemical_solution_state = self._get_chemical_solution_state()

        if kwargs is None:
            reset_kwargs: Dict[str, Any] = {}
        elif isinstance(kwargs, dict):
            reset_kwargs = dict(kwargs)
        else:
            reset_kwargs = {"curriculum_state": kwargs}
        curriculum_state = reset_kwargs.pop("curriculum_state", None)
        if curriculum_state is None:
            curriculum_state = reset_kwargs.pop("chemical_state", None)

        if curriculum_state is not None:
            self._curriculum_state = normalize_chemical_state(curriculum_state)
            saved_used_dispensers = dict(self.used_dispensers)
            self._skill_runner.prepare_chemical_state(self._curriculum_state)
            self.used_dispensers = saved_used_dispensers
            self._last_action_result = None
        elif self._curriculum_enabled and self._chemical_solution_state is not None:
            self._chemical_solution_state = self._get_chemical_solution_state()
            self._curriculum_state = sample_solution_init_state(
                solution_state=self._chemical_solution_state,
                split="train" if self._is_train else "val",
                train_fraction=self._curriculum_train_fraction,
                seed=self._curriculum_seed + self._thread_id,
                num_chemicals=4,
            )
            saved_used_dispensers = dict(self.used_dispensers)
            self._skill_runner.prepare_chemical_state(self._curriculum_state)
            self.used_dispensers = saved_used_dispensers
            self._last_action_result = None

        text_obs, info = self._format_obs_and_info()
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_state_from_info(info)
        
        self._last_info = deepcopy(info)
        self._prev_score = float(info.get("score_normalized", 0.0))
        if not self.location_history:
            self._update_location_history(ui)
        
        return text_obs, info

    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        """Execute one step: validate action, execute if valid, compute reward."""
        assert self._api is not None

        action_text = action
        skill_name = action_text if isinstance(action_text, str) and action_text in SKILL_NAMES else None
        is_valid = bool(skill_name)

        # No skill extracted (``None`` from projection) is an intentional
        # no-op transition.  We still advance the episode so GRPO receives a
        # negative format signal, but never substitute or execute an arbitrary
        # action on the model's behalf.
        if not is_valid:
            self.action_history.append("INVALID")
            self._steps += 1
            self._last_action_result = {
                "success": False,
                "message": "No executable skill was produced",
            }

            text_obs, info = self._format_obs_and_info()
            info["action_status"] = "invalid_no_skill"
            info["executed_action"] = None
            self._update_state_from_info(info)
            
            reward, done = self._compute_step_reward(None, info)
            self._last_info = deepcopy(info)
            return text_obs, reward, done, info

        # Valid action: execute skill
        if self._skill_runner is None:
            self._skill_runner = CombinatorialChemistrySkill(self)

        try:
            skill_fn = self._skill_runner.skill_mapping.get(skill_name)
            if skill_fn is None:
                raise ValueError(f"Unknown skill: {skill_name}")

            skill_fn()
            self.action_history.append(skill_name)
            success = bool((self._last_action_result or {}).get("success", False))
            action_status = "success" if success else "valid_but_failed"

        except Exception:
            logger.exception("DiscoveryWorld skill failed: %s", skill_name)
            self._last_action_result = {"success": False, "message": "Action failed"}
            self.action_history.append(skill_name)
            action_status = "valid_but_failed"

        self._steps += 1

        # Format observation and compute state
        text_obs, info = self._format_obs_and_info()
        info["action_status"] = action_status
        self._update_state_from_info(info)
        
        reward, done = self._compute_step_reward(skill_name, info)
        self._last_info = deepcopy(info)
        
        return text_obs, reward, done, info

    def close(self) -> None:
        return None

    def _game_progress_reward(self, cur_score: float) -> float:
        """Reward based on score increase."""
        return cur_score - self._prev_score
    
    def _teacher_skill_reward(self, skill_name: Optional[str], info: Optional[Dict[str, Any]]) -> float:
        info = info or {}
        teacher_skill = self.teacher.select_skill(self._last_info or info)
        self._last_teacher_skill = teacher_skill
        
        if not teacher_skill:
            ui = (info.get("raw_observation") or {}).get("ui", {})
            logger.debug(
                "Teacher could not select a skill. is_key_in_jar=%s used_dispensers=%s observation=%s",
                info.get("is_key_in_jar", False),
                info.get("used_dispensers", {}),
                compress_ui_observation(ui),
            )
            return 0.0
        
        # In the chemistry phase, the teacher only specifies the action class
        # "add one chemical".  Any concrete dispenser A/B/C/D is a correct
        # teacher action and receives the same shaping reward.
        if is_dispenser_skill(teacher_skill):
            return 1.0 if is_dispenser_skill(skill_name) else 0.0

        if skill_name == teacher_skill:
            return 1.0
        return 0.0

    def _repetition_penalty(self) -> float:
        penalty = 0.0
        if len(self.action_history) >= 3 and len(set(self.action_history[-3:])) == 1:
            penalty = -0.1
        if len(self.action_history) >= 4 and len(set(self.action_history[-4:])) == 1:
            penalty = -0.2
        return penalty
    
    def _invalid_action_penalty(self, info: Dict) -> float:
        """Penalty for invalid or failed actions."""
        action_status = info.get("action_status")
        if action_status == "invalid_no_skill":
            # Keep this smaller than the terminal task reward.  The trainer
            # separately applies is_action_valid's format penalty, so a large
            # duplicate penalty would make early GRPO groups uniformly bad.
            return -0.2
        elif action_status == "valid_but_failed":
            return -0.2
        else:
            return 0.0
    
    def _stage_reward(self, last_info: Dict[str, Any], current_info: Dict[str, Any]) -> float:
        """Reward for progress through key task stages.
        
        Args:
            last_info: Previous step's info (or None on first step)
            current_info: Current step's info
        """
        if last_info is None:
            return 0.0
        
        reward = 0.0
        prev_has_key = last_info.get("has_key", False)
        prev_has_jar = last_info.get("has_jar", False)
        prev_is_key_in_jar = last_info.get("is_key_in_jar", False)
        
        has_key = current_info.get("has_key", False)
        has_jar = current_info.get("has_jar", False)
        is_key_in_jar = current_info.get("is_key_in_jar", False)
        
        if has_key and not prev_has_key:
            reward += 0.2
        if has_jar and not prev_has_jar:
            reward += 0.2
        if is_key_in_jar and not prev_is_key_in_jar:
            reward += 0.4
        return reward
    
    def _no_progress_move_penalty(self, action, cur_score: float, prev_score: float) -> float:
        """Penalty for moving but making no progress."""
        if len(self.location_history) < 4:
            return 0.0

        no_location_change = len(set(self.location_history[-4:])) == 1
        no_score_change = abs(cur_score - prev_score) < 1e-6

        if no_score_change:
            if no_location_change:
                return -0.3
            elif len(self.action_history) >= 2 and action == self.action_history[-2]: # Encourage trying different actions at each step
                return -0.15
            return -0.1
        return 0.0

    def clip_reward(self, reward):
        """Clip reward to reasonable range."""
        clipped = float(np.clip(reward, -1.0, 2.0))
        return clipped
    
    def _compute_step_reward(
        self,
        skill_name: Optional[str],
        info: Dict[str, Any],
    ) -> Tuple[float, bool]:
        cur_score = float(info.get("score_normalized", 0.0))
        prev_score = self._prev_score
        game_progress_reward = self._game_progress_reward(cur_score)
        teacher_skill_reward = self._teacher_skill_reward(skill_name, self._last_info)
        stage_reward = self._stage_reward(self._last_info, info)
        
        repetition_penalty = self._repetition_penalty()
        invalid_penalty = self._invalid_action_penalty(info)
        no_progress_penalty = self._no_progress_move_penalty(skill_name, cur_score, prev_score)
        info["teacher_skill"] = self._last_teacher_skill
        info["reward_components"] = {
            "game_progress": game_progress_reward,
            "teacher_skill": teacher_skill_reward,
            "stage": stage_reward,
            "repetition_penalty": repetition_penalty,
            "invalid_penalty": invalid_penalty,
            "no_progress_penalty": no_progress_penalty,
        }

        task_completed = bool(self._api.areTasksComplete())
        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        reward = 0.0
        # Positive rewards
        reward += 10.0 * game_progress_reward
        reward += teacher_skill_reward
        # reward += stage_reward

        # Penalties
        reward += repetition_penalty
        reward += invalid_penalty
        reward += no_progress_penalty
        
        if not task_completed:
            reward = self.clip_reward(reward)
        else:
            reward += 10.0

        self._prev_score = cur_score
        return reward, done


class DiscoveryWorldWorker:
    """Ray remote worker that wraps a single DiscoveryWorldEnv instance."""

    def __init__(
        self,
        seed: int,
        env_kwargs: Optional[Dict[str, Any]] = None,
        thread_id: int = 0,
    ) -> None:
        self._seed = int(seed)
        self._thread_id = int(thread_id)
        env_kwargs = dict(env_kwargs or {})
        scenario_name = env_kwargs.pop("scenario_name", None)
        difficulty = env_kwargs.pop("difficulty", None)
        max_steps = int(env_kwargs.pop("max_steps", 50))
        save_frames = bool(env_kwargs.pop("save_frames", False))
        frames_dir = env_kwargs.pop("frames_dir", None)
        max_chemical_n = coerce_max_chemical_n(env_kwargs)
        env_kwargs.pop("max_chemical_n", None)
        env_kwargs.pop("max_chemical_N", None)
        env_kwargs.pop("chemical_N", None)
        self._default_reset_kwargs: Dict[str, Any] = {}
        if "curriculum_state" in env_kwargs:
            self._default_reset_kwargs["curriculum_state"] = env_kwargs.pop("curriculum_state")
        curriculum_enabled = bool(env_kwargs.pop("curriculum_enabled", False))
        curriculum_train_fraction = float(env_kwargs.pop("curriculum_train_fraction", 0.7))
        curriculum_mix_ratios = tuple(env_kwargs.pop("curriculum_mix_ratios", (0.7, 0.2, 0.1)))
        curriculum_seed = env_kwargs.pop("curriculum_seed", seed)
        is_train = bool(env_kwargs.pop("is_train", True))

        self._env = DiscoveryWorldEnv(
            seed=seed,
            scenario_name=scenario_name,
            difficulty=difficulty,
            max_steps=max_steps,
            thread_id=thread_id,
            save_frames=save_frames,
            frames_dir=frames_dir,
            max_chemical_n=max_chemical_n,
            curriculum_enabled=curriculum_enabled,
            curriculum_train_fraction=curriculum_train_fraction,
            curriculum_mix_ratios=curriculum_mix_ratios,
            curriculum_seed=curriculum_seed,
            is_train=is_train,
        )

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        reset_kwargs = dict(self._default_reset_kwargs)
        if kwargs is None:
            pass
        elif isinstance(kwargs, dict):
            reset_kwargs.update(kwargs)
        else:
            reset_kwargs["curriculum_state"] = kwargs
        return self._env.reset(kwargs=reset_kwargs)

    def step(self, action: Any) -> Tuple[str, float, bool, Dict[str, Any]]:
        return self._env.step(action)

    def close(self) -> None:
        self._env.close()

    def debug_state(self) -> Dict[str, Any]:
        curriculum_state = self._env._curriculum_state
        return {
            "seed": self._seed,
            "thread_id": self._thread_id,
            "curriculum_state": curriculum_state,
            "chemical_solution_state": self._env._chemical_solution_state,
            "max_chemical_n": self._env._max_chemical_n,
            "scenario_name": self._env._scenario_name,
            "difficulty": self._env._difficulty,
        }


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
        env_kwargs = dict(env_kwargs or {})

        # Allow env_num to be None (e.g. when val_batch_size is None)
        # In that case we simply create zero environments and return
        self.env_num = int(env_num) if env_num is not None else 0
        self.group_n = int(group_n)
        self.num_processes = self.env_num * self.group_n
        self.is_train = is_train

        self._workers: List[Any] = []
        self._curriculum_enabled = bool(env_kwargs.get("curriculum_enabled", False))
        # Use a single curriculum stage value derived from explicit curriculum_stage
        # or fall back to max_chemical_n (canonical external config key).
        self._curriculum_stage = int(env_kwargs.get("curriculum_stage", coerce_max_chemical_n(env_kwargs)))
        self._curriculum_train_fraction = float(env_kwargs.get("curriculum_train_fraction", 0.7))
        self._curriculum_mix_ratios = tuple(env_kwargs.get("curriculum_mix_ratios", (0.7, 0.2, 0.1)))
        self._curriculum_seed = int(env_kwargs.get("curriculum_seed", seed))
        self._curriculum_split = "train" if is_train else "val"
        self._curriculum_seed_pools = build_fixed_seed_pools_by_amount(
            base_seed=self._curriculum_seed,
            max_amount=self._curriculum_stage,
            num_chemicals=4,
            min_chemicals=1,
            train_fraction=self._curriculum_train_fraction,
        ) if self._curriculum_enabled else None
        self._curriculum_stage_pools = build_stage_pools(
            max_chemical_n=self._curriculum_stage,
            num_chemicals=4,
            train_fraction=self._curriculum_train_fraction,
            seed=self._curriculum_seed,
        ) if self._curriculum_enabled else None

        if self.num_processes == 0:
            # No envs to build (e.g. no validation envs configured)
            return

        if resources_per_worker is None:
            # Reasonable default: light CPU-only envs
            resources_per_worker = {"num_cpus": 0.1, "num_gpus": 0}

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init(address="auto", ignore_reinit_error=True)

        if "save_frames" not in env_kwargs:
            env_kwargs["save_frames"] = (not is_train)
        if env_kwargs.get("save_frames") and "frames_dir" not in env_kwargs:
            env_kwargs["frames_dir"] = build_frames_dir(env_kwargs, seed, is_train)
        env_kwargs["curriculum_enabled"] = self._curriculum_enabled
        env_kwargs["curriculum_train_fraction"] = self._curriculum_train_fraction
        env_kwargs["curriculum_mix_ratios"] = self._curriculum_mix_ratios
        env_kwargs["curriculum_seed"] = self._curriculum_seed
        env_kwargs["is_train"] = self.is_train

        env_worker = ray.remote(**resources_per_worker)(DiscoveryWorldWorker)
        selected_split = "train" if self.is_train else "val"
        if self._curriculum_enabled:
            if self.is_train:
                base_goal_stage_plan = plan_curriculum_goal_stages(
                    stage=self._curriculum_stage,
                    batch_size=self.env_num,
                    mix_ratios=self._curriculum_mix_ratios,
                    seed=self._curriculum_seed,
                )
            else:
                base_goal_stage_plan = [self._curriculum_stage] * self.env_num
            goal_stage_plan = [stage for stage in base_goal_stage_plan for _ in range(self.group_n)]
        else:
            configured_chemical_n = coerce_max_chemical_n(env_kwargs)
            goal_stage_plan = [configured_chemical_n] * self.num_processes

        print(f"DEBUG: Starting to launch {self.num_processes} Ray actors...", flush=True)
        for i in range(self.num_processes):
            goal_stage = int(goal_stage_plan[i])
            split_seed_list: List[int] = []
            if self._curriculum_seed_pools is not None:
                split_seed_list = list(
                    self._curriculum_seed_pools.get(goal_stage, {}).get(selected_split, []),
                )
            if not split_seed_list:
                split_seed_list = [int(seed)]

            # Share seed across group_n replicas while keeping train/val seed ranges isolated.
            seed_idx = (i // self.group_n) % len(split_seed_list)
            worker_seed = int(split_seed_list[seed_idx])
            worker_env_kwargs = dict(env_kwargs)
            worker_env_kwargs["max_chemical_n"] = goal_stage
            worker_env_kwargs["curriculum_goal_stage"] = goal_stage
            worker = env_worker.remote(seed=worker_seed, env_kwargs=worker_env_kwargs, thread_id=i)
            self._workers.append(worker)


    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.num_processes == 0:
            return [], []

        obs_list: List[str] = []
        info_list: List[Dict[str, Any]] = []

        # Build per-worker reset kwargs. Goal-stage mixing is assigned when
        # workers are constructed; reset only handles explicit per-worker init
        # overrides or lets each worker sample an init for its assigned goal.
        if kwargs is None:
            worker_kwargs = [None] * self.num_processes
        elif isinstance(kwargs, dict):
            # If caller provided an explicit `curriculum_state`, replicate it.
            # Otherwise each worker uses the goal stage assigned at construction
            # and samples an init state for that goal during reset.
            worker_kwargs = [dict(kwargs) for _ in range(self.num_processes)]
        elif isinstance(kwargs, np.ndarray):
            worker_kwargs = kwargs.tolist()
        elif isinstance(kwargs, (list, tuple)):
            if len(kwargs) != self.num_processes:
                raise ValueError(
                    f"Expected {self.num_processes} reset kwargs, got {len(kwargs)}",
                )
            worker_kwargs = list(kwargs)
        else:
            raise ValueError(f"Unsupported reset kwargs type: {type(kwargs)}")

        futures = [worker.reset.remote(worker_kwargs[i]) for i, worker in enumerate(self._workers)]
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
