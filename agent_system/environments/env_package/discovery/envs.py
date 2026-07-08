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
from agent_system.environments.env_package.discovery.config import (
    DiscoveryWorkerConfig,
    build_frames_dir,
    coerce_bool,
    coerce_max_chemical_n,
)
from agent_system.environments.env_package.discovery.rewards import DiscoveryWorldRewardMixin
from agent_system.environments.env_package.discovery.seed import (
    build_fixed_seed_pools_by_amount,
    build_ordered_seed_pools_by_amount,
)
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.env_package.discovery.utils import (
    SKILL_NAMES,
    compress_ui_observation,
    extract_detailed_status,
    format_rust_level,
)


logger = logging.getLogger(__name__)


class DiscoveryWorldEnv(DiscoveryWorldRewardMixin):

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
        self._save_frames = coerce_bool(save_frames)
        self._frames_dir = frames_dir
        self._max_chemical_n = int(max_chemical_n) if max_chemical_n is not None else 2
        self._curriculum_enabled = coerce_bool(curriculum_enabled)
        self._curriculum_train_fraction = float(curriculum_train_fraction)
        self._curriculum_mix_ratios = tuple(curriculum_mix_ratios)
        self._curriculum_seed = int(curriculum_seed) if curriculum_seed is not None else int(seed)
        self._is_train = coerce_bool(is_train, default=True)
        self._curriculum_state: Optional[Tuple[int, ...]] = None
        self._curriculum_terminal_reset: bool = False
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

    def _is_task_complete(self, info: Optional[Dict[str, Any]] = None) -> bool:
        assert self._api is not None
        return bool(self._api.areTasksComplete())

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
            "curriculum_terminal_reset": self._curriculum_terminal_reset,
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
        self._curriculum_terminal_reset = False
        
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
        curriculum_sample_seed = reset_kwargs.pop("curriculum_sample_seed", None)
        curriculum_terminal_reset = coerce_bool(reset_kwargs.pop("curriculum_terminal_reset", False))

        if curriculum_state is not None:
            self._curriculum_state = normalize_chemical_state(curriculum_state)
            saved_used_dispensers = dict(self.used_dispensers)
            self._skill_runner.prepare_chemical_state(self._curriculum_state)
            self.used_dispensers = saved_used_dispensers
            self._last_action_result = None
        elif self._curriculum_enabled and self._chemical_solution_state is not None:
            self._chemical_solution_state = self._get_chemical_solution_state()
            if curriculum_terminal_reset:
                self._curriculum_state = self._chemical_solution_state
                self._curriculum_terminal_reset = True
            else:
                sample_seed = (
                    int(curriculum_sample_seed)
                    if curriculum_sample_seed is not None
                    else self._curriculum_seed + self._thread_id
                )
                self._curriculum_state = sample_solution_init_state(
                    solution_state=self._chemical_solution_state,
                    split="train" if self._is_train else "val",
                    train_fraction=self._curriculum_train_fraction,
                    seed=sample_seed,
                    num_chemicals=4,
                )
            saved_used_dispensers = dict(self.used_dispensers)
            self._skill_runner.prepare_chemical_state(self._curriculum_state)
            self.used_dispensers = saved_used_dispensers
            self._last_action_result = None

        text_obs, info = self._format_obs_and_info()
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_state_from_info(info)
        info["won"] = bool(self._is_task_complete(info))
        
        self._last_info = deepcopy(info)
        self._prev_score = float(info.get("score_normalized", 0.0))
        if not self.location_history:
            self._update_location_history(ui)
        
        return text_obs, info

    def step(self, action: Any, format_score: float = 0.0) -> Tuple[str, float, bool, Dict[str, Any]]:
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
            
            reward, done = self._compute_step_reward(None, info, format_score=format_score)
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
        
        reward, done = self._compute_step_reward(skill_name, info, format_score=format_score)
        self._last_info = deepcopy(info)
        
        return text_obs, reward, done, info

    def close(self) -> None:
        return None


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
        worker_config = DiscoveryWorkerConfig.from_env_kwargs(seed=seed, env_kwargs=env_kwargs)
        self._default_reset_kwargs = worker_config.default_reset_kwargs
        env_cls = _select_discoveryworld_env_cls(worker_config.env_variant)

        self._env = env_cls(
            seed=seed,
            scenario_name=worker_config.scenario_name,
            difficulty=worker_config.difficulty,
            max_steps=worker_config.max_steps,
            thread_id=thread_id,
            save_frames=worker_config.save_frames,
            frames_dir=worker_config.frames_dir,
            max_chemical_n=worker_config.max_chemical_n,
            curriculum_enabled=worker_config.curriculum_enabled,
            curriculum_train_fraction=worker_config.curriculum_train_fraction,
            curriculum_mix_ratios=worker_config.curriculum_mix_ratios,
            curriculum_seed=worker_config.curriculum_seed,
            is_train=worker_config.is_train,
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

    def step(self, action: Any, format_score: float = 0.0) -> Tuple[str, float, bool, Dict[str, Any]]:
        return self._env.step(action, format_score=format_score)

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
            "env_variant": self._env.__class__.__name__,
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
        self.is_train = coerce_bool(is_train, default=True)

        self._workers: List[Any] = []
        self._curriculum_enabled = coerce_bool(env_kwargs.get("curriculum_enabled", False))
        # Use a single curriculum stage value derived from explicit curriculum_stage
        # or fall back to max_chemical_n (canonical external config key).
        self._curriculum_stage = int(env_kwargs.get("curriculum_stage", coerce_max_chemical_n(env_kwargs)))
        self._curriculum_train_fraction = float(env_kwargs.get("curriculum_train_fraction", 0.7))
        self._target_train_fraction = float(env_kwargs.get("target_train_fraction", 0.8))
        self._curriculum_mix_ratios = tuple(env_kwargs.get("curriculum_mix_ratios", (0.7, 0.2, 0.1)))
        self._curriculum_seed = int(env_kwargs.get("curriculum_seed", seed))
        self._curriculum_terminal_reset_ratio = min(
            max(float(env_kwargs.get("curriculum_terminal_reset_ratio", 0.0)), 0.0),
            1.0,
        )
        self._curriculum_terminal_reset_eval = coerce_bool(
            env_kwargs.get("curriculum_terminal_reset_eval", False),
        )
        self._env_variant = str(env_kwargs.get("env_variant", "original")).strip().lower()
        self._curriculum_split = "train" if self.is_train else "val"
        self._reset_count = 0
        self._curriculum_seed_pools = build_fixed_seed_pools_by_amount(
            base_seed=self._curriculum_seed,
            max_amount=self._curriculum_stage,
            num_chemicals=4,
            min_chemicals=1,
            train_fraction=self._curriculum_train_fraction,
        ) if self._curriculum_enabled else None
        self._target_seed_pools = build_ordered_seed_pools_by_amount(
            max_amount=self._curriculum_stage,
            num_chemicals=4,
            min_chemicals=1,
            train_fraction=self._target_train_fraction,
        ) if not self._curriculum_enabled else None
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
            env_kwargs["save_frames"] = (not self.is_train)
        if coerce_bool(env_kwargs.get("save_frames")) and "frames_dir" not in env_kwargs:
            env_kwargs["frames_dir"] = build_frames_dir(env_kwargs, seed, is_train)
        env_kwargs["curriculum_enabled"] = self._curriculum_enabled
        env_kwargs["curriculum_train_fraction"] = self._curriculum_train_fraction
        env_kwargs["target_train_fraction"] = self._target_train_fraction
        env_kwargs["curriculum_mix_ratios"] = self._curriculum_mix_ratios
        env_kwargs["curriculum_seed"] = self._curriculum_seed
        env_kwargs["curriculum_terminal_reset_ratio"] = self._curriculum_terminal_reset_ratio
        env_kwargs["env_variant"] = self._env_variant
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
        seed_pools = self._curriculum_seed_pools if self._curriculum_enabled else self._target_seed_pools
        for i in range(self.num_processes):
            goal_stage = int(goal_stage_plan[i])
            split_seed_list: List[int] = []
            if seed_pools is not None:
                split_seed_list = list(
                    seed_pools.get(goal_stage, {}).get(selected_split, []),
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
        # workers are constructed; reset coordinates per-group init sampling
        # so GRPO replicas compare actions from the same state.
        if kwargs is None:
            worker_kwargs = self._build_group_reset_kwargs({})
        elif isinstance(kwargs, dict):
            # If caller provided an explicit `curriculum_state`, replicate it.
            # Otherwise each worker uses the goal stage assigned at construction
            # and samples an init state for that goal during reset.
            if self._has_explicit_curriculum_state(kwargs):
                worker_kwargs = [dict(kwargs) for _ in range(self.num_processes)]
            else:
                worker_kwargs = self._build_group_reset_kwargs(dict(kwargs))
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

    def _has_explicit_curriculum_state(self, kwargs: Dict[str, Any]) -> bool:
        return "curriculum_state" in kwargs or "chemical_state" in kwargs

    def _build_group_reset_kwargs(self, base_kwargs: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not self._curriculum_enabled:
            return [dict(base_kwargs) for _ in range(self.num_processes)]

        reset_idx = self._reset_count
        self._reset_count += 1
        rng = np.random.default_rng(self._curriculum_seed + reset_idx)
        terminal_mask = np.zeros(self.env_num, dtype=bool)
        if (
            (self.is_train or self._curriculum_terminal_reset_eval)
            and self._curriculum_terminal_reset_ratio > 0.0
            and self.env_num > 0
        ):
            terminal_count = int(round(self.env_num * self._curriculum_terminal_reset_ratio))
            terminal_count = min(max(terminal_count, 1), self.env_num)
            terminal_groups = rng.permutation(self.env_num)[:terminal_count]
            terminal_mask[terminal_groups] = True

        worker_kwargs: List[Dict[str, Any]] = []
        for group_idx in range(self.env_num):
            sample_seed = self._curriculum_seed + reset_idx * max(self.env_num, 1) + group_idx
            group_kwargs = dict(base_kwargs)
            group_kwargs["curriculum_sample_seed"] = int(sample_seed)
            if coerce_bool(terminal_mask[group_idx]):
                group_kwargs["curriculum_terminal_reset"] = True
            for _ in range(self.group_n):
                worker_kwargs.append(dict(group_kwargs))

        return worker_kwargs

    def step(
        self,
        actions: List[Any],
        format_scores: Optional[List[float]] = None,
    ) -> Tuple[List[str], List[float], List[bool], List[Dict[str, Any]]]:
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
        if format_scores is None:
            format_scores = [0.0] * len(actions)
        if len(format_scores) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} format scores, got {len(format_scores)}",
            )

        obs_list: List[str] = []
        reward_list: List[float] = []
        done_list: List[bool] = []
        info_list: List[Dict[str, Any]] = []

        futures = []
        for worker, act, format_score in zip(self._workers, actions, format_scores):
            future = worker.step.remote(act, float(format_score))
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


def _select_discoveryworld_env_cls(env_variant: Optional[str]):
    variant = str(env_variant or "original").strip().lower()
    if variant in {"", "original", "default", "full"}:
        return DiscoveryWorldEnv
    if variant in {"pickjar", "pick_jar", "pick-jar", "jar"}:
        from agent_system.environments.env_package.discovery.env_variants import CCEnvPickJar

        return CCEnvPickJar
    if variant in {
        "derustmoderate",
        "derust_to_moderate",
        "derust-to-moderate",
        "moderaterust",
        "moderate_rust",
        "lightlyrust",
        "lightly_rust",
        "lightly-rust",
        "lightlyrusted",
    }:
        from agent_system.environments.env_package.discovery.env_variants import CCEnvDerustToModerate

        return CCEnvDerustToModerate
    raise ValueError(
        f"Unsupported DiscoveryWorld env_variant={env_variant!r}. "
        "Expected one of: original, pickjar, derustmoderate."
    )


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
