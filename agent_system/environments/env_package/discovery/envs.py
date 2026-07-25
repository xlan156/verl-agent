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
from agent_system.environments.env_package.discovery.config import (
    DiscoveryWorkerConfig,
    build_frames_dir,
    coerce_bool,
    coerce_max_chemical_n,
)
from agent_system.environments.env_package.discovery.rewards import DiscoveryWorldRewardMixin
from agent_system.environments.env_package.discovery.seed import build_ordered_seed_pools_by_amount
from agent_system.environments.env_package.discovery.dynamic_sampler import DynamicSeedSampler
from agent_system.environments.env_package.discovery.rule_based_agent import RulebasedAgentSkill
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.env_package.discovery.utils import (
    SKILL_NAMES,
    compress_ui_observation,
    extract_detailed_status,
    format_rust_level,
    is_dispenser_skill,
    is_remove_skill,
    reaction_signal_for_tuples,
)


logger = logging.getLogger(__name__)


def _solution_dict_to_state(solution: Dict[str, int]) -> Tuple[int, ...]:
    """Convert DiscoveryWorld's ``Substance X`` keys to the public A-D state."""
    return tuple(
        int(
            solution.get(
                name,
                solution.get(f"Chemical {name}", solution.get(f"Substance {name}", 0)),
            )
            or 0
        )
        for name in ("A", "B", "C", "D")
    )


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
        teacher_skill_reward_coef: float = 1.0,
    ) -> None:
        self._seed = seed
        self._scenario_name = scenario_name
        self._difficulty = difficulty
        self._max_steps = max_steps
        self._thread_id = thread_id
        self._save_frames = coerce_bool(save_frames)
        self._frames_dir = frames_dir
        self._max_chemical_n = int(max_chemical_n) if max_chemical_n is not None else 2
        self._teacher_skill_reward_coef = float(teacher_skill_reward_coef)
        self._hidden_chemical_target: Optional[Tuple[int, ...]] = None
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

    def init_reward_shaping(self):

        self.action_history: List[Optional[str]] = []
        self.location_history: List[Tuple[Optional[int], Optional[int]]] = []

        self._skill_runner = CombinatorialChemistrySkill(self)
        self.teacher = RulebasedAgentSkill(self)
        self._last_teacher_skill: Optional[str] = None
        self._last_info: Optional[Dict[str, Any]] = None

        self.used_dispensers = {"A": False, "B": False, "C": False, "D": False}

    def _read_hidden_chemical_target(self) -> Optional[Tuple[int, ...]]:
        """Read the simulator-only target without exposing it in observations."""
        world = getattr(self._api, "world", None)
        if world is None:
            return None
        for obj in world.getAllWorldObjects():
            target = (getattr(obj, "attributes", None) or {}).get("rustRemovalDict")
            if target:
                return _solution_dict_to_state(target)
        return None

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
        mixture = tuple(int(chemical_dict.get(name, 0) or 0) for name in ("A", "B", "C", "D"))
        if is_key_in_jar and any(mixture) and self._hidden_chemical_target is not None:
            info["current_reaction_signal"] = reaction_signal_for_tuples(
                mixture, self._hidden_chemical_target
            )
        else:
            info["current_reaction_signal"] = "not tested"

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and return initial observation and info."""
        self._init_api()
        self._steps = 0
        self._prev_score = 0.0
        self._last_action_result = None

        self.init_reward_shaping()
        self._hidden_chemical_target = self._read_hidden_chemical_target()
        text_obs, info = self._format_obs_and_info()
        ui = (info.get("raw_observation") or {}).get("ui", {})
        self._update_state_from_info(info)
        info["won"] = bool(self._is_task_complete(info))
        
        self._last_info = deepcopy(info)
        self._prev_score = float(info.get("score_normalized", 0.0))
        if not self.location_history:
            self._update_location_history(ui)
        
        return text_obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
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
            # A dispenser tick updates the jar's Substance mixture after the
            # key has already evaluated it. Settle once more before exposing
            # the transition so the returned rust state describes the action
            # that was just executed, rather than the preceding mixture.
            if is_dispenser_skill(skill_name) or is_remove_skill(skill_name):
                self._skill_runner.settle_reactions(max_ticks=1)
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
        worker_config = DiscoveryWorkerConfig.from_env_kwargs(env_kwargs=env_kwargs)
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
            teacher_skill_reward_coef=worker_config.teacher_skill_reward_coef,
        )

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        reset_kwargs = dict(kwargs or {})
        requested_seed = reset_kwargs.pop("seed", None)
        if requested_seed is not None:
            self._seed = int(requested_seed)
            # DiscoveryWorldEnv recreates its API/scenario during every reset,
            # so updating the seed before reset safely changes the latent task
            # without recreating the Ray actor.
            self._env._seed = self._seed
        if "train_epoch" in reset_kwargs:
            self._env.train_epoch = int(reset_kwargs["train_epoch"])
        obs, info = self._env.reset(kwargs=reset_kwargs)
        info["seed"] = self._seed
        return obs, info

    def step(
        self,
        action: Any,
    ) -> Tuple[str, float, bool, Dict[str, Any]]:
        obs, reward, done, info = self._env.step(action)
        info["seed"] = self._seed
        return obs, reward, done, info

    def close(self) -> None:
        self._env.close()

    def debug_state(self) -> Dict[str, Any]:
        return {
            "seed": self._seed,
            "thread_id": self._thread_id,
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
        self._worker_seeds: List[int] = []
        self._max_chemical_n = coerce_max_chemical_n(env_kwargs)
        self._target_train_fraction = float(env_kwargs.get("target_train_fraction", 0.8))
        self._env_variant = str(env_kwargs.get("env_variant", "original")).strip().lower()
        self._target_seed_pools = build_ordered_seed_pools_by_amount(
            max_amount=self._max_chemical_n,
            num_chemicals=4,
            min_chemicals=1,
            train_fraction=self._target_train_fraction,
        )
        self._train_seed_pool = env_kwargs.get("train_seed_pool")
        self._eval_seed_pool = env_kwargs.get("eval_seed_pool")
        dynamic_sampler_config = dict(env_kwargs.get("dynamic_sampler") or {})
        self._dynamic_sampler: Optional[DynamicSeedSampler] = None

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
        env_kwargs["target_train_fraction"] = self._target_train_fraction
        env_kwargs["env_variant"] = self._env_variant

        env_worker = ray.remote(**resources_per_worker)(DiscoveryWorldWorker)
        selected_split = "train" if self.is_train else "val"
        goal_stage_plan = [self._max_chemical_n] * self.num_processes
        default_split_seed_list = list(
            self._target_seed_pools.get(self._max_chemical_n, {}).get(selected_split, [])
        )
        if self.is_train and self._train_seed_pool is not None:
            default_split_seed_list = [int(value) for value in self._train_seed_pool]
        if not self.is_train and self._eval_seed_pool is not None:
            default_split_seed_list = [int(value) for value in self._eval_seed_pool]
        if not default_split_seed_list:
            default_split_seed_list = [int(seed)]

        if self.is_train and coerce_bool(dynamic_sampler_config.get("enable", False)):
            self._dynamic_sampler = DynamicSeedSampler(
                seeds=default_split_seed_list,
                config=dynamic_sampler_config,
                rng_seed=int(dynamic_sampler_config.get("rng_seed", seed)),
            )
            print(
                "DiscoveryWorld dynamic seed sampler enabled for seeds="
                f"{list(self._dynamic_sampler.seeds)}",
                flush=True,
            )

        print(f"DEBUG: Starting to launch {self.num_processes} Ray actors...", flush=True)
        seed_pools = self._target_seed_pools
        for i in range(self.num_processes):
            goal_stage = int(goal_stage_plan[i])
            split_seed_list: List[int] = []
            if seed_pools is not None:
                split_seed_list = list(
                    seed_pools.get(goal_stage, {}).get(selected_split, []),
                )
            if self.is_train and self._train_seed_pool is not None:
                split_seed_list = [int(value) for value in self._train_seed_pool]
            if not self.is_train and self._eval_seed_pool is not None:
                split_seed_list = [int(value) for value in self._eval_seed_pool]
            if not split_seed_list:
                split_seed_list = [int(seed)]

            # Share seed across group_n replicas while keeping train/val seed ranges isolated.
            seed_idx = (i // self.group_n) % len(split_seed_list)
            worker_seed = int(split_seed_list[seed_idx])
            worker_env_kwargs = dict(env_kwargs)
            worker_env_kwargs["max_chemical_n"] = goal_stage
            worker = env_worker.remote(seed=worker_seed, env_kwargs=worker_env_kwargs, thread_id=i)
            self._workers.append(worker)
            self._worker_seeds.append(worker_seed)

        # GRPO/GiGPO relative advantages are only meaningful when every
        # replica in a group faces the same latent target.
        for group_start in range(0, self.num_processes, self.group_n):
            group_seeds = self._worker_seeds[group_start : group_start + self.group_n]
            if len(set(group_seeds)) != 1:
                raise RuntimeError(f"Mixed hidden targets in rollout group: {group_seeds}")


    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.num_processes == 0:
            return [], []

        obs_list: List[str] = []
        info_list: List[Dict[str, Any]] = []

        if kwargs is None:
            worker_kwargs = [{} for _ in range(self.num_processes)]
        elif isinstance(kwargs, dict):
            worker_kwargs = [dict(kwargs) for _ in range(self.num_processes)]
        elif isinstance(kwargs, np.ndarray):
            worker_kwargs = [dict(value or {}) for value in kwargs.tolist()]
        elif isinstance(kwargs, (list, tuple)):
            if len(kwargs) != self.num_processes:
                raise ValueError(
                    f"Expected {self.num_processes} reset kwargs, got {len(kwargs)}",
                )
            worker_kwargs = [dict(value or {}) for value in kwargs]
        else:
            raise ValueError(f"Unsupported reset kwargs type: {type(kwargs)}")

        if self._dynamic_sampler is not None:
            group_seeds = self._dynamic_sampler.sample(self.env_num)
            for group_index, group_seed in enumerate(group_seeds):
                group_start = group_index * self.group_n
                group_end = group_start + self.group_n
                for worker_index in range(group_start, group_end):
                    worker_kwargs[worker_index]["seed"] = int(group_seed)
                    self._worker_seeds[worker_index] = int(group_seed)

        futures = [worker.reset.remote(worker_kwargs[i]) for i, worker in enumerate(self._workers)]
        print(f"DEBUG: Waiting for {len(futures)} workers to reset via ray.get...", flush=True)
        results = ray.get(futures)
        print("DEBUG: All workers reset successfully.", flush=True)

        for obs, info in results:
            self._append_result(obs_list, info_list, obs, info)

        return obs_list, info_list

    def observe_dynamic_sampler(
        self,
        episode_rewards: np.ndarray,
        episode_task_rewards: np.ndarray,
        success: Dict[str, np.ndarray],
        accepted_groups: np.ndarray,
        global_step: Optional[int] = None,
    ) -> None:
        if self._dynamic_sampler is None:
            return
        rewards = np.asarray(episode_rewards)
        task_rewards = np.asarray(episode_task_rewards)
        seeds = np.asarray(success["eval_seed"])
        successes = np.asarray(success["success_rate"])
        accepted_groups = np.asarray(accepted_groups, dtype=bool)
        if len(rewards) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} rewards, received {len(rewards)}"
            )
        if len(task_rewards) != self.num_processes:
            raise ValueError(
                f"Expected {self.num_processes} task rewards, received {len(task_rewards)}"
            )
        if len(accepted_groups) != self.env_num:
            raise ValueError(
                f"Expected {self.env_num} group decisions, received {len(accepted_groups)}"
            )
        for group_index in range(self.env_num):
            start = group_index * self.group_n
            end = start + self.group_n
            group_seeds = seeds[start:end]
            if len(set(int(value) for value in group_seeds)) != 1:
                raise RuntimeError(f"Mixed seeds in adaptive rollout group: {group_seeds}")
            self._dynamic_sampler.observe_group(
                seed=int(group_seeds[0]),
                rewards=rewards[start:end],
                task_rewards=task_rewards[start:end],
                successes=successes[start:end],
                accepted=bool(accepted_groups[group_index]),
                global_step=global_step,
            )

    def dynamic_sampler_metrics(self) -> Dict[str, float]:
        if self._dynamic_sampler is None:
            return {}
        return self._dynamic_sampler.metrics()

    def dynamic_sampler_state_dict(self) -> Optional[Dict[str, Any]]:
        if self._dynamic_sampler is None:
            return None
        return self._dynamic_sampler.state_dict()

    def load_dynamic_sampler_state_dict(self, state: Dict[str, Any]) -> None:
        if self._dynamic_sampler is None:
            raise ValueError(
                "Checkpoint contains a dynamic sampler, but adaptive seed sampling is disabled"
            )
        self._dynamic_sampler.load_state_dict(state)

    def step(
        self,
        actions: List[Any],
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
