"""
Comprehensive refactoring of DiscoveryWorldEnv to:
1. Remove prev_* injections, use two-info snapshot model instead
2. Remove redundant variables (action_counter, object_seen, teacher_prob, etc.)
3. Simplify info-related functions
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import json
import os
import re
import time
from collections import defaultdict
import ray
from copy import deepcopy
import torch
import math
import numpy as np

from agent_system.environments.env_package.discovery.discoveryworld.discoveryworld.DiscoveryWorldAPI import(
    DiscoveryWorldAPI,
)
from agent_system.environments.env_package.discovery.rule_based_agent import *
from agent_system.environments.env_package.discovery.skills import CombinatorialChemistrySkill
from agent_system.environments.env_package.discovery.seed import assign_split_seeds


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
        chemical_N: int = 2,
    ) -> None:
        self._seed = seed
        self._scenario_name = scenario_name
        self._difficulty = difficulty
        self._max_steps = max_steps
        self._thread_id = thread_id
        self._save_frames = bool(save_frames)
        self._frames_dir = frames_dir
        self._chemical_N = chemical_N

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
            "chemicalMinAmount": self._chemical_N,
            "chemicalMaxAmount": self._chemical_N,
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

        has_key, has_jar, is_key_in_jar, chemical_dict = extract_detailed_status(ui)
        info["has_key"] = has_key
        info["has_jar"] = has_jar
        info["is_key_in_jar"] = is_key_in_jar
        info["chemical_dict"] = deepcopy(chemical_dict)

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        """Reset environment and return initial observation and info."""
        self._init_api()
        self._steps = 0
        self._prev_score = 0.0
        self._last_action_result = None
        
        self.init_reward_shaping()

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

        # Invalid action path
        if not is_valid:
            self.action_history.append("INVALID")
            self._steps += 1

            text_obs, info = self._format_obs_and_info()
            info["action_status"] = "invalid"
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
            success = bool(self._last_action_result.get("success", False))
            action_status = "success" if success else "valid_but_failed"

        except Exception:
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
        reward = cur_score - self._prev_score
        self._prev_score = cur_score
        return reward
    
    def _teacher_skill_reward(self, skill_name: Optional[str], info: Dict[str, Any]) -> float:
        teacher_skill = self.teacher.select_skill(self._last_info)
        self._last_teacher_skill = teacher_skill
        
        if not teacher_skill:
            # Log cases where teacher couldn't decide
            with open("teacher_skill.txt", "a") as f:
                ui = (info.get("raw_observation") or {}).get("ui", {})
                compressed_obs = compress_ui_observation(ui)
                f.write("======================\n")
                f.write(f"Current is_key_in_jar: {info.get('is_key_in_jar', False)}\n")
                f.write(f"Current used_dispensers: {info.get('used_dispensers', {})}\n")
                f.write(f"Observation:\n{compressed_obs}\n\n")
            return 0.0
        
        if skill_name == teacher_skill:
            return 0.7
        else:
            return -0.05

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
        if action_status == "invalid":
            return -0.5
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
    
    def _no_progress_move_penalty(self, action, cur_score) -> float:
        """Penalty for moving but making no progress."""
        if len(self.location_history) < 4:
            return 0.0

        no_location_change = len(set(self.location_history[-4:])) == 1
        no_score_change = abs(cur_score - self._prev_score) < 1e-6

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
        game_progress_reward = self._game_progress_reward(cur_score)
        teacher_skill_reward = self._teacher_skill_reward(skill_name, self._last_info)
        stage_reward = self._stage_reward(self._last_info, info)
        
        repetition_penalty = self._repetition_penalty()
        invalid_penalty = self._invalid_action_penalty(info)
        no_progress_penalty = self._no_progress_move_penalty(skill_name, cur_score)
        info["teacher_skill"] = self._last_teacher_skill

        task_completed = bool(self._api.areTasksComplete())
        done = task_completed or self._steps >= self._max_steps
        info["won"] = task_completed

        reward = 0.0
        # Positive rewards
        reward += 10.0 * game_progress_reward
        # reward += teacher_skill_reward
        # reward += stage_reward

        # Penalties
        reward += repetition_penalty
        reward += invalid_penalty
        reward += no_progress_penalty
        
        if not task_completed:
            reward = self.clip_reward(reward)
        else:
            reward += 10.0

        return reward, done


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
            **env_kwargs,
        )

    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[str, Dict[str, Any]]:
        return self._env.reset(kwargs=kwargs)

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
        configured_train_size = int(env_kwargs.get("train_size", self.env_num))
        configured_val_size = int(env_kwargs.get("val_size", self.env_num))
        configured_chemical_n = int(env_kwargs.get("chemical_N", 2))
        split_seeds = assign_split_seeds(
            base_seed=seed,
            train_size=configured_train_size,
            val_size=configured_val_size,
            num_chemicals=4,
            min_chemicals=1,
            min_amount=configured_chemical_n,
            max_amount=configured_chemical_n,
        )
        selected_split = "train" if self.is_train else "val"
        split_seed_list = split_seeds[selected_split]
        if not split_seed_list:
            split_seed_list = [int(seed)]

        print(f"DEBUG: Starting to launch {self.num_processes} Ray actors...", flush=True)
        for i in range(self.num_processes):
            # Share seed across group_n replicas while keeping train/val seed ranges isolated.
            seed_idx = (i // self.group_n) % len(split_seed_list)
            worker_seed = int(split_seed_list[seed_idx])
            worker = env_worker.remote(seed=worker_seed, env_kwargs=env_kwargs, thread_id=i)
            self._workers.append(worker)


    def reset(self, kwargs: Optional[Dict[str, Any]] = None) -> Tuple[List[str], List[Dict[str, Any]]]:
        if self.num_processes == 0:
            return [], []

        obs_list: List[str] = []
        info_list: List[Dict[str, Any]] = []

        if kwargs is None:
            worker_kwargs = [None] * self.num_processes
        elif isinstance(kwargs, dict):
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
