# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Strict temporal-window dataset for action-to-action (A2A) training.

The regular single-step dataset historically shortened episodes using only the
future action horizon.  That is insufficient for A2A, where proprioceptive
history normally has negative ``delta_indices``.  Passing such an index to
``pandas.DataFrame.iloc`` silently reads from the end of the episode.

This module derives the valid base-step interval from *every* configured
modality and validates the interval again immediately before extraction.  A2A
training therefore never relies on pandas' negative-index behaviour.
"""

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path

import numpy as np
import pandas as pd

from gr00t.data.types import EmbodimentTag, MessageType, ModalityConfig, VLAStepData

from .sharded_single_step_dataset import ShardedSingleStepDataset, extract_step_data


@dataclass(frozen=True)
class A2AWindowBounds:
    """Inclusive valid interval for a base step in one episode."""

    valid_start: int
    valid_end: int
    min_delta: int
    max_delta: int

    @property
    def num_steps(self) -> int:
        """Number of base steps in the interval (zero for an empty interval)."""

        return max(0, self.valid_end - self.valid_start + 1)

    def contains(self, step_index: int) -> bool:
        """Return whether ``step_index`` is inside the inclusive interval."""

        return self.valid_start <= step_index <= self.valid_end


def is_a2a_model_config(model_config: object) -> bool:
    """Return whether a model config requires strict A2A dataset windows."""

    if getattr(model_config, "model_type", "") == "Gr00tN1d7A2A":
        return True
    history_horizon = getattr(model_config, "a2a_history_horizon", None)
    return (
        isinstance(history_horizon, int)
        and not isinstance(history_horizon, bool)
        and history_horizon > 0
    )


def get_modality_delta_bounds(
    modality_configs: dict[str, ModalityConfig],
) -> tuple[int, int]:
    """Return the minimum and maximum delta across all configured modalities.

    Empty configurations are rejected because they would make the valid window
    ambiguous.  ``bool`` is deliberately rejected even though it subclasses
    ``int``: accepting ``True`` as a temporal offset is almost certainly a
    configuration error.
    """

    if not modality_configs:
        raise ValueError("A2A requires at least one configured modality")

    deltas: list[int] = []
    for modality, config in modality_configs.items():
        if not config.delta_indices:
            raise ValueError(f"Modality {modality!r} has no delta_indices")
        for delta in config.delta_indices:
            if isinstance(delta, bool) or not isinstance(delta, Integral):
                raise TypeError(f"Modality {modality!r} contains non-integer delta index {delta!r}")
            deltas.append(int(delta))

    return min(deltas), max(deltas)


def compute_a2a_window_bounds(
    episode_length: int,
    modality_configs: dict[str, ModalityConfig],
) -> A2AWindowBounds:
    """Compute the inclusive base-step interval whose sampled indices are valid.

    For every configured delta ``d``, a valid base step ``t`` must satisfy
    ``0 <= t + d < episode_length``.  Intersecting those inequalities gives::

        valid_start = max(0, -min_delta)
        valid_end = episode_length - 1 - max_delta

    ``valid_end`` can be larger than ``episode_length - 1`` when every delta is
    negative.  That is mathematically valid: the base step is an anchor and all
    actually sampled indices still lie inside the episode.
    """

    if isinstance(episode_length, bool) or not isinstance(episode_length, Integral):
        raise TypeError(f"episode_length must be an integer, got {episode_length!r}")
    if episode_length < 0:
        raise ValueError(f"episode_length must be non-negative, got {episode_length}")

    min_delta, max_delta = get_modality_delta_bounds(modality_configs)
    valid_start = max(0, -min_delta)
    valid_end = int(episode_length) - 1 - max_delta
    return A2AWindowBounds(valid_start, valid_end, min_delta, max_delta)


def validate_a2a_step_index(
    episode_length: int,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
) -> A2AWindowBounds:
    """Validate a base step and every resulting physical row index.

    The second, explicit per-index check is intentionally redundant with the
    interval algebra.  It keeps the safety property local to extraction if the
    interval implementation is changed in the future.
    """

    if isinstance(step_index, bool) or not isinstance(step_index, Integral):
        raise TypeError(f"step_index must be an integer, got {step_index!r}")
    step_index = int(step_index)
    bounds = compute_a2a_window_bounds(episode_length, modality_configs)
    if not bounds.contains(step_index):
        raise IndexError(
            f"A2A base step {step_index} is outside valid interval "
            f"[{bounds.valid_start}, {bounds.valid_end}] for episode length "
            f"{episode_length} and delta range [{bounds.min_delta}, {bounds.max_delta}]"
        )

    invalid_indices = [
        (modality, step_index + int(delta))
        for modality, config in modality_configs.items()
        for delta in config.delta_indices
        if not 0 <= step_index + int(delta) < episode_length
    ]
    if invalid_indices:
        raise IndexError(
            "A2A extraction would access rows outside the episode: "
            + ", ".join(f"{modality}={index}" for modality, index in invalid_indices)
        )
    return bounds


def extract_a2a_step_data(
    episode_data: pd.DataFrame,
    step_index: int,
    modality_configs: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
) -> VLAStepData:
    """Safely extract one A2A window without temporal padding or wraparound."""

    bounds = validate_a2a_step_index(len(episode_data), step_index, modality_configs)
    step = extract_step_data(
        episode_data=episode_data,
        step_index=int(step_index),
        modality_configs=modality_configs,
        embodiment_tag=embodiment_tag,
        allow_padding=False,
    )
    step.metadata["a2a_window"] = {
        "base_step_index": int(step_index),
        "valid_start": bounds.valid_start,
        "valid_end": bounds.valid_end,
        "min_delta": bounds.min_delta,
        "max_delta": bounds.max_delta,
    }
    return step


class A2AShardedSingleStepDataset(ShardedSingleStepDataset):
    """Single-step dataset using strict all-modality A2A temporal windows.

    Padding is intentionally not supported for training.  Runtime cold-start
    padding belongs in the stateful policy/processor path, where a validity mask
    can accompany repeated proprio observations.
    """

    def __init__(
        self,
        dataset_path: str | Path,
        embodiment_tag: EmbodimentTag,
        modality_configs: dict[str, ModalityConfig],
        shard_size: int = 2**10,
        episode_sampling_rate: float = 0.1,
        seed: int = 42,
        allow_padding: bool = False,
    ):
        if allow_padding:
            raise ValueError(
                "A2A training requires strict valid windows; allow_padding must be False. "
                "Handle inference cold-start padding in the policy/processor instead."
            )
        if not 0 < episode_sampling_rate <= 1:
            raise ValueError(
                f"episode_sampling_rate must be in (0, 1] for A2A, got {episode_sampling_rate}"
            )
        if isinstance(shard_size, bool) or not isinstance(shard_size, Integral) or shard_size <= 0:
            raise ValueError(f"shard_size must be a positive integer, got {shard_size!r}")
        super().__init__(
            dataset_path=dataset_path,
            embodiment_tag=embodiment_tag,
            modality_configs=modality_configs,
            shard_size=int(shard_size),
            episode_sampling_rate=episode_sampling_rate,
            seed=seed,
            allow_padding=False,
        )

    def get_valid_step_bounds(self, episode_index: int) -> A2AWindowBounds:
        """Return strict bounds for one episode."""

        return compute_a2a_window_bounds(
            self.episode_loader.get_episode_length(episode_index), self.modality_configs
        )

    def get_effective_episode_length(self, episode_index: int) -> int:
        """Return the number of strict all-modality windows in an episode."""

        return self.get_valid_step_bounds(episode_index).num_steps

    def shard_dataset(self):
        """Build balanced shards containing absolute, already-valid base steps."""

        shuffled_episode_indices = self.rng.permutation(len(self.episode_loader.episode_lengths))
        num_splits = int(1 / self.episode_sampling_rate)
        if len(shuffled_episode_indices) == 0:
            raise AssertionError(f"No valid trajectories found for dataset {self.dataset_path}")

        episode_splits: list[tuple[int, np.ndarray]] = []
        total_steps = 0
        for ep_idx in shuffled_episode_indices:
            bounds = self.get_valid_step_bounds(int(ep_idx))
            step_indices = np.arange(bounds.valid_start, bounds.valid_end + 1, dtype=np.int64)
            self.rng.shuffle(step_indices)
            total_steps += len(step_indices)
            for split_index in range(num_splits):
                split_step_indices = step_indices[split_index::num_splits]
                if len(split_step_indices) > 0:
                    episode_splits.append((int(ep_idx), split_step_indices))

        if total_steps == 0 or not episode_splits:
            min_delta, max_delta = get_modality_delta_bounds(self.modality_configs)
            raise AssertionError(
                f"No valid A2A timesteps found for dataset {self.dataset_path}; "
                f"episodes do not cover delta range [{min_delta}, {max_delta}]"
            )

        num_shards = min(int(np.ceil(total_steps / self.shard_size)), len(episode_splits))
        sharded_episodes: list[list[tuple[int, np.ndarray]]] = [[] for _ in range(num_shards)]
        shard_lengths = np.zeros(num_shards, dtype=int)

        for split_index, (ep_idx, split_step_indices) in enumerate(episode_splits):
            shard_index = split_index if split_index < num_shards else int(np.argmin(shard_lengths))
            sharded_episodes[shard_index].append((ep_idx, split_step_indices))
            shard_lengths[shard_index] += len(split_step_indices)

        if not np.all(shard_lengths > 0):
            raise AssertionError("All A2A shards must have length greater than zero")

        print(f"Generated {num_shards} A2A shards for dataset {self.dataset_path}")
        print(
            f"Total A2A steps: {total_steps}, average shard length: "
            f"{total_steps / num_shards}, shard length std: {np.std(shard_lengths)}"
        )
        self.sharded_episodes = sharded_episodes
        self.shard_lengths = shard_lengths

    def get_datapoint(self, episode_data: pd.DataFrame, step_index: int) -> dict:
        """Extract a strict A2A window and apply the configured processor."""

        assert self.processor is not None, "Processor must be set before getting datapoints"
        vla_step_data = extract_a2a_step_data(
            episode_data=episode_data,
            step_index=step_index,
            modality_configs=self.modality_configs,
            embodiment_tag=self.embodiment_tag,
        )
        messages = [{"type": MessageType.EPISODE_STEP.value, "content": vla_step_data}]
        return self.processor(messages)
