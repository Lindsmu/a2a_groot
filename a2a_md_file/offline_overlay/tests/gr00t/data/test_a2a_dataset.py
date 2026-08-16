# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from gr00t.data.dataset.a2a_single_step_dataset import (
    A2AShardedSingleStepDataset,
    compute_a2a_window_bounds,
    extract_a2a_step_data,
    get_modality_delta_bounds,
    is_a2a_model_config,
    validate_a2a_step_index,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
import numpy as np
import pandas as pd
import pytest


def _a2a_modality_configs() -> dict[str, ModalityConfig]:
    return {
        "state": ModalityConfig(delta_indices=[-2, -1, 0], modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }


def test_window_bounds_use_every_modality_delta():
    configs = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["cam"]),
        "state": ModalityConfig(delta_indices=list(range(-7, 1)), modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=list(range(8)), modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }

    bounds = compute_a2a_window_bounds(20, configs)

    assert bounds.valid_start == 7
    assert bounds.valid_end == 12
    assert bounds.num_steps == 6
    assert (bounds.min_delta, bounds.max_delta) == (-7, 7)


def test_window_bounds_support_all_negative_arbitrary_deltas():
    configs = {
        "state": ModalityConfig(delta_indices=[-3, -1], modality_keys=["joint"]),
    }

    bounds = compute_a2a_window_bounds(10, configs)

    assert bounds.valid_start == 3
    assert bounds.valid_end == 10
    assert all(0 <= step + delta < 10 for step in range(3, 11) for delta in [-3, -1])


def test_invalid_early_step_is_rejected_before_pandas_iloc_wraparound():
    configs = _a2a_modality_configs()
    episode = pd.DataFrame(
        {
            "state.joint": [np.array([index]) for index in range(6)],
            "action.joint": [np.array([100 + index]) for index in range(6)],
            "language.task": ["move"] * 6,
        }
    )

    with pytest.raises(IndexError, match=r"outside valid interval \[2, 4\]"):
        extract_a2a_step_data(episode, 0, configs, EmbodimentTag.NEW_EMBODIMENT)

    step = extract_a2a_step_data(episode, 2, configs, EmbodimentTag.NEW_EMBODIMENT)
    np.testing.assert_array_equal(step.states["joint"].ravel(), [0, 1, 2])
    np.testing.assert_array_equal(step.actions["joint"].ravel(), [102, 103])
    assert step.metadata["a2a_window"]["base_step_index"] == 2


def test_validation_rejects_late_future_access():
    with pytest.raises(IndexError, match="outside valid interval"):
        validate_a2a_step_index(6, 5, _a2a_modality_configs())


def test_delta_validation_rejects_empty_and_non_integer_values():
    with pytest.raises(ValueError, match="no delta_indices"):
        get_modality_delta_bounds(
            {"state": ModalityConfig(delta_indices=[], modality_keys=["joint"])}
        )
    with pytest.raises(TypeError, match="non-integer"):
        get_modality_delta_bounds(
            {"state": ModalityConfig(delta_indices=[0, 0.5], modality_keys=["joint"])}
        )


def test_shards_contain_only_absolute_valid_steps():
    configs = {
        "state": ModalityConfig(delta_indices=[-2, 0], modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=[0, 2], modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    with patch(
        "gr00t.data.dataset.sharded_single_step_dataset.LeRobotEpisodeLoader"
    ) as mock_loader_class:
        mock_loader = MagicMock()
        mock_loader.episode_lengths = [10]
        mock_loader.get_episode_length.return_value = 10
        mock_loader_class.return_value = mock_loader

        dataset = A2AShardedSingleStepDataset(
            dataset_path="/fake/dataset",
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=configs,
            shard_size=128,
            episode_sampling_rate=1.0,
            seed=7,
        )

    sampled_steps = sorted(
        int(step)
        for shard in dataset.sharded_episodes
        for _, step_indices in shard
        for step in step_indices
    )
    assert sampled_steps == list(range(2, 8))
    assert dataset.get_effective_episode_length(0) == 6


def test_a2a_training_dataset_rejects_padding():
    with pytest.raises(ValueError, match="allow_padding must be False"):
        A2AShardedSingleStepDataset(
            dataset_path="/unused",
            embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
            modality_configs=_a2a_modality_configs(),
            allow_padding=True,
        )


@pytest.mark.parametrize(
    ("model_config", "expected"),
    [
        (SimpleNamespace(model_type="Gr00tN1d7A2A"), True),
        (SimpleNamespace(model_type="other", a2a_history_horizon=8), True),
        (SimpleNamespace(model_type="Gr00tN1d7"), False),
        (MagicMock(), False),
    ],
)
def test_a2a_model_config_detection_is_explicit(model_config, expected):
    assert is_a2a_model_config(model_config) is expected
