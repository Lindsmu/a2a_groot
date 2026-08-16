# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from gr00t.policy.a2a_history import A2AProprioHistoryBuffer
import numpy as np
import pytest


def _actual(value: float, batch_size: int = 2) -> dict[str, np.ndarray]:
    return {
        "arm": np.full((batch_size, 2), value, dtype=np.float32),
        "gripper": np.full((batch_size, 1, 1), value + 100, dtype=np.float32),
    }


def _external(start: float, batch_size: int = 2, horizon: int = 3) -> dict[str, np.ndarray]:
    steps = np.arange(start, start + horizon, dtype=np.float32)
    arm = np.broadcast_to(steps[None, :, None], (batch_size, horizon, 2)).copy()
    gripper = np.broadcast_to((steps + 100)[None, :, None], (batch_size, horizon, 1)).copy()
    return {"arm": arm, "gripper": gripper}


def test_repeat_first_state_cold_start_and_source_metadata() -> None:
    buffer = A2AProprioHistoryBuffer(3, state_keys=("arm", "gripper"))

    result = buffer.resolve(_actual(7), timestamp=np.array([1.0, 2.0]))

    assert result.metadata == {
        "source": "repeat_first_state",
        "history_horizon": 3,
        "valid_steps": 1,
        "padded_steps": 2,
        "batch_size": 2,
        "state_keys": ("arm", "gripper"),
        "timestamps_validated": True,
        "max_time_gap_s": None,
        "external_current_ignored": False,
    }
    np.testing.assert_array_equal(result.states["arm"], 7)
    np.testing.assert_array_equal(result.states["gripper"], 107)
    np.testing.assert_array_equal(
        result.timestamps,
        np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
    )
    np.testing.assert_array_equal(
        result.valid_mask,
        np.array([[False, False, True], [False, False, True]]),
    )
    assert not buffer.is_ready


def test_multi_key_ring_buffer_is_ordered_and_rolls_over() -> None:
    buffer = A2AProprioHistoryBuffer(3, state_keys=("arm", "gripper"))
    for step in range(4):
        buffer.append_actual(_actual(step), timestamp=float(step))

    result = buffer.get_history()

    assert buffer.size == 3
    assert buffer.is_ready
    assert result.metadata["source"] == "ring_buffer"
    assert result.metadata["valid_steps"] == 3
    np.testing.assert_array_equal(result.states["arm"][0, :, 0], [1, 2, 3])
    np.testing.assert_array_equal(result.states["gripper"][1, :, 0], [101, 102, 103])
    np.testing.assert_array_equal(result.timestamps, [[1, 2, 3], [1, 2, 3]])
    assert result.valid_mask.all()


def test_external_full_history_has_priority_and_reseeds_ring() -> None:
    buffer = A2AProprioHistoryBuffer(3, state_keys=("arm", "gripper"))
    buffer.append(_actual(-1), timestamp=0.0)
    external = _external(10)

    result = buffer.resolve(
        _actual(999),
        timestamp=999.0,
        external_history=external,
        external_timestamps=np.array([10.0, 11.0, 12.0]),
    )

    assert result.metadata["source"] == "external"
    assert result.metadata["external_current_ignored"] is True
    np.testing.assert_array_equal(result.states["arm"], external["arm"])
    np.testing.assert_array_equal(result.timestamps, [[10, 11, 12], [10, 11, 12]])

    # The authoritative external sequence becomes the local ring's continuity
    # anchor; the ignored current prediction/sample never enters the history.
    buffer.append_actual(_actual(13), timestamp=13.0)
    continued = buffer.get_history()
    np.testing.assert_array_equal(continued.states["arm"][0, :, 0], [11, 12, 13])
    assert 999 not in continued.states["arm"]


def test_external_history_requires_exact_horizon_and_timestamp_shape() -> None:
    buffer = A2AProprioHistoryBuffer(3)

    with pytest.raises(ValueError, match=r"shape \[B, 3, D\]"):
        buffer.use_external_history(_external(0, horizon=2), timestamps=[0, 1, 2])
    with pytest.raises(ValueError, match="external timestamps must have shape"):
        buffer.use_external_history(_external(0), timestamps=[0, 1])
    with pytest.raises(ValueError, match="external_timestamps is required"):
        buffer.resolve(external_history=_external(0))
    with pytest.raises(ValueError, match="without external_history"):
        buffer.resolve(_actual(0), timestamp=0, external_timestamps=[0, 1, 2])


def test_batch_is_checked_across_keys_and_updates() -> None:
    buffer = A2AProprioHistoryBuffer(3, state_keys=("arm", "gripper"))
    mismatched = _actual(0)
    mismatched["gripper"] = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(ValueError, match="share one batch size"):
        buffer.append_actual(mismatched, timestamp=0)

    buffer.append_actual(_actual(0), timestamp=[0, 0])
    with pytest.raises(ValueError, match="batch size changed"):
        buffer.append_actual(_actual(1, batch_size=1), timestamp=1)


def test_state_schema_shape_dtype_and_values_are_checked() -> None:
    buffer = A2AProprioHistoryBuffer(3, state_keys=("arm", "gripper"))

    with pytest.raises(ValueError, match="missing=.*gripper"):
        buffer.append_actual({"arm": _actual(0)["arm"]}, timestamp=0)
    with pytest.raises(ValueError, match="one timestep"):
        buffer.append_actual(
            {
                "arm": np.zeros((2, 2, 2), dtype=np.float32),
                "gripper": _actual(0)["gripper"],
            },
            timestamp=0,
        )
    with pytest.raises(TypeError, match="floating dtype"):
        integer_state = {key: value.astype(np.int32) for key, value in _actual(0).items()}
        buffer.append_actual(integer_state, timestamp=0)
    invalid = _actual(0)
    invalid["arm"][0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        buffer.append_actual(invalid, timestamp=0)

    buffer.append_actual(_actual(0), timestamp=[0, 0])
    changed_dim = _actual(1)
    changed_dim["arm"] = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="feature dimension"):
        buffer.append_actual(changed_dim, timestamp=[1, 1])


def test_timestamp_is_strictly_monotonic_per_batch() -> None:
    buffer = A2AProprioHistoryBuffer(3)
    buffer.append_actual(_actual(0), timestamp=[1.0, 10.0])

    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append_actual(_actual(1), timestamp=[1.0, 11.0])
    with pytest.raises(ValueError, match="strictly increasing"):
        buffer.append_actual(_actual(1), timestamp=[2.0, 9.0])
    with pytest.raises(ValueError, match="NaN or infinity"):
        buffer.append_actual(_actual(1), timestamp=[2.0, np.nan])

    # Failed validation is transactional: no rejected sample was appended.
    assert buffer.size == 1
    np.testing.assert_array_equal(buffer.get_history().states["arm"], 0)


def test_max_time_gap_is_checked_for_local_and_external_histories() -> None:
    buffer = A2AProprioHistoryBuffer(3, max_time_gap_s=0.25)
    buffer.append_actual(_actual(0), timestamp=[0.0, 1.0])
    buffer.append_actual(_actual(1), timestamp=[0.2, 1.2])
    with pytest.raises(ValueError, match="exceeds max_time_gap_s"):
        buffer.append_actual(_actual(2), timestamp=[0.3, 1.6])

    buffer.reset()
    with pytest.raises(ValueError, match="strictly increasing along history"):
        buffer.use_external_history(_external(0), timestamps=[0.0, 0.2, 0.2])
    with pytest.raises(ValueError, match="exceeds max_time_gap_s"):
        buffer.use_external_history(_external(0), timestamps=[0.0, 0.2, 0.5])


def test_reset_clears_episode_and_allows_new_batch_size() -> None:
    buffer = A2AProprioHistoryBuffer(3)
    buffer.append_actual(_actual(0, batch_size=2), timestamp=[0, 0])

    buffer.reset()

    assert buffer.size == 0
    assert not buffer.is_ready
    with pytest.raises(RuntimeError, match="history is empty"):
        buffer.get_history()
    result = buffer.resolve(_actual(5, batch_size=1), timestamp=5)
    assert result.states["arm"].shape == (1, 3, 2)
    assert result.metadata["source"] == "repeat_first_state"


def test_returned_arrays_do_not_alias_internal_or_external_storage() -> None:
    buffer = A2AProprioHistoryBuffer(3)
    external = _external(0)
    result = buffer.use_external_history(external, timestamps=[0, 1, 2])

    external["arm"].fill(-10)
    result.states["arm"].fill(50)
    result.timestamps.fill(50)

    fresh = buffer.get_history()
    np.testing.assert_array_equal(fresh.states["arm"][0, :, 0], [0, 1, 2])
    np.testing.assert_array_equal(fresh.timestamps[0], [0, 1, 2])


def test_prediction_specific_keyword_is_not_accepted() -> None:
    buffer = A2AProprioHistoryBuffer(3)

    with pytest.raises(TypeError, match="predicted_actions"):
        buffer.resolve(predicted_actions=_actual(0), timestamp=0)  # type: ignore[call-arg]
