# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Actual-proprio history buffering for action-to-action inference.

The A2A source trajectory must describe what the robot actually executed.  This
module therefore deliberately has no API for actions predicted or commanded by
the policy.  Callers must feed measured proprioception from robot/environment
observations through :meth:`A2AProprioHistoryBuffer.append_actual` or
:meth:`A2AProprioHistoryBuffer.resolve`.

All public histories use ``[batch, horizon, feature]`` layout.  The buffer is
model-independent and keeps each state key separate so a later processor can
perform embodiment-specific canonicalization.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias

import numpy as np


HistorySource: TypeAlias = Literal["external", "ring_buffer", "repeat_first_state"]
TimestampLike: TypeAlias = float | int | Sequence[float] | np.ndarray


@dataclass(frozen=True)
class A2AHistoryBatch:
    """A fixed-horizon batch of measured proprioception.

    Attributes:
        states: State arrays in ``[B, H, D_key]`` layout.  Arrays are defensive
            copies and may be safely modified by a downstream processor.
        timestamps: Per-batch timestamps in ``[B, H]`` layout.  Cold-start
            padding repeats the first real timestamp, just as it repeats the
            first real state.
        valid_mask: Boolean ``[B, H]`` mask.  ``False`` marks synthetic
            cold-start positions; external and full-buffer histories are all
            valid.
        metadata: Integration-friendly source and shape information.  In
            particular, ``metadata["source"]`` identifies whether the result
            came from an external history, a full ring buffer, or
            ``repeat_first_state`` cold-start padding.
    """

    states: dict[str, np.ndarray]
    timestamps: np.ndarray
    valid_mask: np.ndarray
    metadata: dict[str, Any]


class A2AProprioHistoryBuffer:
    """Fixed-horizon ring buffer containing only measured proprioception.

    Args:
        horizon: Number of historical samples returned to the A2A encoder.
        state_keys: Optional exact set/order of state keys.  If omitted, the
            first measured sample or external history establishes the schema.
        max_time_gap_s: Optional maximum allowed interval between consecutive
            measured samples.  Timestamps are always required, finite, and
            strictly increasing even when this limit is disabled.

    Notes:
        ``reset()`` starts a new episode and clears samples and timestamps while
        preserving the established key/feature schema.  Batch size may change
        after a reset.

        An external history is treated as an authoritative history of *actual
        measured proprioception*.  It takes priority over a simultaneously
        supplied current sample and re-seeds the local ring buffer, allowing a
        later local sample to continue from it.
    """

    def __init__(
        self,
        horizon: int,
        *,
        state_keys: Sequence[str] | None = None,
        max_time_gap_s: float | None = None,
    ) -> None:
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon <= 0:
            raise ValueError(f"horizon must be a positive integer, got {horizon!r}")
        if max_time_gap_s is not None:
            if (
                isinstance(max_time_gap_s, bool)
                or not np.isfinite(max_time_gap_s)
                or max_time_gap_s <= 0
            ):
                raise ValueError(
                    "max_time_gap_s must be a finite positive number or None, "
                    f"got {max_time_gap_s!r}"
                )

        configured_keys: tuple[str, ...] | None = None
        if state_keys is not None:
            if isinstance(state_keys, str):
                raise ValueError("state_keys must be a sequence of keys, not a single string")
            configured_keys = tuple(state_keys)
            if not configured_keys:
                raise ValueError("state_keys must contain at least one key")
            if any(not isinstance(key, str) or not key for key in configured_keys):
                raise ValueError("every state key must be a non-empty string")
            if len(configured_keys) != len(set(configured_keys)):
                raise ValueError(f"state_keys contains duplicates: {configured_keys!r}")

        self.horizon = horizon
        self.max_time_gap_s = max_time_gap_s
        self._state_keys = configured_keys
        self._state_dims: dict[str, int] = {}
        self._state_dtypes: dict[str, np.dtype[Any]] = {}

        self._buffers: dict[str, np.ndarray] = {}
        self._timestamp_buffer: np.ndarray | None = None
        self._batch_size: int | None = None
        self._size = 0
        self._write_index = 0

    @property
    def size(self) -> int:
        """Number of real measured samples currently stored."""

        return self._size

    @property
    def is_ready(self) -> bool:
        """Whether the ring contains ``horizon`` real samples (no cold padding)."""

        return self._size == self.horizon

    @property
    def state_keys(self) -> tuple[str, ...] | None:
        """Established state-key order, or ``None`` before schema inference."""

        return self._state_keys

    def reset(self) -> None:
        """Clear episode-local history without discarding the state schema."""

        self._buffers = {}
        self._timestamp_buffer = None
        self._batch_size = None
        self._size = 0
        self._write_index = 0

    def append_actual(
        self,
        actual_proprio: Mapping[str, np.ndarray],
        *,
        timestamp: TimestampLike,
    ) -> None:
        """Append one timestep of actually measured proprioception.

        Each value must have shape ``[B, D]`` or ``[B, 1, D]``.  A temporal
        dimension larger than one is rejected so a predicted action chunk
        cannot accidentally be passed through the ordinary update path.
        """

        states, keys, batch_size, dimensions, dtypes = self._validate_current(actual_proprio)
        timestamps = self._normalize_current_timestamps(timestamp, batch_size)
        self._validate_new_timestamp(timestamps)
        self._commit_schema(keys, batch_size, dimensions, dtypes)
        self._ensure_storage(states)

        for key in keys:
            self._buffers[key][self._write_index] = states[key]
        assert self._timestamp_buffer is not None
        self._timestamp_buffer[self._write_index] = timestamps

        self._write_index = (self._write_index + 1) % self.horizon
        self._size = min(self._size + 1, self.horizon)

    # A short alias is useful to policy integrations, while its keyword name
    # still makes the measured-only contract explicit at every call site.
    def append(
        self,
        actual_proprio: Mapping[str, np.ndarray],
        *,
        timestamp: TimestampLike,
    ) -> None:
        """Alias for :meth:`append_actual`; accepts measured proprio only."""

        self.append_actual(actual_proprio, timestamp=timestamp)

    def get_history(self) -> A2AHistoryBatch:
        """Return the ordered, fixed-horizon local history.

        Before the buffer fills, missing leading positions repeat the first
        measured state.  ``valid_mask`` and metadata distinguish this synthetic
        prefix from real observations.
        """

        if self._size == 0:
            raise RuntimeError("proprio history is empty; append a measured sample first")
        return self._snapshot(
            source="ring_buffer" if self.is_ready else "repeat_first_state",
            external_current_ignored=False,
        )

    def use_external_history(
        self,
        external_history: Mapping[str, np.ndarray],
        *,
        timestamps: TimestampLike,
    ) -> A2AHistoryBatch:
        """Validate and install an authoritative full measured history.

        ``external_history`` must contain exactly ``horizon`` samples per state
        key in ``[B, H, D]`` layout.  ``timestamps`` must be ``[H]`` (shared by
        the batch) or ``[B, H]``.  This method re-seeds the local ring buffer.
        """

        states, keys, batch_size, dimensions, dtypes = self._validate_external(external_history)
        external_timestamps = self._normalize_external_timestamps(timestamps, batch_size)
        self._validate_timestamp_sequence(external_timestamps)
        self._commit_schema(keys, batch_size, dimensions, dtypes)

        # External history is already oldest-to-newest.  With a full ring, index
        # zero is the next write position and therefore the oldest item.
        self._buffers = {
            key: np.ascontiguousarray(np.swapaxes(states[key], 0, 1)).copy() for key in keys
        }
        self._timestamp_buffer = np.ascontiguousarray(external_timestamps.T).copy()
        self._size = self.horizon
        self._write_index = 0
        return self._snapshot(source="external", external_current_ignored=False)

    def resolve(
        self,
        actual_proprio: Mapping[str, np.ndarray] | None = None,
        *,
        timestamp: TimestampLike | None = None,
        external_history: Mapping[str, np.ndarray] | None = None,
        external_timestamps: TimestampLike | None = None,
    ) -> A2AHistoryBatch:
        """Resolve the history for one inference request.

        A supplied full external measured history has priority and re-seeds the
        local ring.  Otherwise, this method appends ``actual_proprio`` and
        returns the local fixed-horizon snapshot.  ``actual_proprio`` is ignored
        when an external history is present; metadata records that fact.
        """

        if external_history is not None:
            if external_timestamps is None:
                raise ValueError("external_timestamps is required with external_history")
            result = self.use_external_history(
                external_history,
                timestamps=external_timestamps,
            )
            if actual_proprio is None:
                return result
            metadata = dict(result.metadata)
            metadata["external_current_ignored"] = True
            return A2AHistoryBatch(
                states=result.states,
                timestamps=result.timestamps,
                valid_mask=result.valid_mask,
                metadata=metadata,
            )

        if external_timestamps is not None:
            raise ValueError("external_timestamps was provided without external_history")
        if actual_proprio is None:
            raise ValueError("actual_proprio is required when external_history is absent")
        if timestamp is None:
            raise ValueError("timestamp is required when appending actual_proprio")

        self.append_actual(actual_proprio, timestamp=timestamp)
        return self.get_history()

    def _validate_current(
        self, states: Mapping[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[str, ...],
        int,
        dict[str, int],
        dict[str, np.dtype[Any]],
    ]:
        if not isinstance(states, Mapping) or not states:
            raise ValueError("actual_proprio must be a non-empty mapping of state arrays")
        keys = self._validate_keys(states)

        normalized: dict[str, np.ndarray] = {}
        dimensions: dict[str, int] = {}
        dtypes: dict[str, np.dtype[Any]] = {}
        batch_size: int | None = None
        for key in keys:
            value = self._validate_numeric_array(states[key], key=key)
            if value.ndim == 3:
                if value.shape[1] != 1:
                    raise ValueError(
                        f"actual proprio key {key!r} must contain one timestep; "
                        f"expected [B, D] or [B, 1, D], got {value.shape}"
                    )
                value = value[:, 0, :]
            elif value.ndim != 2:
                raise ValueError(
                    f"actual proprio key {key!r} must have shape [B, D] or [B, 1, D], "
                    f"got {value.shape}"
                )
            if value.shape[0] <= 0 or value.shape[1] <= 0:
                raise ValueError(
                    f"actual proprio key {key!r} has an empty dimension: {value.shape}"
                )
            batch_size = self._merge_batch_size(batch_size, value.shape[0], key)
            normalized[key] = np.ascontiguousarray(value)
            dimensions[key] = value.shape[1]
            dtypes[key] = value.dtype

        assert batch_size is not None
        self._validate_existing_schema(keys, batch_size, dimensions, dtypes)
        return normalized, keys, batch_size, dimensions, dtypes

    def _validate_external(
        self, states: Mapping[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        tuple[str, ...],
        int,
        dict[str, int],
        dict[str, np.dtype[Any]],
    ]:
        if not isinstance(states, Mapping) or not states:
            raise ValueError("external_history must be a non-empty mapping of state arrays")
        keys = self._validate_keys(states)

        normalized: dict[str, np.ndarray] = {}
        dimensions: dict[str, int] = {}
        dtypes: dict[str, np.dtype[Any]] = {}
        batch_size: int | None = None
        for key in keys:
            value = self._validate_numeric_array(states[key], key=key)
            if value.ndim != 3 or value.shape[1] != self.horizon:
                raise ValueError(
                    f"external history key {key!r} must have shape [B, {self.horizon}, D], "
                    f"got {value.shape}"
                )
            if value.shape[0] <= 0 or value.shape[2] <= 0:
                raise ValueError(
                    f"external history key {key!r} has an empty dimension: {value.shape}"
                )
            batch_size = self._merge_batch_size(batch_size, value.shape[0], key)
            normalized[key] = np.ascontiguousarray(value)
            dimensions[key] = value.shape[2]
            dtypes[key] = value.dtype

        assert batch_size is not None
        self._validate_existing_schema(keys, batch_size, dimensions, dtypes)
        return normalized, keys, batch_size, dimensions, dtypes

    def _validate_keys(self, states: Mapping[str, np.ndarray]) -> tuple[str, ...]:
        supplied = tuple(states.keys())
        if any(not isinstance(key, str) or not key for key in supplied):
            raise ValueError("every state key must be a non-empty string")
        expected = self._state_keys
        if expected is None:
            return supplied
        if set(supplied) != set(expected):
            missing = sorted(set(expected) - set(supplied))
            extra = sorted(set(supplied) - set(expected))
            raise ValueError(f"state keys do not match schema; missing={missing}, extra={extra}")
        return expected

    @staticmethod
    def _validate_numeric_array(value: np.ndarray, *, key: str) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError(f"state key {key!r} must be a numpy array, got {type(value).__name__}")
        if not np.issubdtype(value.dtype, np.floating):
            raise TypeError(f"state key {key!r} must have a floating dtype, got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"state key {key!r} contains NaN or infinity")
        return value

    @staticmethod
    def _merge_batch_size(current: int | None, candidate: int, key: str) -> int:
        if current is not None and current != candidate:
            raise ValueError(
                f"all state keys must share one batch size; key {key!r} has {candidate}, "
                f"expected {current}"
            )
        return candidate

    def _validate_existing_schema(
        self,
        keys: tuple[str, ...],
        batch_size: int,
        dimensions: Mapping[str, int],
        dtypes: Mapping[str, np.dtype[Any]],
    ) -> None:
        if self._state_keys is not None and keys != self._state_keys:
            raise ValueError(f"state-key order mismatch: got {keys}, expected {self._state_keys}")
        if self._batch_size is not None and batch_size != self._batch_size:
            raise ValueError(
                f"batch size changed from {self._batch_size} to {batch_size}; call reset() "
                "before starting a new batch/episode"
            )
        for key in keys:
            if key in self._state_dims and dimensions[key] != self._state_dims[key]:
                raise ValueError(
                    f"feature dimension for state key {key!r} changed from "
                    f"{self._state_dims[key]} to {dimensions[key]}"
                )
            if key in self._state_dtypes and dtypes[key] != self._state_dtypes[key]:
                raise TypeError(
                    f"dtype for state key {key!r} changed from {self._state_dtypes[key]} "
                    f"to {dtypes[key]}"
                )

    def _commit_schema(
        self,
        keys: tuple[str, ...],
        batch_size: int,
        dimensions: Mapping[str, int],
        dtypes: Mapping[str, np.dtype[Any]],
    ) -> None:
        if self._state_keys is None:
            self._state_keys = keys
        if not self._state_dims:
            self._state_dims = dict(dimensions)
            self._state_dtypes = dict(dtypes)
        self._batch_size = batch_size

    def _ensure_storage(self, states: Mapping[str, np.ndarray]) -> None:
        if self._buffers:
            return
        assert self._state_keys is not None
        assert self._batch_size is not None
        self._buffers = {
            key: np.empty(
                (self.horizon, self._batch_size, self._state_dims[key]),
                dtype=states[key].dtype,
            )
            for key in self._state_keys
        }
        self._timestamp_buffer = np.empty((self.horizon, self._batch_size), dtype=np.float64)

    @staticmethod
    def _normalize_current_timestamps(timestamp: TimestampLike, batch_size: int) -> np.ndarray:
        value = np.asarray(timestamp, dtype=np.float64)
        if value.ndim == 0:
            value = np.full((batch_size,), value.item(), dtype=np.float64)
        elif value.shape != (batch_size,):
            raise ValueError(
                f"current timestamp must be scalar or shape [{batch_size}], got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("timestamp contains NaN or infinity")
        return value

    def _normalize_external_timestamps(
        self, timestamps: TimestampLike, batch_size: int
    ) -> np.ndarray:
        value = np.asarray(timestamps, dtype=np.float64)
        if value.shape == (self.horizon,):
            value = np.broadcast_to(value[None, :], (batch_size, self.horizon)).copy()
        elif value.shape != (batch_size, self.horizon):
            raise ValueError(
                "external timestamps must have shape "
                f"[{self.horizon}] or [{batch_size}, {self.horizon}], got {value.shape}"
            )
        if not np.all(np.isfinite(value)):
            raise ValueError("external timestamps contain NaN or infinity")
        return value

    def _validate_new_timestamp(self, timestamps: np.ndarray) -> None:
        if self._size == 0:
            return
        assert self._timestamp_buffer is not None
        previous = self._timestamp_buffer[(self._write_index - 1) % self.horizon]
        gaps = timestamps - previous
        if np.any(gaps <= 0):
            bad = np.flatnonzero(gaps <= 0).tolist()
            raise ValueError(
                "timestamps must be strictly increasing for every batch item; "
                f"invalid batch indices={bad}"
            )
        self._validate_max_gap(gaps)

    def _validate_timestamp_sequence(self, timestamps: np.ndarray) -> None:
        if self.horizon <= 1:
            return
        gaps = np.diff(timestamps, axis=1)
        if np.any(gaps <= 0):
            bad = np.argwhere(gaps <= 0).tolist()
            raise ValueError(
                "external timestamps must be strictly increasing along history; "
                f"invalid [batch, interval] indices={bad}"
            )
        self._validate_max_gap(gaps)

    def _validate_max_gap(self, gaps: np.ndarray) -> None:
        if self.max_time_gap_s is None:
            return
        if np.any(gaps > self.max_time_gap_s):
            bad = np.argwhere(gaps > self.max_time_gap_s).tolist()
            maximum = float(np.max(gaps))
            raise ValueError(
                f"timestamp gap {maximum:.9g}s exceeds max_time_gap_s="
                f"{self.max_time_gap_s:.9g}; invalid indices={bad}"
            )

    def _ordered_indices(self) -> np.ndarray:
        if self._size < self.horizon:
            return np.arange(self._size)
        return (np.arange(self.horizon) + self._write_index) % self.horizon

    def _snapshot(
        self,
        *,
        source: HistorySource,
        external_current_ignored: bool,
    ) -> A2AHistoryBatch:
        assert self._state_keys is not None
        assert self._batch_size is not None
        assert self._timestamp_buffer is not None

        indices = self._ordered_indices()
        ordered_states = {
            key: np.swapaxes(self._buffers[key][indices], 0, 1) for key in self._state_keys
        }
        ordered_timestamps = self._timestamp_buffer[indices].T
        padded_steps = self.horizon - self._size

        if padded_steps:
            states = {}
            for key, value in ordered_states.items():
                prefix = np.repeat(value[:, :1, :], padded_steps, axis=1)
                states[key] = np.ascontiguousarray(np.concatenate((prefix, value), axis=1))
            timestamp_prefix = np.repeat(
                ordered_timestamps[:, :1],
                padded_steps,
                axis=1,
            )
            result_timestamps = np.ascontiguousarray(
                np.concatenate((timestamp_prefix, ordered_timestamps), axis=1)
            )
            valid_mask = np.concatenate(
                (
                    np.zeros((self._batch_size, padded_steps), dtype=bool),
                    np.ones((self._batch_size, self._size), dtype=bool),
                ),
                axis=1,
            )
        else:
            states = {
                key: np.ascontiguousarray(value).copy()
                for key, value in ordered_states.items()
            }
            result_timestamps = np.ascontiguousarray(ordered_timestamps).copy()
            valid_mask = np.ones((self._batch_size, self.horizon), dtype=bool)

        metadata: dict[str, Any] = {
            "source": source,
            "history_horizon": self.horizon,
            "valid_steps": self._size,
            "padded_steps": padded_steps,
            "batch_size": self._batch_size,
            "state_keys": self._state_keys,
            "timestamps_validated": True,
            "max_time_gap_s": self.max_time_gap_s,
            "external_current_ignored": external_current_ignored,
        }
        return A2AHistoryBatch(
            states=states,
            timestamps=result_timestamps,
            valid_mask=np.ascontiguousarray(valid_mask),
            metadata=metadata,
        )


__all__ = ["A2AHistoryBatch", "A2AProprioHistoryBuffer", "HistorySource"]
