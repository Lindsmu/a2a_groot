# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Processor for canonical executed-history to future-action trajectories."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoProcessor

from gr00t.data.a2a_contract import canonical_statistics_sha256, validate_channel_contract
from gr00t.data.types import A2AChannelSpec
from gr00t.data.utils import normalize_values_minmax, unnormalize_values_minmax
from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7Processor


_RAW_LIBERO_CONTROLLER_KEYS = {"x", "y", "z", "roll", "pitch", "yaw", "gripper"}


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


class Gr00tN1d7A2AProcessor(Gr00tN1d7Processor):
    """Build source and target trajectories in one shared, absolute space.

    Strict training requires explicit :class:`A2AChannelSpec` metadata and a
    hash-bound canonical statistics artifact. The equal-dimension inference
    fallback is retained only for deliberately non-strict exploratory runs.
    """

    def __init__(
        self,
        *args,
        a2a_history_horizon: int = 8,
        a2a_future_horizon: int = 8,
        a2a_channel_specs: dict[str, list[dict[str, Any]]] | None = None,
        a2a_cold_start: str = "repeat_first_state",
        a2a_canonical_statistics: dict[str, Any] | None = None,
        a2a_canonical_statistics_path: str | Path | None = None,
        a2a_require_canonical_statistics: bool = False,
        a2a_require_explicit_channel_specs: bool = False,
        a2a_require_semantic_metadata: bool = False,
        a2a_expected_contract_sha256: str | None = None,
        **kwargs,
    ):
        self.a2a_history_horizon = int(a2a_history_horizon)
        self.a2a_future_horizon = int(a2a_future_horizon)
        if self.a2a_history_horizon <= 0 or self.a2a_future_horizon <= 0:
            raise ValueError("A2A history/future horizons must be positive")
        if self.a2a_history_horizon != self.a2a_future_horizon:
            raise ValueError(
                "The strict latent A2A implementation requires equal history/future horizons"
            )
        if a2a_cold_start not in {"repeat_first_state", "require_full_history"}:
            raise ValueError(f"Unsupported A2A cold-start mode: {a2a_cold_start}")
        self.a2a_cold_start = a2a_cold_start
        if a2a_canonical_statistics is not None and a2a_canonical_statistics_path is not None:
            raise ValueError(
                "Provide either embedded a2a_canonical_statistics or "
                "a2a_canonical_statistics_path, not both"
            )
        if a2a_canonical_statistics_path is not None:
            statistics_path = Path(a2a_canonical_statistics_path)
            with statistics_path.open("r", encoding="utf-8") as handle:
                a2a_canonical_statistics = json.load(handle)
        self.a2a_canonical_statistics = a2a_canonical_statistics
        self.a2a_require_canonical_statistics = bool(a2a_require_canonical_statistics)
        self.a2a_require_explicit_channel_specs = bool(a2a_require_explicit_channel_specs)
        self.a2a_require_semantic_metadata = bool(a2a_require_semantic_metadata)
        self.a2a_expected_contract_sha256 = a2a_expected_contract_sha256
        self._configured_channel_specs = a2a_channel_specs or {}
        self._resolved_channel_specs: dict[str, list[A2AChannelSpec]] = {}
        self.a2a_statistics: dict[str, dict[str, dict[str, list[float]]]] = {}
        self.a2a_norm_params: dict[str, dict[str, dict[str, np.ndarray]]] = {}
        super().__init__(*args, **kwargs)
        if self.state_action_processor.statistics:
            self._rebuild_a2a_parameters(self.state_action_processor.statistics)

    def set_statistics(self, statistics: dict, override: bool = False) -> None:
        super().set_statistics(statistics, override=override)
        self._rebuild_a2a_parameters(self.state_action_processor.statistics)

    def _explicit_specs(self, embodiment: str) -> list[A2AChannelSpec] | None:
        configured = self._configured_channel_specs.get(embodiment)
        if configured is None:
            return None
        return [spec if isinstance(spec, A2AChannelSpec) else A2AChannelSpec(**spec) for spec in configured]

    @staticmethod
    def _dimension(stats: dict[str, Any]) -> int:
        return int(np.asarray(stats["mean"]).shape[-1])

    def _resolve_specs(self, embodiment: str, statistics: dict) -> list[A2AChannelSpec]:
        action_cfg = self.modality_configs[embodiment]["action"]
        action_keys = action_cfg.modality_keys
        action_configs = action_cfg.action_configs
        explicit = self._explicit_specs(embodiment)
        if self.a2a_require_explicit_channel_specs and explicit is None:
            raise ValueError(
                f"Strict A2A requires explicit a2a_channel_specs for {embodiment!r}; "
                "equal tensor dimensions do not prove matching physical semantics"
            )
        explicit_by_key = {spec.action_key: spec for spec in explicit or []}
        unknown = set(explicit_by_key) - set(action_keys)
        if unknown:
            raise ValueError(f"A2A specs reference unknown action keys for {embodiment}: {sorted(unknown)}")
        if self.a2a_require_explicit_channel_specs:
            missing = set(action_keys) - set(explicit_by_key)
            if missing:
                raise ValueError(
                    f"Strict A2A specs do not classify action keys for {embodiment}: "
                    f"{sorted(missing)}"
                )

        resolved: list[A2AChannelSpec] = []
        for index, action_key in enumerate(action_keys):
            configured = explicit_by_key.get(action_key)
            action_config = action_configs[index] if action_configs is not None else None
            source_key = (
                configured.source_state_key
                if configured is not None and configured.source_state_key is not None
                else getattr(action_config, "state_key", None) or action_key
            )
            action_stats = statistics[embodiment]["action"][action_key]
            action_dim = self._dimension(action_stats)
            state_stats = statistics[embodiment].get("state", {}).get(source_key)
            state_dim = self._dimension(state_stats) if state_stats is not None else None
            if configured is not None and configured.dim is not None:
                if int(configured.dim) != action_dim:
                    raise ValueError(
                        f"A2A spec {embodiment}/{action_key} declares dim={configured.dim}, "
                        f"but dataset statistics report {action_dim}"
                    )

            kind = configured.kind if configured is not None else "auto"
            if self.a2a_require_explicit_channel_specs and kind == "auto":
                raise ValueError(
                    f"Strict A2A spec {embodiment}/{action_key} must choose an explicit kind"
                )
            if kind == "auto":
                kind = "continuous" if state_dim == action_dim else "regression"
            if kind == "continuous" and state_dim != action_dim:
                raise ValueError(
                    f"A2A continuous channel {embodiment}/{action_key} requires matching state "
                    f"dimension, got state {source_key!r}={state_dim}, action={action_dim}"
                )
            if (
                kind == "continuous"
                and self.a2a_require_semantic_metadata
                and embodiment == "libero_sim"
                and action_key in _RAW_LIBERO_CONTROLLER_KEYS
            ):
                raise ValueError(
                    "Raw LIBERO state columns are absolute measured EEF pose while its "
                    "same-named action columns are normalized OSC delta commands. Strict "
                    "A2A requires newly preprocessed canonical column names and a versioned "
                    "forward/inverse controller adapter."
                )
            if kind == "continuous" and self.a2a_require_semantic_metadata:
                assert configured is not None
                semantic_pairs = {
                    "format": (configured.source_format, configured.target_format),
                    "unit": (configured.source_unit, configured.target_unit),
                    "frame": (configured.source_frame, configured.target_frame),
                }
                for semantic, (source_value, target_value) in semantic_pairs.items():
                    if source_value is None or target_value is None:
                        raise ValueError(
                            f"Strict continuous channel {embodiment}/{action_key} must declare "
                            f"source_{semantic} and target_{semantic}"
                        )
                    if source_value != target_value:
                        raise ValueError(
                            f"Continuous channel {embodiment}/{action_key} has mismatched "
                            f"{semantic}: source={source_value!r}, target={target_value!r}"
                        )
                if configured.canonical_format == "auto":
                    raise ValueError(
                        f"Strict continuous channel {embodiment}/{action_key} must declare "
                        "canonical_format"
                    )
                if configured.canonical_format != configured.source_format:
                    raise ValueError(
                        f"Canonical format for {embodiment}/{action_key} must equal the "
                        "already-canonical source/target format"
                    )
            if kind == "categorical":
                if configured is None or configured.num_classes is None:
                    raise ValueError(
                        f"Categorical A2A channel {embodiment}/{action_key} requires num_classes"
                    )
                if action_dim != configured.num_classes:
                    raise ValueError(
                        "Categorical channels must be stored as one-hot action groups whose "
                        "dimension equals num_classes"
                    )
            if kind == "unsupported":
                raise ValueError(
                    f"Action channel {embodiment}/{action_key} is marked unsupported; remove it "
                    "from the action modality or choose an explicit auxiliary kind"
                )
            resolved.append(
                A2AChannelSpec(
                    action_key=action_key,
                    source_state_key=source_key,
                    kind=kind,
                    canonical_format=(configured.canonical_format if configured else "auto"),
                    source_format=(configured.source_format if configured else None),
                    target_format=(configured.target_format if configured else None),
                    source_unit=(configured.source_unit if configured else None),
                    target_unit=(configured.target_unit if configured else None),
                    source_frame=(configured.source_frame if configured else None),
                    target_frame=(configured.target_frame if configured else None),
                    dim=action_dim,
                    num_classes=(configured.num_classes if configured else None),
                )
            )
        if not any(spec.kind == "continuous" for spec in resolved):
            raise ValueError(
                f"Latent A2A requires at least one executed-history continuous channel for "
                f"{embodiment!r}"
            )
        return resolved

    def _bounds(self, stats: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        lower_key, upper_key = ("q01", "q99") if self.use_percentiles else ("min", "max")
        return np.asarray(stats[lower_key], dtype=np.float32), np.asarray(
            stats[upper_key], dtype=np.float32
        )

    def _canonical_statistics_for_embodiment(
        self, embodiment: str
    ) -> dict[str, Any] | None:
        statistics = self.a2a_canonical_statistics
        if statistics is None:
            return None
        if "feature_dim" in statistics:
            configured_embodiments = [
                name
                for name in self.modality_configs
                if name in self.state_action_processor.statistics
            ]
            if len(configured_embodiments) > 1:
                raise ValueError(
                    "A flat canonical statistics file can only be used with one embodiment; "
                    "provide an embodiment-keyed mapping for mixtures"
                )
            return statistics
        selected = statistics.get(embodiment)
        if selected is None:
            raise ValueError(
                f"Canonical A2A statistics do not contain embodiment {embodiment!r}"
            )
        if not isinstance(selected, dict) or "feature_dim" not in selected:
            raise ValueError(
                f"Invalid canonical statistics entry for embodiment {embodiment!r}"
            )
        return selected

    def _canonical_bounds(
        self, statistics: dict[str, Any], start: int, end: int, embodiment: str, key: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if statistics.get("input_space") != "unnormalized_canonical_physical":
            raise ValueError(
                "Canonical A2A statistics must be built from unnormalized physical trajectories"
            )
        feature_dim = int(statistics.get("feature_dim", -1))
        if feature_dim < end:
            raise ValueError(
                f"Canonical statistics feature_dim={feature_dim} does not cover "
                f"{embodiment}/{key} at channels [{start}:{end}]"
            )
        active = np.asarray(statistics.get("active_channel_mask"), dtype=bool)
        if active.shape != (feature_dim,) or not np.all(active[start:end]):
            raise ValueError(
                f"Canonical statistics mark continuous channel {embodiment}/{key} inactive"
            )
        if self.use_percentiles:
            lower_values = statistics.get("q_low", statistics.get("q01"))
            upper_values = statistics.get("q_high", statistics.get("q99"))
        else:
            lower_values = statistics.get("min")
            upper_values = statistics.get("max")
        lower = np.asarray(lower_values, dtype=np.float32)
        upper = np.asarray(upper_values, dtype=np.float32)
        if lower.shape != (feature_dim,) or upper.shape != (feature_dim,):
            raise ValueError("Canonical A2A bounds must be one vector per feature")
        if not np.all(np.isfinite(lower[start:end])) or not np.all(
            np.isfinite(upper[start:end])
        ):
            raise ValueError(f"Canonical bounds for {embodiment}/{key} are non-finite")
        if np.any(upper[start:end] < lower[start:end]):
            raise ValueError(f"Canonical bounds for {embodiment}/{key} are reversed")
        return lower[start:end], upper[start:end]

    def _validate_canonical_channel_contract(
        self,
        statistics: dict[str, Any],
        embodiment: str,
        specs: list[A2AChannelSpec],
    ) -> None:
        if statistics.get("version") != 1:
            raise ValueError(
                f"Unsupported canonical A2A statistics version: {statistics.get('version')!r}"
            )
        if statistics.get("normalization") != (
            "shared_history_future_continuous_intersection"
        ):
            raise ValueError("Canonical A2A statistics use an unexpected normalization contract")
        if int(statistics.get("history_horizon", -1)) != self.a2a_history_horizon or int(
            statistics.get("future_horizon", -1)
        ) != self.a2a_future_horizon:
            raise ValueError(
                "Canonical A2A statistics horizons do not match the processor configuration"
            )
        if int(statistics.get("feature_dim", -1)) != self.max_action_dim:
            raise ValueError(
                "Canonical A2A statistics feature_dim must equal processor max_action_dim"
            )
        if int(statistics.get("num_windows", 0)) <= 0:
            raise ValueError("Canonical A2A statistics must contain at least one window")
        if int(statistics.get("nonfinite_active_count", -1)) != 0:
            raise ValueError("Canonical A2A statistics report non-finite active values")
        stored_statistics_hash = statistics.get("statistics_sha256")
        if stored_statistics_hash != canonical_statistics_sha256(statistics):
            raise ValueError("Canonical A2A statistics SHA-256 does not match the payload")
        feature_dim = int(statistics["feature_dim"])
        vector_keys = (
            "mean",
            "std",
            "min",
            "max",
            "q_low",
            "q_high",
            "valid_count",
            "history_valid_count",
            "future_valid_count",
            "active_channel_mask",
            "constant_mask",
        )
        vectors = {key: np.asarray(statistics.get(key)) for key in vector_keys}
        if any(vector.shape != (feature_dim,) for vector in vectors.values()):
            raise ValueError("Canonical A2A statistics vectors must all match feature_dim")
        for key in ("mean", "std", "min", "max", "q_low", "q_high"):
            if not np.all(np.isfinite(vectors[key].astype(np.float64))):
                raise ValueError(f"Canonical A2A statistics {key} contains non-finite values")
        if np.any(vectors["std"].astype(np.float64) <= 0):
            raise ValueError("Canonical A2A statistics std must be positive")
        for key in ("valid_count", "history_valid_count", "future_valid_count"):
            values = vectors[key].astype(np.float64)
            if (
                not np.all(np.isfinite(values))
                or np.any(values < 0)
                or not np.all(values == np.floor(values))
            ):
                raise ValueError(f"Canonical A2A statistics {key} must be non-negative integers")
        for key in ("active_channel_mask", "constant_mask"):
            values = vectors[key]
            if not (
                np.issubdtype(values.dtype, np.bool_)
                or (
                    np.issubdtype(values.dtype, np.number)
                    and np.all(np.isfinite(values))
                    and np.all(np.isin(values, [0, 1]))
                )
            ):
                raise ValueError(f"Canonical A2A statistics {key} must be boolean/0-1")
        minimum = vectors["min"].astype(np.float64)
        maximum = vectors["max"].astype(np.float64)
        q_low = vectors["q_low"].astype(np.float64)
        q_high = vectors["q_high"].astype(np.float64)
        if np.any(minimum > q_low) or np.any(q_low > q_high) or np.any(q_high > maximum):
            raise ValueError("Canonical A2A quantiles must lie monotonically inside min/max")
        active = vectors["active_channel_mask"].astype(bool)
        valid_count = vectors["valid_count"].astype(np.int64)
        history_count = vectors["history_valid_count"].astype(np.int64)
        future_count = vectors["future_valid_count"].astype(np.int64)
        if np.any(valid_count[active] <= 0) or np.any(history_count[active] <= 0) or np.any(
            future_count[active] <= 0
        ):
            raise ValueError("Every active canonical channel needs history and future samples")
        if np.any(valid_count != history_count + future_count):
            raise ValueError("Canonical valid_count must equal history_count + future_count")
        if (
            np.any(valid_count[~active] != 0)
            or np.any(history_count[~active] != 0)
            or np.any(future_count[~active] != 0)
        ):
            raise ValueError("Inactive canonical channels must have zero sample counts")
        if np.any(vectors["constant_mask"].astype(bool) & active):
            raise ValueError("Strict A2A rejects active constant canonical channels")
        contract = statistics.get("channel_contract")
        if contract is None:
            if self.a2a_require_semantic_metadata:
                raise ValueError(
                    "Strict A2A canonical statistics must embed a versioned channel_contract"
                )
            return
        contract = validate_channel_contract(
            contract,
            feature_dim=int(statistics["feature_dim"]),
            history_horizon=self.a2a_history_horizon,
            future_horizon=self.a2a_future_horizon,
        )
        if self.a2a_require_semantic_metadata:
            if self.a2a_expected_contract_sha256 is None:
                raise ValueError(
                    "Strict A2A requires a2a_expected_contract_sha256 in the model config"
                )
            if contract["sha256"] != self.a2a_expected_contract_sha256:
                raise ValueError(
                    "Canonical statistics contract does not match "
                    "a2a_expected_contract_sha256"
                )
        if contract["embodiment"] != embodiment:
            raise ValueError(
                f"Canonical statistics are bound to {contract['embodiment']!r}, "
                f"not {embodiment!r}"
            )
        expected_channels = []
        cursor = 0
        for spec in specs:
            end = cursor + int(spec.dim)
            expected_channels.append(
                {
                    "action_key": spec.action_key,
                    "source_state_key": spec.source_state_key,
                    "kind": spec.kind,
                    "start": cursor,
                    "end": end,
                    "canonical_format": spec.canonical_format,
                    "source_format": spec.source_format,
                    "target_format": spec.target_format,
                    "source_unit": spec.source_unit,
                    "target_unit": spec.target_unit,
                    "source_frame": spec.source_frame,
                    "target_frame": spec.target_frame,
                }
            )
            cursor = end
        if contract["channels"] != expected_channels:
            raise ValueError(
                "Canonical statistics channel order/semantics do not match a2a_channel_specs"
            )

    def _rebuild_a2a_parameters(self, statistics: dict) -> None:
        self._resolved_channel_specs = {}
        self.a2a_statistics = {}
        self.a2a_norm_params = {}
        for embodiment in self.modality_configs:
            if embodiment not in statistics or "action" not in statistics[embodiment]:
                continue
            specs = self._resolve_specs(embodiment, statistics)
            canonical_statistics = self._canonical_statistics_for_embodiment(embodiment)
            if canonical_statistics is None and self.a2a_require_canonical_statistics:
                raise ValueError(
                    "Strict latent A2A requires canonical statistics generated by "
                    "scripts/a2a/build_canonical_stats.py; set "
                    "model.a2a_canonical_statistics_path"
                )
            if canonical_statistics is not None:
                self._validate_canonical_channel_contract(
                    canonical_statistics, embodiment, specs
                )
            self._resolved_channel_specs[embodiment] = specs
            self.a2a_statistics[embodiment] = {}
            self.a2a_norm_params[embodiment] = {}
            cursor = 0
            for spec in specs:
                action_stats = statistics[embodiment]["action"][spec.action_key]
                action_min, action_max = self._bounds(action_stats)
                if spec.kind == "continuous":
                    if canonical_statistics is not None:
                        lower, upper = self._canonical_bounds(
                            canonical_statistics,
                            cursor,
                            cursor + int(spec.dim),
                            embodiment,
                            spec.action_key,
                        )
                    else:
                        # Explicit non-strict fallback for exploratory runs.
                        # Both sides still use these shared union bounds.
                        state_stats = statistics[embodiment]["state"][spec.source_state_key]
                        state_min, state_max = self._bounds(state_stats)
                        lower = np.minimum(state_min, action_min)
                        upper = np.maximum(state_max, action_max)
                else:
                    lower, upper = action_min, action_max
                params = {
                    "min": lower.astype(np.float32),
                    "max": upper.astype(np.float32),
                }
                self.a2a_norm_params[embodiment][spec.action_key] = params
                self.a2a_statistics[embodiment][spec.action_key] = {
                    "min": params["min"].tolist(),
                    "max": params["max"].tolist(),
                    "kind": spec.kind,
                    "source_state_key": spec.source_state_key,
                }
                cursor += int(spec.dim)

    def _require_parameters(self, embodiment: str) -> list[A2AChannelSpec]:
        if embodiment not in self._resolved_channel_specs:
            raise RuntimeError(
                f"A2A statistics for {embodiment!r} are not initialized. Build the dataset or "
                "load a checkpoint processor before processing samples."
            )
        return self._resolved_channel_specs[embodiment]

    def _normalize(self, values: np.ndarray, embodiment: str, key: str) -> np.ndarray:
        normalized = normalize_values_minmax(
            values.astype(np.float32), self.a2a_norm_params[embodiment][key]
        )
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)

    def _history_window(
        self, values: np.ndarray, supplied_valid_time: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError(f"Expected state trajectory [T,D], got {values.shape}")
        if values.shape[0] == 0:
            raise ValueError("A2A state trajectory cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("A2A state trajectory contains NaN or infinity")
        horizon = self.a2a_history_horizon
        if supplied_valid_time is not None:
            supplied_valid_time = np.asarray(supplied_valid_time)
            if supplied_valid_time.shape != (values.shape[0],):
                raise ValueError(
                    "A2A history valid mask must have one entry per supplied state timestep"
                )
            if not np.all(np.isin(supplied_valid_time, [0, 1])):
                raise ValueError("A2A history valid mask must contain only 0/1 values")
            supplied_valid_time = supplied_valid_time.astype(np.float32)
        else:
            supplied_valid_time = np.ones(values.shape[0], dtype=np.float32)
        if self.a2a_cold_start == "require_full_history" and (
            values.shape[0] < horizon or not np.all(supplied_valid_time[-horizon:])
        ):
            raise ValueError(
                f"A2A requires {horizon} real history steps; padded/invalid steps were supplied"
            )
        if values.shape[0] >= horizon:
            return values[-horizon:], supplied_valid_time[-horizon:]
        if self.a2a_cold_start == "require_full_history":
            raise ValueError(f"A2A requires {horizon} history steps, got {values.shape[0]}")
        pad_count = horizon - values.shape[0]
        padded = np.concatenate([np.repeat(values[:1], pad_count, axis=0), values], axis=0)
        valid = np.concatenate([np.zeros(pad_count, dtype=np.float32), supplied_valid_time])
        return padded, valid

    def _build_a2a_inputs(self, content) -> dict[str, torch.Tensor]:
        embodiment = content.embodiment.value
        specs = self._require_parameters(embodiment)
        history = np.zeros(
            (self.a2a_history_horizon, self.max_action_dim), dtype=np.float32
        )
        future = np.zeros((self.a2a_future_horizon, self.max_action_dim), dtype=np.float32)
        history_mask = np.zeros_like(history)
        future_mask = np.zeros_like(future)
        continuous_mask = np.zeros_like(future)
        auxiliary_mask = np.zeros_like(future)
        binary_mask = np.zeros_like(future)
        categorical_mask = np.zeros_like(future)
        categorical_group_index = np.full(future.shape, -1, dtype=np.int64)

        supplied_history_mask = content.metadata.get("a2a_history_valid_mask")
        if supplied_history_mask is not None:
            supplied_history_mask = _as_numpy(supplied_history_mask)
        categorical_group = 0

        cursor = 0
        for spec in specs:
            dim = int(spec.dim)
            end = cursor + dim
            if end > self.max_action_dim:
                raise ValueError(
                    f"Canonical action dimension exceeds max_action_dim={self.max_action_dim}"
                )
            if spec.kind == "continuous":
                source, valid_time = self._history_window(
                    _as_numpy(content.states[spec.source_state_key]), supplied_history_mask
                )
                history[:, cursor:end] = self._normalize(
                    source, embodiment, spec.action_key
                )
                history_mask[:, cursor:end] = valid_time[:, None]
                continuous_mask[:, cursor:end] = 1.0

            if content.actions and spec.action_key in content.actions:
                target = _as_numpy(content.actions[spec.action_key]).astype(np.float32)
                if target.ndim != 2 or target.shape[1] != dim:
                    raise ValueError(
                        f"A2A target {embodiment}/{spec.action_key} must have shape "
                        f"[T,{dim}], got {target.shape}"
                    )
                if target.shape[0] != self.a2a_future_horizon:
                    raise ValueError(
                        f"Strict A2A target {embodiment}/{spec.action_key} must contain "
                        f"exactly {self.a2a_future_horizon} steps, got {target.shape[0]}"
                    )
                if not np.all(np.isfinite(target)):
                    raise ValueError(
                        f"A2A target {embodiment}/{spec.action_key} contains NaN or infinity"
                    )
                valid_steps = self.a2a_future_horizon
                if spec.kind == "binary":
                    lower = self.a2a_norm_params[embodiment][spec.action_key]["min"]
                    upper = self.a2a_norm_params[embodiment][spec.action_key]["max"]
                    threshold = (lower + upper) / 2.0
                    future[:valid_steps, cursor:end] = (target > threshold).astype(np.float32)
                elif spec.kind == "categorical":
                    if not np.all((target == 0) | (target == 1)) or not np.allclose(
                        target.sum(axis=-1), 1.0
                    ):
                        raise ValueError(
                            f"Categorical target {embodiment}/{spec.action_key} must be one-hot"
                        )
                    future[:valid_steps, cursor:end] = target
                else:
                    future[:valid_steps, cursor:end] = self._normalize(
                        target, embodiment, spec.action_key
                    )
                future_mask[:valid_steps, cursor:end] = 1.0
            else:
                # Inference still needs output masks even without a target.
                future_mask[:, cursor:end] = 1.0

            if spec.kind == "regression":
                auxiliary_mask[:, cursor:end] = 1.0
            elif spec.kind == "binary":
                binary_mask[:, cursor:end] = 1.0
            elif spec.kind == "categorical":
                categorical_mask[:, cursor:end] = 1.0
                categorical_group_index[:, cursor:end] = categorical_group
                categorical_group += 1
            cursor = end

        # Only masks for valid future steps participate during training.
        continuous_mask *= future_mask
        auxiliary_mask *= future_mask
        binary_mask *= future_mask
        categorical_mask *= future_mask
        return {
            "history_action_canonical": torch.from_numpy(history),
            "future_action_canonical": torch.from_numpy(future),
            "history_action_mask": torch.from_numpy(history_mask),
            "future_action_mask": torch.from_numpy(future_mask),
            "continuous_action_mask": torch.from_numpy(continuous_mask),
            "auxiliary_action_mask": torch.from_numpy(auxiliary_mask),
            "binary_action_mask": torch.from_numpy(binary_mask),
            "categorical_action_mask": torch.from_numpy(categorical_mask),
            "categorical_group_index": torch.from_numpy(categorical_group_index),
        }

    def __call__(self, messages: list[dict[str, Any]]):
        content = messages[0]["content"]
        transformed = super().__call__(messages)
        # A2A consumes the full proprio history separately; keep only the current
        # state in the legacy field for optional state-conditioning ablations.
        transformed["state"] = transformed["state"][-1:]
        transformed.pop("action", None)
        transformed.pop("action_mask", None)
        transformed.update(self._build_a2a_inputs(content))
        return transformed

    def decode_action(self, action: np.ndarray, embodiment_tag, state=None):
        del state  # Canonical outputs are already absolute physical targets.
        embodiment = embodiment_tag.value
        specs = self._require_parameters(embodiment)
        action = _as_numpy(action).astype(np.float32)
        horizon = self.a2a_future_horizon
        output: dict[str, np.ndarray] = {}
        cursor = 0
        for spec in specs:
            end = cursor + int(spec.dim)
            values = action[..., :horizon, cursor:end]
            if spec.kind == "binary":
                output[spec.action_key] = (values >= 0.5).astype(np.float32)
            elif spec.kind == "categorical":
                index = values.argmax(axis=-1)
                output[spec.action_key] = np.eye(int(spec.num_classes), dtype=np.float32)[index]
            else:
                output[spec.action_key] = unnormalize_values_minmax(
                    values, self.a2a_norm_params[embodiment][spec.action_key]
                ).astype(np.float32)
            cursor = end
        return output

    def save_pretrained(self, save_directory: str | Path) -> list[Path]:
        files = super().save_pretrained(save_directory)
        save_directory = Path(save_directory)
        config_path = save_directory / "processor_config.json"
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        config["processor_class"] = self.__class__.__name__
        kwargs = config["processor_kwargs"]
        kwargs.update(
            {
                "a2a_history_horizon": self.a2a_history_horizon,
                "a2a_future_horizon": self.a2a_future_horizon,
                "a2a_channel_specs": {
                    embodiment: [asdict(spec) for spec in specs]
                    for embodiment, specs in self._resolved_channel_specs.items()
                },
                "a2a_cold_start": self.a2a_cold_start,
                "a2a_canonical_statistics": self.a2a_canonical_statistics,
                "a2a_require_canonical_statistics": self.a2a_require_canonical_statistics,
                "a2a_require_explicit_channel_specs": (
                    self.a2a_require_explicit_channel_specs
                ),
                "a2a_require_semantic_metadata": self.a2a_require_semantic_metadata,
                "a2a_expected_contract_sha256": self.a2a_expected_contract_sha256,
            }
        )
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
        a2a_stats_path = save_directory / "a2a_statistics.json"
        with a2a_stats_path.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "canonical_statistics": self.a2a_canonical_statistics,
                    "normalization_by_embodiment": self.a2a_statistics,
                },
                handle,
                indent=2,
            )
        return [*files, a2a_stats_path]


AutoProcessor.register("Gr00tN1d7A2A", Gr00tN1d7A2AProcessor)
