#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Export strict, unnormalized canonical A2A windows from preprocessed data.

This exporter deliberately implements only the identity transform.  Every
continuous source-state and target-action column must therefore already be in
the same physical representation, unit, and frame recorded by the channel
contract.  Controller deltas, joint velocities, and other raw command spaces
must be canonicalized into new dataset columns before this script is used.

The output NPZ is the input expected by ``build_canonical_stats.py`` and
``audit_a2a_data.py``.  No model Processor is constructed or called, so the
arrays remain in unnormalized physical units.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import importlib
import json
from pathlib import Path
import sys
from typing import Any

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.a2a_contract import validate_channel_contract
from gr00t.data.dataset.a2a_single_step_dataset import (
    A2AShardedSingleStepDataset,
    compute_a2a_window_bounds,
    extract_a2a_step_data,
)
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VLAStepData
import numpy as np


IDENTITY_CANONICALIZER = "identity_preprocessed"
EXPORTER_VERSION = "1"
_SUPPORTED_KINDS = {"continuous", "regression", "binary", "categorical"}
_RAW_LIBERO_ACTION_KEYS = {"x", "y", "z", "roll", "pitch", "yaw", "gripper"}


def load_modality_config(modality_config_path: str | Path) -> Path:
    """Import a user-selected local Python module that registers modality configs.

    This mirrors the explicit local registration path used by
    ``gr00t.experiment.launch_finetune``.  Importing a Python file executes it,
    so the path is accepted only through the dedicated CLI option.
    """

    path = Path(modality_config_path)
    if not path.exists() or not path.is_file() or path.suffix != ".py":
        raise FileNotFoundError(
            f"Modality config path does not exist or is not a .py file: {path}"
        )
    sys.path.append(str(path.parent))
    importlib.import_module(path.stem)
    print(f"Loaded modality config: {path}")
    return path


def _required_positive_int(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
    ):
        raise ValueError(f"A2A contract {key} must be a positive integer, got {value!r}")
    return int(value)


def _require_nonempty_string(payload: dict[str, Any], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{context} requires a non-empty {key}")
    return value


def validate_identity_preprocessed_contract(contract: dict[str, Any]) -> dict[str, Any]:
    """Validate that a contract is safe for a no-op canonicalizer.

    String equality is not treated as a conversion.  Continuous columns must
    already have identical source, target, and canonical semantics.  LIBERO's
    well-known raw controller keys are rejected as continuous targets because
    they are OSC delta commands rather than absolute EEF state.
    """

    feature_dim = _required_positive_int(contract, "feature_dim")
    history_horizon = _required_positive_int(contract, "history_horizon")
    future_horizon = _required_positive_int(contract, "future_horizon")
    normalized = validate_channel_contract(
        contract,
        feature_dim=feature_dim,
        history_horizon=history_horizon,
        future_horizon=future_horizon,
    )

    provenance = normalized["provenance"]
    if provenance["canonicalizer"] != IDENTITY_CANONICALIZER:
        raise ValueError(
            "export_canonical_windows.py only supports "
            f"provenance.canonicalizer={IDENTITY_CANONICALIZER!r}, got "
            f"{provenance['canonicalizer']!r}"
        )
    if provenance["exporter_version"] != EXPORTER_VERSION:
        raise ValueError(
            f"Contract exporter_version={provenance['exporter_version']!r} does not match "
            f"this exporter version {EXPORTER_VERSION!r}"
        )

    controller = normalized["controller"]
    for key in ("type", "version", "frame", "rotation_composition"):
        _require_nonempty_string(controller, key, "A2A controller contract")
    if not isinstance(controller["control_delta"], bool):
        raise ValueError("A2A controller control_delta must be a JSON boolean")
    scale = np.asarray(controller["scale"])
    if (
        scale.ndim != 1
        or scale.size == 0
        or not np.issubdtype(scale.dtype, np.number)
        or not np.isfinite(scale).all()
    ):
        raise ValueError("A2A controller scale must be a non-empty finite numeric vector")

    continuous_count = 0
    continuous_sources: set[str] = set()
    for channel in normalized["channels"]:
        action_key = channel["action_key"]
        kind = channel.get("kind")
        if kind not in _SUPPORTED_KINDS:
            raise ValueError(
                f"A2A channel {action_key!r} has unsupported kind {kind!r}; "
                f"expected one of {sorted(_SUPPORTED_KINDS)}"
            )
        if kind != "continuous":
            continue

        continuous_count += 1
        source_key = _require_nonempty_string(
            channel, "source_state_key", f"Continuous A2A channel {action_key!r}"
        )
        if source_key in continuous_sources:
            raise ValueError(
                f"Continuous source_state_key {source_key!r} is mapped more than once"
            )
        continuous_sources.add(source_key)

        canonical_format = _require_nonempty_string(
            channel, "canonical_format", f"Continuous A2A channel {action_key!r}"
        )
        source_format = _require_nonempty_string(
            channel, "source_format", f"Continuous A2A channel {action_key!r}"
        )
        target_format = _require_nonempty_string(
            channel, "target_format", f"Continuous A2A channel {action_key!r}"
        )
        if canonical_format == "auto" or not (
            canonical_format == source_format == target_format
        ):
            raise ValueError(
                f"Identity continuous channel {action_key!r} must already share one explicit "
                "source/target/canonical format"
            )
        source_unit = _require_nonempty_string(
            channel, "source_unit", f"Continuous A2A channel {action_key!r}"
        )
        target_unit = _require_nonempty_string(
            channel, "target_unit", f"Continuous A2A channel {action_key!r}"
        )
        source_frame = _require_nonempty_string(
            channel, "source_frame", f"Continuous A2A channel {action_key!r}"
        )
        target_frame = _require_nonempty_string(
            channel, "target_frame", f"Continuous A2A channel {action_key!r}"
        )
        if source_unit != target_unit or source_frame != target_frame:
            raise ValueError(
                f"Identity continuous channel {action_key!r} has mismatched units or frames"
            )

        if normalized["embodiment"] == "libero_sim" and action_key in _RAW_LIBERO_ACTION_KEYS:
            raise ValueError(
                "Raw LIBERO action keys are OSC delta commands and cannot be exported as "
                "identity continuous canonical targets. Preprocess them into new absolute "
                "canonical action columns first."
            )

    if continuous_count == 0:
        raise ValueError("Identity latent A2A export requires at least one continuous channel")
    return normalized


def build_export_modality_configs(
    contract: dict[str, Any],
    base_modality_configs: dict[str, ModalityConfig],
) -> dict[str, ModalityConfig]:
    """Build the smallest raw-data configuration needed by the exporter."""

    contract = validate_identity_preprocessed_contract(contract)
    if "language" not in base_modality_configs:
        raise ValueError("A2A export requires one language modality for VLAStepData extraction")
    state_keys = [
        channel["source_state_key"]
        for channel in contract["channels"]
        if channel["kind"] == "continuous"
    ]
    action_keys = [channel["action_key"] for channel in contract["channels"]]
    return {
        "state": ModalityConfig(
            delta_indices=list(range(-(int(contract["history_horizon"]) - 1), 1)),
            modality_keys=state_keys,
        ),
        "action": ModalityConfig(
            delta_indices=list(range(int(contract["future_horizon"]))),
            modality_keys=action_keys,
        ),
        "language": deepcopy(base_modality_configs["language"]),
    }


def _finite_matrix(value: Any, expected_shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must be real numeric, got dtype {array.dtype}")
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {array.shape}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    return array.astype(np.float32, copy=False)


def pack_identity_preprocessed_window(
    step: VLAStepData, contract: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Pack one raw, already-canonical VLA step into the shared A2A layout."""

    contract = validate_identity_preprocessed_contract(contract)
    return _pack_validated_identity_preprocessed_window(step, contract)


def _pack_validated_identity_preprocessed_window(
    step: VLAStepData, contract: dict[str, Any]
) -> dict[str, np.ndarray]:
    """Pack one step after the caller has validated ``contract`` once."""

    history_horizon = int(contract["history_horizon"])
    future_horizon = int(contract["future_horizon"])
    feature_dim = int(contract["feature_dim"])
    history = np.zeros((history_horizon, feature_dim), dtype=np.float32)
    future = np.zeros((future_horizon, feature_dim), dtype=np.float32)
    history_mask = np.zeros_like(history, dtype=np.uint8)
    future_mask = np.zeros_like(future, dtype=np.uint8)

    for channel in contract["channels"]:
        action_key = channel["action_key"]
        kind = channel["kind"]
        start, end = int(channel["start"]), int(channel["end"])
        dim = end - start
        if action_key not in step.actions:
            raise KeyError(f"Canonical target action {action_key!r} is missing from the sample")
        target = _finite_matrix(
            step.actions[action_key],
            (future_horizon, dim),
            f"action.{action_key}",
        )
        if kind == "binary" and not np.isin(target, [0.0, 1.0]).all():
            raise ValueError(
                f"Binary canonical target action.{action_key} must contain only 0/1"
            )
        if kind == "categorical" and (
            not np.isin(target, [0.0, 1.0]).all()
            or not np.allclose(target.sum(axis=-1), 1.0)
        ):
            raise ValueError(
                f"Categorical canonical target action.{action_key} must be one-hot"
            )
        future[:, start:end] = target
        future_mask[:, start:end] = 1

        if kind == "continuous":
            source_key = channel["source_state_key"]
            if source_key not in step.states:
                raise KeyError(
                    f"Canonical history state {source_key!r} for action {action_key!r} is missing"
                )
            source = _finite_matrix(
                step.states[source_key],
                (history_horizon, dim),
                f"state.{source_key}",
            )
            history[:, start:end] = source
            history_mask[:, start:end] = 1

    if not np.isin(history_mask, [0, 1]).all() or not np.isin(future_mask, [0, 1]).all():
        raise AssertionError("Internal A2A exporter masks are not binary")
    if not history_mask.any() or not future_mask.any():
        raise ValueError("Canonical window has no active history/future values")
    if not np.isfinite(history[history_mask.astype(bool)]).all() or not np.isfinite(
        future[future_mask.astype(bool)]
    ).all():
        raise ValueError("Canonical window contains non-finite active values")
    return {
        "history": history,
        "future": future,
        "history_mask": history_mask,
        "future_mask": future_mask,
    }


def _validate_export_dataset_contract(
    dataset: A2AShardedSingleStepDataset,
    contract: dict[str, Any],
) -> None:
    """Require the loader configuration to encode the contract's exact timeline."""

    dataset_embodiment = getattr(dataset.embodiment_tag, "value", dataset.embodiment_tag)
    if dataset_embodiment != contract["embodiment"]:
        raise ValueError(
            f"Dataset embodiment {dataset_embodiment!r} does not match contract embodiment "
            f"{contract['embodiment']!r}"
        )

    expected_state_deltas = list(range(-(int(contract["history_horizon"]) - 1), 1))
    expected_action_deltas = list(range(int(contract["future_horizon"])))
    expected_state_keys = [
        channel["source_state_key"]
        for channel in contract["channels"]
        if channel["kind"] == "continuous"
    ]
    expected_action_keys = [channel["action_key"] for channel in contract["channels"]]
    expected = {
        "state": (expected_state_deltas, expected_state_keys),
        "action": (expected_action_deltas, expected_action_keys),
    }
    for modality, (expected_deltas, expected_keys) in expected.items():
        if modality not in dataset.modality_configs:
            raise ValueError(f"A2A export dataset is missing {modality!r} modality config")
        config = dataset.modality_configs[modality]
        actual_deltas = list(config.delta_indices)
        actual_keys = list(config.modality_keys)
        if actual_deltas != expected_deltas:
            raise ValueError(
                f"A2A export {modality} delta_indices={actual_deltas}, expected the exact "
                f"contract timeline {expected_deltas}"
            )
        if actual_keys != expected_keys:
            raise ValueError(
                f"A2A export {modality} modality_keys={actual_keys}, expected {expected_keys}"
            )


def collect_identity_preprocessed_windows(
    dataset: A2AShardedSingleStepDataset,
    contract: dict[str, Any],
    *,
    max_windows: int | None = None,
) -> dict[str, np.ndarray]:
    """Traverse every strict anchor in every complete episode."""

    contract = validate_identity_preprocessed_contract(contract)
    _validate_export_dataset_contract(dataset, contract)
    if max_windows is not None and (
        isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows <= 0
    ):
        raise ValueError(f"max_windows must be a positive integer, got {max_windows!r}")

    windows: list[dict[str, np.ndarray]] = []
    episode_indices: list[int] = []
    base_step_indices: list[int] = []
    stop = False
    for episode_index in range(len(dataset.episode_loader.episode_lengths)):
        # LeRobotEpisodeLoader.__getitem__ returns the complete processed episode.
        episode_data = dataset.episode_loader[episode_index]
        bounds = compute_a2a_window_bounds(len(episode_data), dataset.modality_configs)
        for step_index in range(bounds.valid_start, bounds.valid_end + 1):
            step = extract_a2a_step_data(
                episode_data=episode_data,
                step_index=step_index,
                modality_configs=dataset.modality_configs,
                embodiment_tag=dataset.embodiment_tag,
            )
            windows.append(_pack_validated_identity_preprocessed_window(step, contract))
            episode_indices.append(episode_index)
            base_step_indices.append(step_index)
            if max_windows is not None and len(windows) >= max_windows:
                stop = True
                break
        if stop:
            break

    if not windows:
        raise ValueError("Dataset contains no strict canonical A2A windows")
    result = {
        key: np.stack([window[key] for window in windows], axis=0)
        for key in ("history", "future", "history_mask", "future_mask")
    }
    result["episode_index"] = np.asarray(episode_indices, dtype=np.int64)
    result["base_step_index"] = np.asarray(base_step_indices, dtype=np.int64)
    return result


def _validate_export_arrays(
    arrays: dict[str, np.ndarray],
    contract: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Validate the complete NPZ tensor contract before any bytes are written."""

    required = {
        "history",
        "future",
        "history_mask",
        "future_mask",
        "episode_index",
        "base_step_index",
    }
    missing = required - set(arrays)
    if missing:
        raise KeyError(f"Canonical export arrays are missing: {sorted(missing)}")

    history = np.asarray(arrays["history"])
    future = np.asarray(arrays["future"])
    history_mask = np.asarray(arrays["history_mask"])
    future_mask = np.asarray(arrays["future_mask"])
    episode_index = np.asarray(arrays["episode_index"])
    base_step_index = np.asarray(arrays["base_step_index"])

    history_shape = (
        history.shape[0] if history.ndim == 3 else -1,
        int(contract["history_horizon"]),
        int(contract["feature_dim"]),
    )
    future_shape = (
        history_shape[0],
        int(contract["future_horizon"]),
        int(contract["feature_dim"]),
    )
    if history.shape != history_shape or history_shape[0] <= 0:
        raise ValueError(
            f"history must have non-empty shape [N,H,D]={history_shape}, "
            f"got {history.shape}"
        )
    if future.shape != future_shape:
        raise ValueError(f"future must have shape {future_shape}, got {future.shape}")
    if history_mask.shape != history_shape or future_mask.shape != future_shape:
        raise ValueError("Canonical masks must exactly match their history/future array shapes")
    for value, name in ((history, "history"), (future, "future")):
        if not np.issubdtype(value.dtype, np.number) or np.issubdtype(
            value.dtype, np.complexfloating
        ):
            raise TypeError(f"{name} must be real numeric, got dtype {value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"{name} must contain only finite values, including padding")
    for value, name in ((history_mask, "history_mask"), (future_mask, "future_mask")):
        if not (
            np.issubdtype(value.dtype, np.bool_)
            or (
                np.issubdtype(value.dtype, np.number)
                and not np.issubdtype(value.dtype, np.complexfloating)
            )
        ):
            raise TypeError(f"{name} must be boolean or real numeric, got dtype {value.dtype}")
        if not np.isfinite(value).all() or not np.isin(value, [0, 1]).all():
            raise ValueError(f"{name} must contain only finite binary 0/1 values")

    expected_history_mask = np.zeros(history_shape[1:], dtype=np.uint8)
    expected_future_mask = np.zeros(future_shape[1:], dtype=np.uint8)
    for channel in contract["channels"]:
        start, end = int(channel["start"]), int(channel["end"])
        expected_future_mask[:, start:end] = 1
        if channel["kind"] == "continuous":
            expected_history_mask[:, start:end] = 1
    if not np.array_equal(
        history_mask.astype(np.uint8, copy=False),
        np.broadcast_to(expected_history_mask, history_shape),
    ):
        raise ValueError("history_mask does not match the contract's continuous channel layout")
    if not np.array_equal(
        future_mask.astype(np.uint8, copy=False),
        np.broadcast_to(expected_future_mask, future_shape),
    ):
        raise ValueError("future_mask does not match the contract's target channel layout")
    for channel in contract["channels"]:
        start, end = int(channel["start"]), int(channel["end"])
        target = future[..., start:end]
        if channel["kind"] == "binary" and not np.isin(target, [0.0, 1.0]).all():
            raise ValueError(
                f"Binary canonical target {channel['action_key']!r} must contain only 0/1"
            )
        if channel["kind"] == "categorical" and (
            not np.isin(target, [0.0, 1.0]).all()
            or not np.allclose(target.sum(axis=-1), 1.0)
        ):
            raise ValueError(
                f"Categorical canonical target {channel['action_key']!r} must be one-hot"
            )
    if np.any(history[~history_mask.astype(bool)]) or np.any(
        future[~future_mask.astype(bool)]
    ):
        raise ValueError("Inactive canonical padding must be exactly zero")

    for value, name in (
        (episode_index, "episode_index"),
        (base_step_index, "base_step_index"),
    ):
        if value.shape != (history_shape[0],):
            raise ValueError(f"{name} must have shape {(history_shape[0],)}, got {value.shape}")
        if not np.issubdtype(value.dtype, np.integer) or np.any(value < 0):
            raise ValueError(f"{name} must contain non-negative integers")
    anchors = np.stack((episode_index, base_step_index), axis=-1)
    if np.unique(anchors, axis=0).shape[0] != history_shape[0]:
        raise ValueError("Canonical export contains duplicate episode/base-step anchors")
    expected_order = np.lexsort((base_step_index, episode_index))
    if not np.array_equal(expected_order, np.arange(history_shape[0])):
        raise ValueError("Canonical export anchors must be ordered by episode then base step")

    return {
        "history": history,
        "future": future,
        "history_mask": history_mask.astype(np.uint8, copy=False),
        "future_mask": future_mask.astype(np.uint8, copy=False),
        "episode_index": episode_index.astype(np.int64, copy=False),
        "base_step_index": base_step_index.astype(np.int64, copy=False),
    }


def save_identity_preprocessed_windows(
    arrays: dict[str, np.ndarray],
    contract: dict[str, Any],
    output_path: str | Path,
) -> Path:
    """Save canonical arrays together with their immutable contract provenance."""

    contract = validate_identity_preprocessed_contract(contract)
    arrays = _validate_export_arrays(arrays, contract)
    output_path = Path(output_path)
    if output_path.suffix.lower() != ".npz":
        raise ValueError(f"Canonical export output must use a .npz suffix: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    provenance = contract["provenance"]
    metadata = {
        "input_space": np.asarray("unnormalized_canonical_physical"),
        "contract_json": np.asarray(
            json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        ),
        "contract_sha256": np.asarray(contract["sha256"]),
        "contract_version": np.asarray(contract["version"], dtype=np.int64),
        "embodiment": np.asarray(contract["embodiment"]),
        "dataset": np.asarray(provenance["dataset"]),
        "dataset_revision": np.asarray(provenance["dataset_revision"]),
        "dataset_fingerprint_sha256": np.asarray(
            provenance["dataset_fingerprint_sha256"]
        ),
        "source_schema_sha256": np.asarray(provenance["source_schema_sha256"]),
        "canonicalizer": np.asarray(provenance["canonicalizer"]),
        "canonicalizer_version": np.asarray(provenance["canonicalizer_version"]),
        "canonicalizer_sha256": np.asarray(provenance["canonicalizer_sha256"]),
        "exporter_version": np.asarray(provenance["exporter_version"]),
        "target_definition": np.asarray(provenance["target_definition"]),
        "time_alignment": np.asarray(provenance["time_alignment"]),
        "num_windows": np.asarray(arrays["history"].shape[0], dtype=np.int64),
        "history_horizon": np.asarray(contract["history_horizon"], dtype=np.int64),
        "future_horizon": np.asarray(contract["future_horizon"], dtype=np.int64),
        "feature_dim": np.asarray(contract["feature_dim"], dtype=np.int64),
    }
    np.savez_compressed(output_path, **arrays, **metadata)
    return output_path


def export_identity_preprocessed_dataset(
    dataset: A2AShardedSingleStepDataset,
    contract: dict[str, Any],
    output_path: str | Path,
    *,
    max_windows: int | None = None,
) -> Path:
    """Collect and save all strict identity-preprocessed windows."""

    arrays = collect_identity_preprocessed_windows(
        dataset, contract, max_windows=max_windows
    )
    return save_identity_preprocessed_windows(arrays, contract, output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--embodiment-tag",
        help="Embodiment enum name or value; defaults to contract.embodiment",
    )
    parser.add_argument(
        "--modality-config-path",
        type=Path,
        help="Local .py module that registers a custom entry in MODALITY_CONFIGS",
    )
    parser.add_argument("--max-windows", type=int)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    contract = validate_identity_preprocessed_contract(
        json.loads(args.contract.read_text(encoding="utf-8"))
    )
    embodiment = EmbodimentTag.resolve(args.embodiment_tag or contract["embodiment"])
    if embodiment.value != contract["embodiment"]:
        raise ValueError(
            f"Contract embodiment {contract['embodiment']!r} does not match "
            f"--embodiment-tag {embodiment.value!r}"
        )
    if args.modality_config_path is not None:
        load_modality_config(args.modality_config_path)
    if embodiment.value not in MODALITY_CONFIGS:
        raise ValueError(
            f"No registered modality configuration for embodiment {embodiment.value!r}. "
            "Pass --modality-config-path for preprocessed custom embodiments."
        )
    modality_configs = build_export_modality_configs(
        contract, MODALITY_CONFIGS[embodiment.value]
    )
    dataset = A2AShardedSingleStepDataset(
        dataset_path=args.dataset_path,
        embodiment_tag=embodiment,
        modality_configs=modality_configs,
        episode_sampling_rate=1.0,
        seed=args.seed,
        allow_padding=False,
    )
    output_path = export_identity_preprocessed_dataset(
        dataset,
        contract,
        args.output,
        max_windows=args.max_windows,
    )
    print(f"Wrote strict unnormalized canonical A2A windows to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
