#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Audit canonical A2A windows before training.

The CLI consumes the same NPZ contract as ``build_canonical_stats.py``.  It
checks masks and finite values, reports constant channels and temporal jumps,
and compares history-to-future distance with a standard-Gaussian source
baseline in the shared normalized space.  Optional channel-spec JSON auditing
verifies that every continuous action channel has an executed proprio source.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


try:
    from .build_canonical_stats import (
        _trajectory_horizon_and_dim,
        compute_shared_canonical_statistics,
        load_a2a_npz,
        load_a2a_npz_export_metadata,
        prepare_canonical_inputs,
    )
except ImportError:  # Direct execution: python scripts/a2a/audit_a2a_data.py
    from build_canonical_stats import (  # type: ignore[no-redef]
        _trajectory_horizon_and_dim,
        compute_shared_canonical_statistics,
        load_a2a_npz,
        load_a2a_npz_export_metadata,
        prepare_canonical_inputs,
    )
from gr00t.data.a2a_contract import validate_channel_contract


def distribution_summary(values: np.ndarray) -> dict[str, float | int | None]:
    """Return compact finite-only distribution statistics."""

    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0, "mean": None, "p50": None, "p90": None, "p95": None}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
    }


def masked_sample_rms(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Compute one RMS value per batch element under ``mask``."""

    values = np.asarray(values, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool) & np.isfinite(values)
    reduce_axes = tuple(range(1, values.ndim))
    counts = mask.sum(axis=reduce_axes)
    squared_sum = np.where(mask, np.square(values), 0.0).sum(axis=reduce_axes)
    result = np.full(values.shape[0], np.nan, dtype=np.float64)
    np.divide(squared_sum, counts, out=result, where=counts > 0)
    return np.sqrt(result)


def audit_channel_spec(
    channels: list[dict[str, Any]],
    available_state_keys: set[str] | list[str],
    available_action_keys: set[str] | list[str],
) -> dict[str, Any]:
    """Check explicit action-to-executed-state channel mappings."""

    state_keys = set(available_state_keys)
    action_keys = set(available_action_keys)
    missing_action_keys: list[str] = []
    missing_source_state_keys: list[str] = []
    unsupported_channels: list[str] = []
    discrete_channels: list[str] = []
    duplicate_action_keys: list[str] = []
    dimension_mismatches: list[dict[str, Any]] = []
    semantic_mismatches: list[dict[str, Any]] = []
    seen_action_keys: set[str] = set()

    for index, channel in enumerate(channels):
        name = str(channel.get("action_key", f"channel[{index}]"))
        kind = channel.get("kind", "unsupported")
        if name in seen_action_keys:
            duplicate_action_keys.append(name)
        seen_action_keys.add(name)
        if name not in action_keys:
            missing_action_keys.append(name)
        if kind == "continuous":
            source_key = channel.get("source_state_key")
            if not source_key or source_key not in state_keys:
                missing_source_state_keys.append(name)

            source_dim = channel.get("source_dim")
            target_dim = channel.get("target_dim", channel.get("dim"))
            if source_dim is not None and target_dim is not None and source_dim != target_dim:
                dimension_mismatches.append(
                    {"action_key": name, "source_dim": source_dim, "target_dim": target_dim}
                )

            for semantic in ("format", "unit", "frame"):
                source_value = channel.get(f"source_{semantic}")
                target_value = channel.get(f"target_{semantic}")
                if (
                    source_value is not None
                    and target_value is not None
                    and source_value != target_value
                ):
                    semantic_mismatches.append(
                        {
                            "action_key": name,
                            "field": semantic,
                            "source": source_value,
                            "target": target_value,
                        }
                    )
        elif kind in {"binary", "categorical"}:
            discrete_channels.append(name)
        else:
            unsupported_channels.append(name)

    errors = []
    if missing_action_keys:
        errors.append("Channel spec references action keys absent from the dataset")
    if missing_source_state_keys:
        errors.append("Continuous channels are missing an executed proprio source")
    if duplicate_action_keys:
        errors.append("Channel spec contains duplicate action keys")
    if dimension_mismatches:
        errors.append("Continuous source and target dimensions do not match")
    if semantic_mismatches:
        errors.append("Continuous source and target physical semantics do not match")
    return {
        "num_channels": len(channels),
        "missing_action_keys": sorted(set(missing_action_keys)),
        "missing_source_state_keys": sorted(set(missing_source_state_keys)),
        "discrete_channels": sorted(set(discrete_channels)),
        "unsupported_channels": sorted(set(unsupported_channels)),
        "duplicate_action_keys": sorted(set(duplicate_action_keys)),
        "dimension_mismatches": dimension_mismatches,
        "semantic_mismatches": semantic_mismatches,
        "errors": errors,
    }


def _temporal_jump_report(
    trajectory: np.ndarray,
    mask: np.ndarray,
    std: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    if trajectory.shape[1] < 2:
        return {"valid_transitions": 0, "jump_count": 0, "jump_ratio": 0.0}
    transition_mask = mask[:, 1:, :] & mask[:, :-1, :]
    finite_pair = np.isfinite(trajectory[:, 1:, :]) & np.isfinite(trajectory[:, :-1, :])
    transition_mask &= finite_pair
    normalized_delta = np.abs(np.diff(trajectory, axis=1)) / std[None, None, :]
    valid_transitions = int(transition_mask.sum())
    jump_count = int((transition_mask & (normalized_delta > threshold)).sum())
    return {
        "threshold_in_shared_std": threshold,
        "valid_transitions": valid_transitions,
        "jump_count": jump_count,
        "jump_ratio": jump_count / valid_transitions if valid_transitions else 0.0,
    }


def audit_a2a_arrays(
    history: np.ndarray,
    future: np.ndarray,
    history_mask: np.ndarray | None = None,
    future_mask: np.ndarray | None = None,
    *,
    jump_threshold: float = 5.0,
    random_seed: int = 42,
) -> dict[str, Any]:
    """Audit A2A tensors and return a JSON-serializable report."""

    if jump_threshold <= 0:
        raise ValueError(f"jump_threshold must be positive, got {jump_threshold}")
    history, future, history_mask, future_mask, shared_channel_mask = prepare_canonical_inputs(
        history, future, history_mask, future_mask
    )
    statistics = compute_shared_canonical_statistics(
        history,
        future,
        history_mask,
        future_mask,
        allow_nonfinite=True,
    )
    mean = np.asarray(statistics["mean"], dtype=np.float64)
    std = np.asarray(statistics["std"], dtype=np.float64)
    history_normalized = (history - mean[None, None, :]) / std[None, None, :]
    future_normalized = (future - mean[None, None, :]) / std[None, None, :]

    history_nonfinite = int((history_mask & ~np.isfinite(history)).sum())
    future_nonfinite = int((future_mask & ~np.isfinite(future)).sum())
    active_channels = np.flatnonzero(shared_channel_mask).tolist()
    constant_channels = np.flatnonzero(
        np.asarray(statistics["constant_mask"], dtype=bool) & shared_channel_mask
    ).tolist()

    rng = np.random.default_rng(random_seed)
    gaussian = rng.standard_normal(size=future_normalized.shape)
    gaussian_distance = masked_sample_rms(gaussian - future_normalized, future_mask)

    trajectory_distance: np.ndarray | None = None
    if history.shape[1] == future.shape[1]:
        paired_mask = history_mask & future_mask
        trajectory_distance = masked_sample_rms(history_normalized - future_normalized, paired_mask)

    boundary_mask = history_mask[:, -1, :] & future_mask[:, 0, :]
    boundary_distance = masked_sample_rms(
        history_normalized[:, -1, :] - future_normalized[:, 0, :], boundary_mask
    )
    gaussian_summary = distribution_summary(gaussian_distance)
    trajectory_summary = (
        distribution_summary(trajectory_distance) if trajectory_distance is not None else None
    )

    errors: list[str] = []
    warnings: list[str] = []
    if history_nonfinite or future_nonfinite:
        errors.append("Non-finite values occur under active canonical masks")
    if not active_channels:
        errors.append("History and future have no shared continuous canonical channels")
    if constant_channels:
        warnings.append("Some active canonical channels are constant or near-constant")

    distance_ratio = None
    distance_advantage = None
    if trajectory_summary is not None:
        history_mean = trajectory_summary["mean"]
        gaussian_mean = gaussian_summary["mean"]
        if history_mean is not None and gaussian_mean not in {None, 0.0}:
            distance_ratio = float(history_mean / gaussian_mean)
            distance_advantage = bool(history_mean < gaussian_mean)
            if not distance_advantage:
                warnings.append(
                    "History is not closer to future than a Gaussian source in shared normalized space"
                )

    return {
        "version": 1,
        "num_windows": history.shape[0],
        "history_shape": list(history.shape),
        "future_shape": list(future.shape),
        "active_channel_indices": active_channels,
        "constant_channel_indices": constant_channels,
        "history_nonfinite_active_count": history_nonfinite,
        "future_nonfinite_active_count": future_nonfinite,
        "history_temporal_jumps": _temporal_jump_report(history, history_mask, std, jump_threshold),
        "future_temporal_jumps": _temporal_jump_report(future, future_mask, std, jump_threshold),
        "history_future_trajectory_rms": trajectory_summary,
        "history_last_future_first_rms": distribution_summary(boundary_distance),
        "gaussian_future_rms": gaussian_summary,
        "history_to_gaussian_mean_distance_ratio": distance_ratio,
        "history_closer_than_gaussian": distance_advantage,
        "errors": errors,
        "warnings": warnings,
        "shared_statistics": statistics,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A2A windows NPZ")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument(
        "--channel-spec",
        type=Path,
        help=(
            "Optional JSON with channels, available_state_keys and available_action_keys for "
            "mapping validation"
        ),
    )
    parser.add_argument(
        "--jump-threshold",
        type=float,
        default=5.0,
        help="Temporal jump threshold measured in shared standard deviations",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument(
        "--fail-on-errors",
        action="store_true",
        help="Return exit code 2 when hard audit errors are present",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    history, future, history_mask, future_mask = load_a2a_npz(args.input)
    if history_mask is None or future_mask is None:
        raise KeyError(
            "Strict A2A audit requires both history_mask and future_mask; re-export the "
            "windows instead of inferring validity"
        )
    history_horizon, history_dim = _trajectory_horizon_and_dim(history, "history")
    future_horizon, future_dim = _trajectory_horizon_and_dim(future, "future")
    if history_dim != future_dim:
        raise ValueError(
            f"history/future feature dimensions differ: {history_dim} and {future_dim}"
        )
    metadata = load_a2a_npz_export_metadata(args.input)
    num_windows = 1 if np.asarray(history).ndim == 2 else int(np.asarray(history).shape[0])
    expected_metadata = {
        "input_space": "unnormalized_canonical_physical",
        "num_windows": num_windows,
        "history_horizon": history_horizon,
        "future_horizon": future_horizon,
        "feature_dim": history_dim,
    }
    mismatches = {
        key: {"expected": expected, "actual": metadata[key]}
        for key, expected in expected_metadata.items()
        if metadata[key] != expected
    }
    if mismatches:
        raise ValueError(f"A2A NPZ exporter metadata does not match its arrays: {mismatches}")
    contract = validate_channel_contract(
        metadata["contract"],
        feature_dim=history_dim,
        history_horizon=history_horizon,
        future_horizon=future_horizon,
    )
    if contract["sha256"] != metadata["contract_sha256"]:
        raise ValueError("A2A NPZ contract_json does not match contract_sha256")
    report = audit_a2a_arrays(
        history,
        future,
        history_mask,
        future_mask,
        jump_threshold=args.jump_threshold,
        random_seed=args.random_seed,
    )
    report["export_contract_sha256"] = contract["sha256"]

    if args.channel_spec:
        spec_payload = json.loads(args.channel_spec.read_text(encoding="utf-8"))
        report["channel_spec"] = audit_channel_spec(
            spec_payload.get("channels", []),
            spec_payload.get("available_state_keys", []),
            spec_payload.get("available_action_keys", []),
        )
        report["errors"].extend(report["channel_spec"]["errors"])

    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(f"Wrote A2A data audit to {args.output}")
    else:
        print(rendered, end="")
    return 2 if args.fail_on_errors and report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
