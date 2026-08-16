#!/usr/bin/env python

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build shared normalization statistics for canonical A2A trajectories.

The action-to-action encoder is shared by executed history and future targets,
so using independent state/action statistics would put its two inputs in
different coordinate systems.  This utility pools both sides and emits one set
of statistics.  Only dimensions enabled on *both* sides participate; inactive
padded dimensions receive neutral ``mean=0, std=1`` values plus an explicit
``active_channel_mask``.

The CLI consumes an NPZ emitted by ``export_canonical_windows.py`` with arrays
``history``, ``future``, ``history_mask`` and ``future_mask``.  The pure NumPy
functions are reusable from conversion jobs without invoking the CLI.  History
and future arrays must be unnormalized canonical physical trajectories; feeding
already-normalized arrays would create a second, incompatible normalization.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from gr00t.data.a2a_contract import canonical_statistics_sha256, validate_channel_contract
import numpy as np


CANONICAL_STATS_VERSION = 1


def as_trajectory_batch(array: np.ndarray, name: str) -> np.ndarray:
    """Convert ``[T,D]`` or ``[N,T,D]`` input to a float64 batch."""

    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape [T,D] or [N,T,D], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be numeric, got dtype {array.dtype}")
    return array.astype(np.float64, copy=False)


def broadcast_trajectory_mask(
    mask: np.ndarray | None,
    shape: tuple[int, int, int],
    name: str,
) -> np.ndarray:
    """Validate and broadcast a mask to ``[N,T,D]``."""

    if mask is None:
        return np.ones(shape, dtype=bool)
    mask = np.asarray(mask)
    if not (np.issubdtype(mask.dtype, np.bool_) or np.issubdtype(mask.dtype, np.number)):
        raise TypeError(f"{name} must be boolean or numeric, got dtype {mask.dtype}")
    if np.issubdtype(mask.dtype, np.number):
        if not np.isfinite(mask).all() or not np.isin(mask, [0, 1]).all():
            raise ValueError(f"{name} must contain only finite 0/1 values")
    try:
        return np.broadcast_to(mask.astype(bool, copy=False), shape).copy()
    except ValueError as error:
        raise ValueError(f"{name} with shape {mask.shape} cannot broadcast to {shape}") from error


def prepare_canonical_inputs(
    history: np.ndarray,
    future: np.ndarray,
    history_mask: np.ndarray | None = None,
    future_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate canonical trajectories and return their shared channel mask."""

    history = as_trajectory_batch(history, "history")
    future = as_trajectory_batch(future, "future")
    if 0 in history.shape or 0 in future.shape:
        raise ValueError(
            f"Canonical trajectory batches must be non-empty, got {history.shape} and {future.shape}"
        )
    if history.shape[0] != future.shape[0]:
        raise ValueError(
            "history and future must have the same number of windows, got "
            f"{history.shape[0]} and {future.shape[0]}"
        )
    if history.shape[2] != future.shape[2]:
        raise ValueError(
            "history and future must share canonical feature dimension, got "
            f"{history.shape[2]} and {future.shape[2]}"
        )

    history_mask = broadcast_trajectory_mask(history_mask, history.shape, "history_mask")
    future_mask = broadcast_trajectory_mask(future_mask, future.shape, "future_mask")

    # A channel is canonical-A2A compatible only if both executed proprio and
    # future target supply at least one valid element for it.  Per-element masks
    # remain in force after this dataset-level intersection.
    shared_channel_mask = np.any(history_mask, axis=(0, 1)) & np.any(future_mask, axis=(0, 1))
    history_mask &= shared_channel_mask[None, None, :]
    future_mask &= shared_channel_mask[None, None, :]
    return history, future, history_mask, future_mask, shared_channel_mask


def compute_shared_canonical_statistics(
    history: np.ndarray,
    future: np.ndarray,
    history_mask: np.ndarray | None = None,
    future_mask: np.ndarray | None = None,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    min_std: float = 1e-6,
    allow_nonfinite: bool = False,
    channel_contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute one set of per-channel stats from history and future values.

    Non-finite values under an active mask fail by default so a stats build can
    never silently legitimize corrupt training data.  Audit callers may set
    ``allow_nonfinite=True`` to obtain a diagnostic report from the remaining
    finite values.
    """

    if not 0 <= lower_quantile < upper_quantile <= 1:
        raise ValueError(
            "Quantiles must satisfy 0 <= lower < upper <= 1, got "
            f"{lower_quantile}, {upper_quantile}"
        )
    if min_std <= 0:
        raise ValueError(f"min_std must be positive, got {min_std}")

    history, future, history_mask, future_mask, shared_channel_mask = prepare_canonical_inputs(
        history, future, history_mask, future_mask
    )
    history_bad = history_mask & ~np.isfinite(history)
    future_bad = future_mask & ~np.isfinite(future)
    nonfinite_count = int(history_bad.sum() + future_bad.sum())
    if nonfinite_count and not allow_nonfinite:
        raise ValueError(
            f"Canonical trajectories contain {nonfinite_count} non-finite active values; "
            "run audit_a2a_data.py before building statistics"
        )
    history_mask &= np.isfinite(history)
    future_mask &= np.isfinite(future)

    feature_dim = history.shape[2]
    mean = np.zeros(feature_dim, dtype=np.float64)
    std = np.ones(feature_dim, dtype=np.float64)
    minimum = np.zeros(feature_dim, dtype=np.float64)
    maximum = np.zeros(feature_dim, dtype=np.float64)
    q_low = np.zeros(feature_dim, dtype=np.float64)
    q_high = np.zeros(feature_dim, dtype=np.float64)
    valid_count = np.zeros(feature_dim, dtype=np.int64)
    history_valid_count = history_mask.sum(axis=(0, 1), dtype=np.int64)
    future_valid_count = future_mask.sum(axis=(0, 1), dtype=np.int64)
    constant_mask = np.zeros(feature_dim, dtype=bool)

    for channel in np.flatnonzero(shared_channel_mask):
        values = np.concatenate(
            [
                history[..., channel][history_mask[..., channel]],
                future[..., channel][future_mask[..., channel]],
            ]
        )
        if values.size == 0:
            shared_channel_mask[channel] = False
            continue
        channel_std = float(values.std())
        mean[channel] = float(values.mean())
        std[channel] = max(channel_std, min_std)
        minimum[channel] = float(values.min())
        maximum[channel] = float(values.max())
        q_low[channel] = float(np.quantile(values, lower_quantile))
        q_high[channel] = float(np.quantile(values, upper_quantile))
        valid_count[channel] = values.size
        constant_mask[channel] = channel_std < min_std or q_high[channel] - q_low[channel] < min_std

    result = {
        "version": CANONICAL_STATS_VERSION,
        "normalization": "shared_history_future_continuous_intersection",
        "input_space": "unnormalized_canonical_physical",
        "feature_dim": feature_dim,
        "num_windows": history.shape[0],
        "history_horizon": history.shape[1],
        "future_horizon": future.shape[1],
        "lower_quantile": lower_quantile,
        "upper_quantile": upper_quantile,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "min": minimum.tolist(),
        "max": maximum.tolist(),
        "q01": q_low.tolist(),
        "q99": q_high.tolist(),
        "q_low": q_low.tolist(),
        "q_high": q_high.tolist(),
        "valid_count": valid_count.tolist(),
        "history_valid_count": history_valid_count.tolist(),
        "future_valid_count": future_valid_count.tolist(),
        "active_channel_mask": shared_channel_mask.tolist(),
        "constant_mask": constant_mask.tolist(),
        "nonfinite_active_count": nonfinite_count,
    }
    if channel_contract is not None:
        result["channel_contract"] = validate_channel_contract(
            channel_contract,
            feature_dim=feature_dim,
            history_horizon=history.shape[1],
            future_horizon=future.shape[1],
        )
    result["statistics_sha256"] = canonical_statistics_sha256(result)
    return result


def save_canonical_statistics(statistics: dict[str, Any], output_path: str | Path) -> Path:
    """Write canonical statistics as deterministic, human-readable JSON."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(statistics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_path


def load_a2a_npz(
    input_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    """Load the standard A2A audit interchange NPZ."""

    with np.load(Path(input_path), allow_pickle=False) as payload:
        missing = {"history", "future"} - set(payload.files)
        if missing:
            raise KeyError(f"A2A NPZ is missing required arrays: {sorted(missing)}")
        history = payload["history"]
        future = payload["future"]
        history_mask = payload["history_mask"] if "history_mask" in payload else None
        future_mask = payload["future_mask"] if "future_mask" in payload else None
    return history, future, history_mask, future_mask


def load_a2a_npz_export_metadata(input_path: str | Path) -> dict[str, Any]:
    """Load the provenance written by ``export_canonical_windows.py``.

    Requiring the complete scalar metadata prevents an arbitrary or already
    normalized NPZ from becoming trusted merely by copying a contract hash.
    """

    required = {
        "input_space",
        "contract_json",
        "contract_sha256",
        "num_windows",
        "history_horizon",
        "future_horizon",
        "feature_dim",
    }
    with np.load(Path(input_path), allow_pickle=False) as payload:
        missing = required - set(payload.files)
        if missing:
            raise KeyError(
                "A2A NPZ is missing exporter provenance fields "
                f"{sorted(missing)}; re-export it with export_canonical_windows.py"
            )
        metadata = {}
        for key in required:
            encoded = np.asarray(payload[key])
            if encoded.shape != ():
                raise ValueError(
                    f"A2A NPZ {key} must be one scalar value, got shape {encoded.shape}"
                )
            metadata[key] = encoded.item()

    for key in ("input_space", "contract_json", "contract_sha256"):
        if not isinstance(metadata[key], str):
            raise TypeError(f"A2A NPZ {key} must be a Unicode string")
    contract_sha256 = metadata["contract_sha256"]
    if len(contract_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in contract_sha256
    ):
        raise ValueError("A2A NPZ contract_sha256 must be a lowercase SHA-256")
    for key in ("num_windows", "history_horizon", "future_horizon", "feature_dim"):
        value = metadata[key]
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"A2A NPZ {key} must be a positive integer, got {value!r}")
        metadata[key] = int(value)
    try:
        metadata["contract"] = json.loads(metadata["contract_json"])
    except json.JSONDecodeError as error:
        raise ValueError("A2A NPZ contract_json is not valid JSON") from error
    return metadata


def _trajectory_horizon_and_dim(array: np.ndarray, name: str) -> tuple[int, int]:
    """Read temporal/feature dimensions without materializing a float64 copy."""

    shape = np.asarray(array).shape
    if len(shape) not in {2, 3}:
        raise ValueError(f"{name} must have shape [T,D] or [N,T,D], got {shape}")
    return int(shape[-2]), int(shape[-1])


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="A2A windows NPZ")
    parser.add_argument("--output", type=Path, required=True, help="Output statistics JSON")
    parser.add_argument("--lower-quantile", type=float, default=0.01)
    parser.add_argument("--upper-quantile", type=float, default=0.99)
    parser.add_argument("--min-std", type=float, default=1e-6)
    parser.add_argument(
        "--contract",
        type=Path,
        required=True,
        help="Versioned channel/provenance contract JSON bound into the statistics",
    )
    parser.add_argument(
        "--allow-nonfinite",
        action="store_true",
        help="Ignore non-finite active values after recording their count (audit only)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    history, future, history_mask, future_mask = load_a2a_npz(args.input)
    if history_mask is None or future_mask is None:
        raise KeyError(
            "Strict canonical statistics require both history_mask and future_mask; "
            "re-export the windows instead of inferring validity"
        )
    history_horizon, history_dim = _trajectory_horizon_and_dim(history, "history")
    future_horizon, future_dim = _trajectory_horizon_and_dim(future, "future")
    if history_dim != future_dim:
        raise ValueError(
            f"history/future feature dimensions differ: {history_dim} and {future_dim}"
        )
    channel_contract = validate_channel_contract(
        json.loads(args.contract.read_text(encoding="utf-8")),
        feature_dim=history_dim,
        history_horizon=history_horizon,
        future_horizon=future_horizon,
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
    exported_contract = validate_channel_contract(
        metadata["contract"],
        feature_dim=history_dim,
        history_horizon=history_horizon,
        future_horizon=future_horizon,
    )
    if (
        metadata["contract_sha256"] != exported_contract["sha256"]
        or exported_contract["sha256"] != channel_contract["sha256"]
    ):
        raise ValueError(
            "A2A NPZ contract_sha256/contract_json does not match --contract: "
            f"npz_hash={metadata['contract_sha256']}, "
            f"npz_contract={exported_contract['sha256']}, "
            f"cli_contract={channel_contract['sha256']}"
        )
    statistics = compute_shared_canonical_statistics(
        history,
        future,
        history_mask,
        future_mask,
        lower_quantile=args.lower_quantile,
        upper_quantile=args.upper_quantile,
        min_std=args.min_std,
        allow_nonfinite=args.allow_nonfinite,
        channel_contract=channel_contract,
    )
    save_canonical_statistics(statistics, args.output)
    print(f"Wrote shared canonical A2A statistics to {args.output}")
    print(f"Channel contract SHA-256: {statistics['channel_contract']['sha256']}")
    print(f"Statistics SHA-256: {statistics['statistics_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
