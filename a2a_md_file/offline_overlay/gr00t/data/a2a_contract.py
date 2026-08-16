# SPDX-License-Identifier: Apache-2.0

"""Versioned, hashable data contract for canonical latent-A2A trajectories."""

from __future__ import annotations

import hashlib
import json
from typing import Any


A2A_CHANNEL_CONTRACT_VERSION = 1


def canonical_contract_sha256(contract: dict[str, Any]) -> str:
    """Return a deterministic SHA-256 over a contract excluding any stored hash."""
    payload = dict(contract)
    payload.pop("sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_statistics_sha256(statistics: dict[str, Any]) -> str:
    """Hash the complete statistics payload, excluding only its stored hash."""
    payload = dict(statistics)
    payload.pop("statistics_sha256", None)
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_channel_contract(
    contract: dict[str, Any],
    *,
    feature_dim: int,
    history_horizon: int,
    future_horizon: int,
) -> dict[str, Any]:
    """Validate the generic shape/provenance fields and return a normalized copy."""
    if not isinstance(contract, dict):
        raise TypeError("A2A channel contract must be a JSON object")
    normalized = dict(contract)
    if normalized.get("version") != A2A_CHANNEL_CONTRACT_VERSION:
        raise ValueError(
            "Unsupported A2A channel contract version: "
            f"{normalized.get('version')!r}"
        )
    expected = {
        "feature_dim": int(feature_dim),
        "history_horizon": int(history_horizon),
        "future_horizon": int(future_horizon),
    }
    for key, value in expected.items():
        if int(normalized.get(key, -1)) != value:
            raise ValueError(
                f"A2A channel contract {key}={normalized.get(key)!r}, expected {value}"
            )
    if not isinstance(normalized.get("embodiment"), str) or not normalized["embodiment"]:
        raise ValueError("A2A channel contract requires a non-empty embodiment")
    if not isinstance(normalized.get("channels"), list) or not normalized["channels"]:
        raise ValueError("A2A channel contract requires a non-empty channels list")
    provenance = normalized.get("provenance")
    required_provenance = (
        "dataset",
        "dataset_revision",
        "dataset_fingerprint_sha256",
        "source_schema_sha256",
        "canonicalizer",
        "canonicalizer_version",
        "canonicalizer_sha256",
        "exporter_version",
        "target_definition",
        "time_alignment",
    )
    if not isinstance(provenance, dict) or not all(
        isinstance(provenance.get(key), str) and provenance[key]
        for key in required_provenance
    ):
        raise ValueError(
            "A2A channel contract provenance is incomplete; it must bind the dataset/schema, "
            "canonicalizer code, exporter, target definition, and time alignment"
        )
    for key in (
        "dataset_fingerprint_sha256",
        "source_schema_sha256",
        "canonicalizer_sha256",
    ):
        value = provenance[key]
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError(f"A2A provenance {key} must be a lowercase SHA-256")
    controller = normalized.get("controller")
    required_controller = (
        "type",
        "version",
        "frame",
        "control_delta",
        "rotation_composition",
        "scale",
    )
    if not isinstance(controller, dict) or any(key not in controller for key in required_controller):
        raise ValueError(
            "A2A channel contract must explicitly bind controller type/version/scale/frame"
        )
    channels = normalized["channels"]
    expected_start = 0
    seen_action_keys = set()
    for channel in channels:
        if not isinstance(channel, dict):
            raise TypeError("Each A2A contract channel must be a JSON object")
        action_key = channel.get("action_key")
        if not isinstance(action_key, str) or not action_key or action_key in seen_action_keys:
            raise ValueError("A2A contract action keys must be unique non-empty strings")
        seen_action_keys.add(action_key)
        start = int(channel.get("start", -1))
        end = int(channel.get("end", -1))
        if start != expected_start or end <= start or end > feature_dim:
            raise ValueError(
                "A2A contract channels must be positive, contiguous, non-overlapping slices"
            )
        expected_start = end
    stored_hash = normalized.get("sha256")
    computed_hash = canonical_contract_sha256(normalized)
    if stored_hash is not None and stored_hash != computed_hash:
        raise ValueError("A2A channel contract SHA-256 does not match its contents")
    normalized["sha256"] = computed_hash
    return normalized


__all__ = [
    "A2A_CHANNEL_CONTRACT_VERSION",
    "canonical_contract_sha256",
    "canonical_statistics_sha256",
    "validate_channel_contract",
]
