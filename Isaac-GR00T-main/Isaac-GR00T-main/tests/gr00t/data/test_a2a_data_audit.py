# SPDX-License-Identifier: Apache-2.0

import json

from gr00t.data.a2a_contract import canonical_contract_sha256
import numpy as np
import pytest
from scripts.a2a.audit_a2a_data import audit_a2a_arrays, audit_channel_spec, main as audit_main
from scripts.a2a.build_canonical_stats import (
    compute_shared_canonical_statistics,
    main as stats_main,
)


def test_shared_stats_pool_history_and_future_on_continuous_intersection():
    history = np.array([[[0.0, 10.0, 100.0], [2.0, 12.0, 101.0]]])
    future = np.array([[[4.0, 14.0, 200.0], [6.0, 16.0, 201.0]]])
    history_mask = np.array([[[1, 1, 0], [1, 1, 0]]], dtype=bool)
    future_mask = np.ones_like(future, dtype=bool)

    stats = compute_shared_canonical_statistics(history, future, history_mask, future_mask)

    assert stats["normalization"] == "shared_history_future_continuous_intersection"
    assert stats["active_channel_mask"] == [True, True, False]
    assert stats["mean"] == pytest.approx([3.0, 13.0, 0.0])
    assert stats["valid_count"] == [4, 4, 0]
    assert stats["std"][2] == 1.0


def test_stats_reject_nonfinite_active_values_but_audit_reports_them():
    history = np.array([[[0.0], [np.nan]]])
    future = np.array([[[1.0], [2.0]]])

    with pytest.raises(ValueError, match="non-finite active values"):
        compute_shared_canonical_statistics(history, future)

    report = audit_a2a_arrays(history, future)
    assert report["history_nonfinite_active_count"] == 1
    assert report["errors"] == ["Non-finite values occur under active canonical masks"]


def test_audit_detects_history_distance_advantage_over_gaussian():
    history = np.array(
        [
            [[0.0, 1.0], [0.5, 1.5]],
            [[1.0, 2.0], [1.5, 2.5]],
            [[2.0, 3.0], [2.5, 3.5]],
        ]
    )
    future = history.copy()

    report = audit_a2a_arrays(history, future, random_seed=123)

    assert report["history_closer_than_gaussian"] is True
    assert report["history_to_gaussian_mean_distance_ratio"] == pytest.approx(0.0)
    assert report["history_future_trajectory_rms"]["mean"] == pytest.approx(0.0)
    assert report["errors"] == []


def test_audit_supports_different_history_and_future_horizons():
    history = np.zeros((2, 3, 2))
    future = np.zeros((2, 5, 2))

    report = audit_a2a_arrays(history, future)

    assert report["history_future_trajectory_rms"] is None
    assert report["history_last_future_first_rms"]["count"] == 2


def test_channel_spec_requires_executed_source_for_continuous_actions():
    report = audit_channel_spec(
        [
            {
                "action_key": "arm",
                "source_state_key": "joint_positions",
                "kind": "continuous",
            },
            {"action_key": "gripper", "source_state_key": None, "kind": "binary"},
            {"action_key": "base", "source_state_key": "missing", "kind": "continuous"},
        ],
        available_state_keys={"joint_positions"},
        available_action_keys={"arm", "gripper", "base"},
    )

    assert report["missing_source_state_keys"] == ["base"]
    assert report["discrete_channels"] == ["gripper"]
    assert report["errors"]


def test_stats_and_audit_cli_npz_contract(tmp_path):
    input_path = tmp_path / "windows.npz"
    stats_path = tmp_path / "a2a_statistics.json"
    audit_path = tmp_path / "a2a_audit.json"
    contract_path = tmp_path / "channel_contract.json"
    history = np.arange(24, dtype=np.float32).reshape(2, 4, 3)
    future = history + 0.25
    mask = np.ones_like(history, dtype=np.uint8)
    contract = {
        "version": 1,
        "embodiment": "test_robot",
        "feature_dim": 3,
        "history_horizon": 4,
        "future_horizon": 4,
        "channels": [
            {
                "action_key": "arm",
                "source_state_key": "arm",
                "kind": "continuous",
                "start": 0,
                "end": 3,
                "canonical_format": "joint_position",
                "source_format": "joint_position",
                "target_format": "joint_position",
                "source_unit": "radian",
                "target_unit": "radian",
                "source_frame": "joint",
                "target_frame": "joint",
            }
        ],
        "provenance": {
            "dataset": "unit_test",
            "dataset_revision": "1",
            "dataset_fingerprint_sha256": "0" * 64,
            "source_schema_sha256": "1" * 64,
            "canonicalizer": "identity",
            "canonicalizer_version": "1",
            "canonicalizer_sha256": "2" * 64,
            "exporter_version": "1",
            "target_definition": "absolute_joint_target",
            "time_alignment": "history_t_minus_3_to_t__future_t_to_t_plus_3",
        },
        "controller": {
            "type": "joint_position",
            "version": "1",
            "frame": "joint",
            "control_delta": False,
            "rotation_composition": "none",
            "scale": [1.0, 1.0, 1.0],
        },
    }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    export_metadata = {
        "input_space": np.asarray("unnormalized_canonical_physical"),
        "contract_json": np.asarray(json.dumps(contract, sort_keys=True)),
        "contract_sha256": np.asarray(canonical_contract_sha256(contract)),
        "num_windows": np.asarray(2, dtype=np.int64),
        "history_horizon": np.asarray(4, dtype=np.int64),
        "future_horizon": np.asarray(4, dtype=np.int64),
        "feature_dim": np.asarray(3, dtype=np.int64),
    }
    np.savez(
        input_path,
        history=history,
        future=future,
        history_mask=mask,
        future_mask=mask,
        **export_metadata,
    )

    assert (
        stats_main(
            [
                "--input",
                str(input_path),
                "--output",
                str(stats_path),
                "--contract",
                str(contract_path),
            ]
        )
        == 0
    )
    assert (
        audit_main(
            [
                "--input",
                str(input_path),
                "--output",
                str(audit_path),
                "--fail-on-errors",
            ]
        )
        == 0
    )

    saved_stats = json.loads(stats_path.read_text())
    assert saved_stats["feature_dim"] == 3
    assert len(saved_stats["channel_contract"]["sha256"]) == 64
    saved_audit = json.loads(audit_path.read_text())
    assert saved_audit["num_windows"] == 2
    assert saved_audit["export_contract_sha256"] == canonical_contract_sha256(contract)

    np.savez(
        input_path,
        history=history,
        future=future,
        history_mask=mask,
        future_mask=mask,
        **{**export_metadata, "contract_sha256": np.asarray("f" * 64)},
    )
    with pytest.raises(ValueError, match="does not match --contract"):
        stats_main(
            [
                "--input",
                str(input_path),
                "--output",
                str(stats_path),
                "--contract",
                str(contract_path),
            ]
        )

    np.savez(
        input_path,
        history=history,
        future=future,
        history_mask=mask,
        future_mask=mask,
        **{
            **export_metadata,
            "input_space": np.asarray("already_normalized_model_space"),
        },
    )
    with pytest.raises(ValueError, match="exporter metadata"):
        stats_main(
            [
                "--input",
                str(input_path),
                "--output",
                str(stats_path),
                "--contract",
                str(contract_path),
            ]
        )

    np.savez(
        input_path,
        history=history,
        future=future,
        future_mask=mask,
        **export_metadata,
    )
    with pytest.raises(KeyError, match="both history_mask and future_mask"):
        stats_main(
            [
                "--input",
                str(input_path),
                "--output",
                str(stats_path),
                "--contract",
                str(contract_path),
            ]
        )
    with pytest.raises(KeyError, match="both history_mask and future_mask"):
        audit_main(["--input", str(input_path), "--output", str(audit_path)])
