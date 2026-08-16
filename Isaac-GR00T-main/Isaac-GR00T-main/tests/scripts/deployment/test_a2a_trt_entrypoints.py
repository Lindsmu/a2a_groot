# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from gr00t.deployment.modes import BenchmarkMode, BuildEngineMode, InferenceMode, VerifyMode
import pytest
from scripts.deployment.a2a_trt_model_forward import (
    A2A_ENGINE_INPUT_NAMES,
    setup_a2a_tensorrt_engine,
)


_REPO_ROOT = Path(__file__).resolve().parents[3]


def _policy(
    model_type: str = "Gr00tN1d7A2A",
    *,
    data_contract_sha256: str = "a" * 64,
) -> SimpleNamespace:
    config = SimpleNamespace(
        model_type=model_type,
        a2a_history_horizon=8,
        a2a_future_horizon=8,
        a2a_num_inference_steps=1,
        a2a_latent_dim=512,
        a2a_flow_backbone="mlp",
        a2a_dit_token_dim=256,
        a2a_dit_num_layers=8,
        a2a_dit_num_heads=8,
        a2a_dit_mlp_ratio=4,
        a2a_dit_dropout=0.0,
        a2a_expected_contract_sha256=data_contract_sha256,
        max_action_dim=64,
    )
    processor = SimpleNamespace(a2a_canonical_statistics={"version": 1, "value": [1.0]})
    return SimpleNamespace(
        model=SimpleNamespace(config=config),
        processor=processor,
        model_path=Path("unused-for-early-validation-test"),
    )


def test_a2a_modes_are_registered() -> None:
    assert InferenceMode("a2a_action_head") is InferenceMode.a2a_action_head
    assert BuildEngineMode("a2a_action_head") is BuildEngineMode.a2a_action_head
    assert BenchmarkMode("a2a_action_head") is BenchmarkMode.a2a_action_head
    assert VerifyMode("a2a_action_head") is VerifyMode.a2a_action_head


def test_a2a_setup_rejects_non_a2a_policy_before_loading_engine(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Gr00tN1d7A2A"):
        setup_a2a_tensorrt_engine(_policy("Gr00tN1d7"), tmp_path)


def test_a2a_setup_checks_engine_sha_before_importing_tensorrt(tmp_path: Path) -> None:
    engine_path = tmp_path / "a2a_action_head.engine"
    engine_path.write_bytes(b"engine-under-test")
    metadata = {
        "model_type": "Gr00tN1d7A2A",
        "engine": engine_path.name,
        "flow_backbone": "mlp",
        "history_horizon": 8,
        "future_horizon": 8,
        "action_dim": 64,
        "num_inference_steps": 1,
        "inputs": list(A2A_ENGINE_INPUT_NAMES),
        "engine_sha256": hashlib.sha256(b"different-engine").hexdigest(),
    }
    (tmp_path / "a2a_export_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA256 mismatch"):
        setup_a2a_tensorrt_engine(_policy(), tmp_path)


def test_a2a_builder_declares_fp32_profile_and_sha_contract() -> None:
    source = (_REPO_ROOT / "scripts" / "deployment" / "build_tensorrt_engine.py").read_text(
        encoding="utf-8"
    )
    assert 'dim_name in {"batch", "batch_size"}' in source
    assert "def build_a2a_action_head_engine(" in source
    assert 'opt_seq_lens={"vl_sequence": int(vl_sequence_length)}' in source
    assert 'precision="fp32"' in source
    assert '"engine_sha256": digest.hexdigest()' in source
    assert 'metadata["onnx_sha256"] != actual_onnx_sha256' in source


def test_standard_setup_routes_a2a_before_n17_bundle_contract() -> None:
    source = (_REPO_ROOT / "scripts" / "deployment" / "trt_model_forward.py").read_text(
        encoding="utf-8"
    )
    route = source.index("if mode == InferenceMode.a2a_action_head:")
    n17_contract = source.index("assert_engine_bundle_present", route)
    assert route < n17_contract
    assert "InferenceMode.a2a_action_head: _setup_a2a_action_head" in source
