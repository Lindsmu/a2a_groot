# SPDX-License-Identifier: Apache-2.0

"""TensorRT patch for the fused latent A2A action-head ONNX engine."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from types import MethodType

from gr00t.deployment.a2a_artifacts import (
    canonical_json_sha256,
    canonical_statistics_sha256,
    checkpoint_sha256,
    sha256_file,
)
from transformers.feature_extraction_utils import BatchFeature


A2A_ENGINE_INPUT_NAMES = (
    "backbone_features",
    "backbone_attention_mask",
    "image_mask",
    "history_action_canonical",
    "history_action_mask",
    "continuous_action_mask",
    "auxiliary_action_mask",
    "binary_action_mask",
    "categorical_action_mask",
    "embodiment_id",
)

A2A_ENGINE_INPUT_DTYPES = {
    "backbone_features": "float32",
    "backbone_attention_mask": "int64",
    "image_mask": "bool",
    "history_action_canonical": "float32",
    "history_action_mask": "float32",
    "continuous_action_mask": "float32",
    "auxiliary_action_mask": "float32",
    "binary_action_mask": "float32",
    "categorical_action_mask": "float32",
    "embodiment_id": "int64",
}


def _expected_export_contract(config):
    return {
        "model_type": "Gr00tN1d7A2A",
        "source": "executed_proprio_history",
        "latent_dim": int(config.a2a_latent_dim),
        "flow_backbone": str(config.a2a_flow_backbone),
        "dit_token_dim": int(config.a2a_dit_token_dim),
        "dit_num_layers": int(config.a2a_dit_num_layers),
        "dit_num_heads": int(config.a2a_dit_num_heads),
        "dit_mlp_ratio": int(config.a2a_dit_mlp_ratio),
        "dit_dropout": float(config.a2a_dit_dropout),
        "history_horizon": int(config.a2a_history_horizon),
        "future_horizon": int(config.a2a_future_horizon),
        "action_dim": int(config.max_action_dim),
        "num_inference_steps": int(config.a2a_num_inference_steps),
        "random_initialization": False,
        "inputs": list(A2A_ENGINE_INPUT_NAMES),
        "input_dtypes": A2A_ENGINE_INPUT_DTYPES,
    }


def _run_a2a_engine(engine, tensors):
    prepared = {}
    for name in A2A_ENGINE_INPUT_NAMES:
        value = tensors[name]
        expected_dtype = engine.dtype_of(name)
        value = value.to(dtype=expected_dtype).contiguous()
        engine.set_runtime_tensor_shape(name, tuple(value.shape))
        prepared[name] = value
    return engine(**prepared)


def _a2a_trt_get_action(self, backbone_output, action_input, options=None):
    if options and int(options.get("num_inference_steps", self.num_inference_timesteps)) != int(
        self.num_inference_timesteps
    ):
        raise ValueError(
            "The fused A2A TensorRT graph has a fixed number of Euler steps; re-export "
            "the checkpoint to change it"
        )
    result = _run_a2a_engine(
        self.a2a_action_head_engine,
        {
            "backbone_features": backbone_output["backbone_features"],
            "backbone_attention_mask": backbone_output["backbone_attention_mask"],
            "image_mask": backbone_output["image_mask"],
            "history_action_canonical": action_input["history_action_canonical"],
            "history_action_mask": action_input["history_action_mask"],
            "continuous_action_mask": action_input["continuous_action_mask"],
            "auxiliary_action_mask": action_input["auxiliary_action_mask"],
            "binary_action_mask": action_input["binary_action_mask"],
            "categorical_action_mask": action_input["categorical_action_mask"],
            "embodiment_id": action_input["embodiment_id"],
        },
    )
    output = result.get("action_pred", result.get("output"))
    if output is None:
        raise KeyError(f"A2A TensorRT engine returned unexpected outputs: {sorted(result)}")
    return BatchFeature(
        data={
            "action_pred": output,
            "num_inference_steps": self.num_inference_timesteps,
        }
    )


def _validate_a2a_engine_metadata(policy, engine_directory: Path, engine_path: Path) -> dict:
    metadata_path = engine_directory / "a2a_export_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; build the engine with a2a_action_head mode"
        )
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

    config = policy.model.config
    expected = {
        "model_type": "Gr00tN1d7A2A",
        "engine": engine_path.name,
        "flow_backbone": str(config.a2a_flow_backbone),
        "history_horizon": int(config.a2a_history_horizon),
        "future_horizon": int(config.a2a_future_horizon),
        "action_dim": int(config.max_action_dim),
        "num_inference_steps": int(config.a2a_num_inference_steps),
    }
    mismatches = {
        key: {"expected": value, "actual": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(f"A2A TensorRT metadata/config mismatch: {mismatches}")

    metadata_inputs = tuple(metadata.get("inputs", ()))
    if metadata_inputs != A2A_ENGINE_INPUT_NAMES:
        raise ValueError(
            "A2A TensorRT metadata input contract mismatch: "
            f"expected={A2A_ENGINE_INPUT_NAMES}, actual={metadata_inputs}"
        )
    expected_sha = metadata.get("engine_sha256")
    if not isinstance(expected_sha, str) or len(expected_sha) != 64:
        raise ValueError("A2A TensorRT metadata is missing a valid engine_sha256")
    actual_sha = sha256_file(engine_path)
    if actual_sha != expected_sha:
        raise ValueError(
            f"A2A TensorRT engine SHA256 mismatch: expected={expected_sha}, actual={actual_sha}"
        )

    expected_data_contract = getattr(config, "a2a_expected_contract_sha256", None)
    if metadata.get("data_contract_sha256") != expected_data_contract:
        raise ValueError("A2A TensorRT data-contract SHA-256 does not match the loaded checkpoint")
    if not hasattr(policy, "processor"):
        raise ValueError("A2A TensorRT validation requires the checkpoint processor")
    expected_statistics_sha256 = canonical_statistics_sha256(policy.processor)
    if metadata.get("canonical_statistics_sha256") != expected_statistics_sha256:
        raise ValueError(
            "A2A TensorRT canonical-statistics SHA-256 does not match the loaded processor"
        )
    model_path = getattr(policy, "model_path", None)
    if model_path is None:
        raise ValueError("A2A TensorRT validation requires policy.model_path")
    expected_checkpoint_sha256 = checkpoint_sha256(model_path)
    if metadata.get("checkpoint_sha256") != expected_checkpoint_sha256:
        raise ValueError("A2A TensorRT checkpoint SHA-256 does not match the loaded model weights")
    onnx_sha256 = metadata.get("onnx_sha256")
    if not isinstance(onnx_sha256, str) or len(onnx_sha256) != 64:
        raise ValueError("A2A TensorRT metadata is missing a valid onnx_sha256")
    if metadata.get("verification") != "onnx_checker+torch_ort_rtol1e-4_atol1e-4":
        raise ValueError("A2A TensorRT metadata does not record a successful ONNX oracle")
    return metadata


def setup_a2a_tensorrt_engine(policy, engine_directory: str | Path):
    """Replace only the A2A action-head forward; the backbone may remain eager or TRT."""
    if getattr(policy.model.config, "model_type", None) != "Gr00tN1d7A2A":
        raise ValueError("A2A TensorRT setup requires a Gr00tN1d7A2A policy")
    engine_directory = Path(engine_directory)
    engine_path = engine_directory / "a2a_action_head.engine"
    if not engine_path.is_file():
        raise FileNotFoundError(f"Missing {engine_path}; build it from a2a_action_head.onnx first")
    _validate_a2a_engine_metadata(policy, engine_directory, engine_path)
    metadata_path = Path(engine_directory) / "a2a_export_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing {metadata_path}; A2A TensorRT setup requires export/build metadata"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_contract = _expected_export_contract(policy.model.config)
    actual_contract = {key: metadata.get(key) for key in expected_contract}
    if actual_contract != expected_contract:
        raise ValueError("A2A TensorRT metadata does not match the loaded checkpoint configuration")
    expected_contract_hash = canonical_json_sha256(expected_contract)
    if metadata.get("contract_sha256") != expected_contract_hash:
        raise ValueError("A2A TensorRT metadata contract hash is invalid")
    if metadata.get("engine_sha256") is not None and metadata["engine_sha256"] != sha256_file(
        engine_path
    ):
        raise ValueError("A2A TensorRT engine SHA-256 does not match export metadata")
    # scripts/deployment is intentionally not a Python package. Resolve the
    # sibling loader lazily so importing this helper does not require TensorRT.
    deploy_directory = os.path.dirname(os.path.abspath(__file__))
    if deploy_directory not in sys.path:
        sys.path.insert(0, deploy_directory)
    from trt_torch import Engine

    head = policy.model.action_head
    head.a2a_action_head_engine = Engine(str(engine_path))
    actual_inputs = tuple(item[0] for item in head.a2a_action_head_engine.in_meta)
    if set(actual_inputs) != set(A2A_ENGINE_INPUT_NAMES):
        raise ValueError(
            "A2A TensorRT engine input contract mismatch: "
            f"expected={A2A_ENGINE_INPUT_NAMES}, actual={actual_inputs}"
        )
    head.get_action = MethodType(_a2a_trt_get_action, head)
    return policy


__all__ = [
    "A2A_ENGINE_INPUT_NAMES",
    "A2A_ENGINE_INPUT_DTYPES",
    "setup_a2a_tensorrt_engine",
]
