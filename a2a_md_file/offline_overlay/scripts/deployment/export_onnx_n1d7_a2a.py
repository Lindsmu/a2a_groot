# SPDX-License-Identifier: Apache-2.0

"""Export the fixed-horizon latent A2A action head to one ONNX graph.

The graph starts from actual canonical proprio history and VLM backbone
features.  The configured Euler steps are unrolled; no random action tensor is
created in the exported graph.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from gr00t.deployment.a2a_artifacts import (
    canonical_statistics_sha256,
    checkpoint_sha256,
    sha256_file,
)

# Import registrations before AutoModel resolves the model type.
import gr00t.model  # noqa: F401
import numpy as np
import torch
from torch import nn
from transformers import AutoModel
from transformers.feature_extraction_utils import BatchFeature
import tyro


class A2AActionHeadExportWrapper(nn.Module):
    def __init__(self, action_head):
        super().__init__()
        self.action_head = action_head

    def forward(
        self,
        backbone_features,
        backbone_attention_mask,
        image_mask,
        history_action_canonical,
        history_action_mask,
        continuous_action_mask,
        auxiliary_action_mask,
        binary_action_mask,
        categorical_action_mask,
        embodiment_id,
    ):
        # future_action_mask is exactly the union of mutually-exclusive head
        # masks. Keeping one source of truth avoids ONNX DCE silently changing
        # an 11-input export into a 10-input TensorRT engine.
        future_action_mask = (
            continuous_action_mask
            + auxiliary_action_mask
            + binary_action_mask
            + categorical_action_mask
        ).clamp(0, 1)
        backbone = BatchFeature(
            data={
                "backbone_features": backbone_features,
                "backbone_attention_mask": backbone_attention_mask,
                "image_mask": image_mask,
            }
        )
        action = BatchFeature(
            data={
                "history_action_canonical": history_action_canonical,
                "history_action_mask": history_action_mask,
                "future_action_mask": future_action_mask,
                "continuous_action_mask": continuous_action_mask,
                "auxiliary_action_mask": auxiliary_action_mask,
                "binary_action_mask": binary_action_mask,
                "categorical_action_mask": categorical_action_mask,
                "embodiment_id": embodiment_id,
            }
        )
        return self.action_head.get_action(backbone, action)["action_pred"]


def _checkpoint_canonical_statistics(model_path: Path) -> dict:
    candidates = (
        model_path / "a2a_statistics.json",
        model_path / "processor" / "a2a_statistics.json",
    )
    for path in candidates:
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        statistics = payload.get("canonical_statistics", payload)
        if not isinstance(statistics, dict) or not statistics:
            raise ValueError(f"A2A canonical statistics in {path} are absent or invalid")
        return statistics
    raise FileNotFoundError(
        "A2A checkpoint is missing a2a_statistics.json at its root or processor/ directory"
    )


def _verify_a2a_onnx(
    onnx_path: Path,
    wrapper: A2AActionHeadExportWrapper,
    inputs: tuple[torch.Tensor, ...],
    input_names: list[str],
) -> None:
    """Run structural ONNX validation and a PyTorch/ORT fixed-input oracle."""

    try:
        import onnx
        import onnxruntime as ort
    except ImportError as error:
        raise RuntimeError(
            "A2A export verification requires both onnx and onnxruntime; install the "
            "deployment dependencies or pass verify_export=False explicitly"
        ) from error

    onnx.checker.check_model(onnx.load(str(onnx_path)))
    with torch.inference_mode():
        expected = wrapper(*inputs).detach().cpu().numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_inputs = {
        name: value.detach().cpu().numpy() for name, value in zip(input_names, inputs, strict=True)
    }
    actual = session.run(["action_pred"], ort_inputs)[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)


def export_a2a_action_head(
    model_path: str,
    output_dir: str,
    batch_size: int = 1,
    vl_sequence_length: int = 512,
    opset: int = 18,
    verify_export: bool = True,
) -> Path:
    model_path = Path(model_path).resolve()
    model = AutoModel.from_pretrained(model_path, local_files_only=True)
    if getattr(model.config, "model_type", None) != "Gr00tN1d7A2A":
        raise ValueError("Checkpoint is not a Gr00tN1d7A2A model")
    if float(model.config.a2a_history_noise_infer_std) != 0.0:
        raise ValueError("Deterministic A2A ONNX export requires a2a_history_noise_infer_std=0")
    if bool(model.config.a2a_include_current_state_condition):
        raise ValueError(
            "The optional current-state conditioning ablation is not part of the strict "
            "10-input A2A deployment contract; export a strict checkpoint instead"
        )
    data_contract_sha256 = getattr(model.config, "a2a_expected_contract_sha256", None)
    if not isinstance(data_contract_sha256, str) or len(data_contract_sha256) != 64:
        raise ValueError(
            "A2A ONNX export requires a checkpoint bound to a 64-character "
            "a2a_expected_contract_sha256"
        )
    statistics_payload = _checkpoint_canonical_statistics(model_path)
    statistics_sha256 = canonical_statistics_sha256(statistics_payload)
    model_checkpoint_sha256 = checkpoint_sha256(model_path)
    wrapper = A2AActionHeadExportWrapper(model.action_head.float().cpu()).eval()
    config = model.config
    history_shape = (
        batch_size,
        config.a2a_history_horizon,
        config.max_action_dim,
    )
    future_shape = (
        batch_size,
        config.a2a_future_horizon,
        config.max_action_dim,
    )
    continuous = torch.ones(future_shape, dtype=torch.float32)
    zeros = torch.zeros(future_shape, dtype=torch.float32)
    inputs = (
        torch.randn(batch_size, vl_sequence_length, config.backbone_embedding_dim),
        torch.ones(batch_size, vl_sequence_length, dtype=torch.long),
        torch.zeros(batch_size, vl_sequence_length, dtype=torch.bool),
        torch.randn(history_shape),
        torch.ones(history_shape),
        continuous,
        zeros,
        zeros,
        zeros,
        torch.zeros(batch_size, dtype=torch.long),
    )
    input_names = [
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
    ]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    onnx_path = output / "a2a_action_head.onnx"
    # The generic TensorRT builder recognizes the symbolic name
    # ``batch_size`` when deriving optimization profiles.
    dynamic_axes = {name: {0: "batch_size"} for name in input_names}
    dynamic_axes["backbone_features"][1] = "vl_sequence"
    dynamic_axes["backbone_attention_mask"][1] = "vl_sequence"
    dynamic_axes["image_mask"][1] = "vl_sequence"
    dynamic_axes["action_pred"] = {0: "batch_size"}
    torch.onnx.export(
        wrapper,
        inputs,
        onnx_path,
        input_names=input_names,
        output_names=["action_pred"],
        dynamic_axes=dynamic_axes,
        opset_version=opset,
        do_constant_folding=True,
        dynamo=False,
    )
    if verify_export:
        _verify_a2a_onnx(onnx_path, wrapper, inputs, input_names)
    contract = {
        "model_type": config.model_type,
        "source": "executed_proprio_history",
        "latent_dim": config.a2a_latent_dim,
        "flow_backbone": config.a2a_flow_backbone,
        "dit_token_dim": config.a2a_dit_token_dim,
        "dit_num_layers": config.a2a_dit_num_layers,
        "dit_num_heads": config.a2a_dit_num_heads,
        "dit_mlp_ratio": config.a2a_dit_mlp_ratio,
        "dit_dropout": config.a2a_dit_dropout,
        "history_horizon": config.a2a_history_horizon,
        "future_horizon": config.a2a_future_horizon,
        "action_dim": config.max_action_dim,
        "num_inference_steps": config.a2a_num_inference_steps,
        "random_initialization": False,
        "inputs": input_names,
        "input_dtypes": {
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
        },
    }
    contract_sha256 = hashlib.sha256(
        json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    metadata = {
        **contract,
        "contract_sha256": contract_sha256,
        "onnx": onnx_path.name,
        "onnx_sha256": sha256_file(onnx_path),
        "checkpoint_sha256": model_checkpoint_sha256,
        "data_contract_sha256": data_contract_sha256,
        "canonical_statistics_sha256": statistics_sha256,
        "verification": "onnx_checker+torch_ort_rtol1e-4_atol1e-4" if verify_export else "skipped",
    }
    (output / "a2a_export_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return onnx_path


if __name__ == "__main__":
    tyro.cli(export_a2a_action_head)
