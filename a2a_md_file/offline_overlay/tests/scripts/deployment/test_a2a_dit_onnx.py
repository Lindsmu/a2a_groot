# SPDX-License-Identifier: Apache-2.0

"""CPU ONNX oracle for the temporal-token A2A DiT action head."""

from __future__ import annotations

from gr00t.configs.model.gr00t_n1d7_a2a import Gr00tN1d7A2AConfig
from gr00t.model.gr00t_n1d7_a2a.gr00t_n1d7_a2a import Gr00tN1d7A2AActionHead
import numpy as np
import pytest
from scripts.deployment.a2a_trt_model_forward import A2A_ENGINE_INPUT_NAMES
from scripts.deployment.export_onnx_n1d7_a2a import A2AActionHeadExportWrapper, _verify_a2a_onnx
import torch


def _dit_config() -> Gr00tN1d7A2AConfig:
    return Gr00tN1d7A2AConfig(
        backbone_embedding_dim=16,
        max_action_dim=4,
        action_horizon=4,
        max_num_embodiments=2,
        a2a_history_horizon=4,
        a2a_future_horizon=4,
        a2a_latent_dim=16,
        a2a_encoder_channels=(8, 12, 16),
        a2a_encoder_kernel_size=5,
        a2a_decoder_res_blocks=2,
        a2a_flow_mlp_ratio=2,
        a2a_flow_backbone="dit",
        a2a_dit_token_dim=16,
        a2a_dit_num_layers=1,
        a2a_dit_num_heads=4,
        a2a_dit_mlp_ratio=2,
        a2a_dit_dropout=0.0,
        a2a_strict_paper_architecture=False,
        a2a_history_noise_infer_std=0.0,
        use_vlln=True,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
    )


def test_temporal_dit_action_head_onnx_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    torch.manual_seed(31)
    config = _dit_config()
    wrapper = A2AActionHeadExportWrapper(Gr00tN1d7A2AActionHead(config).float().eval())
    batch_size = 1
    sequence_length = 5
    history_shape = (batch_size, config.a2a_history_horizon, config.max_action_dim)
    future_shape = (batch_size, config.a2a_future_horizon, config.max_action_dim)
    continuous = torch.ones(future_shape)
    zeros = torch.zeros(future_shape)
    inputs = (
        torch.randn(batch_size, sequence_length, config.backbone_embedding_dim),
        torch.ones(batch_size, sequence_length, dtype=torch.long),
        torch.tensor([[1, 1, 0, 0, 0]], dtype=torch.bool),
        torch.randn(history_shape),
        torch.ones(history_shape),
        continuous,
        zeros,
        zeros,
        zeros,
        torch.zeros(batch_size, dtype=torch.long),
    )
    output_path = tmp_path / "a2a_dit_action_head.onnx"

    torch.onnx.export(
        wrapper,
        inputs,
        output_path,
        input_names=list(A2A_ENGINE_INPUT_NAMES),
        output_names=["action_pred"],
        dynamic_axes={
            **{name: {0: "batch_size"} for name in A2A_ENGINE_INPUT_NAMES},
            "backbone_features": {0: "batch_size", 1: "vl_sequence"},
            "backbone_attention_mask": {0: "batch_size", 1: "vl_sequence"},
            "image_mask": {0: "batch_size", 1: "vl_sequence"},
            "action_pred": {0: "batch_size"},
        },
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    _verify_a2a_onnx(output_path, wrapper, inputs, list(A2A_ENGINE_INPUT_NAMES))

    # The production export declares dynamic batch and VLM sequence axes. Use a
    # second shape to prove that the traced DiT graph did not bake in batch=1.
    dynamic_batch = 2
    dynamic_sequence = 7
    dynamic_history_shape = (
        dynamic_batch,
        config.a2a_history_horizon,
        config.max_action_dim,
    )
    dynamic_future_shape = (
        dynamic_batch,
        config.a2a_future_horizon,
        config.max_action_dim,
    )
    dynamic_continuous = torch.ones(dynamic_future_shape)
    dynamic_zeros = torch.zeros(dynamic_future_shape)
    dynamic_inputs = (
        torch.randn(dynamic_batch, dynamic_sequence, config.backbone_embedding_dim),
        torch.ones(dynamic_batch, dynamic_sequence, dtype=torch.long),
        torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0]],
            dtype=torch.bool,
        ),
        torch.randn(dynamic_history_shape),
        torch.ones(dynamic_history_shape),
        dynamic_continuous,
        dynamic_zeros,
        dynamic_zeros,
        dynamic_zeros,
        torch.tensor([0, 1], dtype=torch.long),
    )
    with torch.inference_mode():
        expected = wrapper(*dynamic_inputs).numpy()
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    actual = session.run(
        ["action_pred"],
        {
            name: value.numpy()
            for name, value in zip(A2A_ENGINE_INPUT_NAMES, dynamic_inputs, strict=True)
        },
    )[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
