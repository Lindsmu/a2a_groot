# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the latent A2A GR00T action head."""

from types import SimpleNamespace

from gr00t.configs.model.gr00t_n1d7_a2a import Gr00tN1d7A2AConfig
from gr00t.model.gr00t_n1d7_a2a.gr00t_n1d7_a2a import Gr00tN1d7A2AActionHead
import pytest
import torch
from transformers.feature_extraction_utils import BatchFeature


def _config(**overrides):
    values = dict(
        backbone_embedding_dim=32,
        max_action_dim=7,
        action_horizon=4,
        max_num_embodiments=3,
        a2a_history_horizon=4,
        a2a_future_horizon=4,
        a2a_latent_dim=16,
        a2a_encoder_channels=(8, 12, 16),
        a2a_encoder_kernel_size=5,
        a2a_flow_blocks=4,
        a2a_flow_mlp_ratio=2,
        a2a_decoder_res_blocks=4,
        a2a_num_inference_steps=1,
        a2a_ic_train_steps=1,
        a2a_history_noise_train_std=0.0,
        a2a_history_noise_infer_std=0.0,
        a2a_training_stage="joint",
        a2a_strict_paper_architecture=False,
        use_vlln=True,
        tune_projector=True,
        tune_diffusion_model=True,
        tune_vlln=True,
    )
    values.update(overrides)
    return Gr00tN1d7A2AConfig(**values)


def _strict_paper_overrides(**overrides):
    values = dict(
        action_horizon=8,
        a2a_history_horizon=8,
        a2a_future_horizon=8,
        a2a_latent_dim=512,
        a2a_encoder_channels=(128, 256, 512),
        a2a_encoder_kernel_size=5,
        a2a_flow_blocks=4,
        a2a_flow_mlp_ratio=4,
        a2a_decoder_res_blocks=4,
        a2a_strict_paper_architecture=True,
    )
    values.update(overrides)
    return values


def _backbone(config, batch_size=2):
    attention = torch.ones(batch_size, 6, dtype=torch.long)
    image = torch.zeros(batch_size, 6, dtype=torch.bool)
    image[:, :3] = True
    return BatchFeature(
        data={
            "backbone_features": torch.randn(batch_size, 6, config.backbone_embedding_dim),
            "backbone_attention_mask": attention,
            "image_mask": image,
        }
    )


def _inputs(config, batch_size=2):
    shape = (batch_size, config.a2a_future_horizon, config.max_action_dim)
    continuous = torch.zeros(shape)
    continuous[..., :6] = 1
    auxiliary = torch.zeros(shape)
    auxiliary[..., 6:] = 1
    return BatchFeature(
        data={
            "history_action_canonical": torch.randn(shape),
            "future_action_canonical": torch.randn(shape).clamp(-1, 1),
            "history_action_mask": continuous.clone(),
            "future_action_mask": torch.ones(shape),
            "continuous_action_mask": continuous,
            "auxiliary_action_mask": auxiliary,
            "binary_action_mask": torch.zeros(shape),
            "categorical_action_mask": torch.zeros(shape),
            "categorical_group_index": torch.full(shape, -1, dtype=torch.long),
            "embodiment_id": torch.zeros(batch_size, dtype=torch.long),
            "state": torch.randn(batch_size, 1, config.max_state_dim),
        }
    )


def test_forward_exposes_paper_losses_and_gradients():
    config = _config()
    head = Gr00tN1d7A2AActionHead(config).train()
    output = head(_backbone(config), _inputs(config))
    for key in ("loss", "loss_fm", "loss_ae", "loss_ic", "loss_aux"):
        assert output[key].ndim == 0
        assert torch.isfinite(output[key])
    output["loss"].backward()
    assert any(parameter.grad is not None for parameter in head.flow_net.parameters())
    assert any(parameter.grad is not None for parameter in head.trajectory_encoder.parameters())
    assert output["action_pred"].shape == (2, 4, 7)


def test_mlp_backend_remains_default_with_vector_latent():
    config = _config()
    head = Gr00tN1d7A2AActionHead(config).eval()
    output = head.get_action(_backbone(config), _inputs(config))

    assert config.a2a_flow_backbone == "mlp"
    assert output["z0"].shape == (2, 16)
    assert output["z1_inferred"].shape == (2, 16)

    # The MLP branch adds only an Identity summary adapter, so no DiT-only or
    # compatibility-shim parameter is serialized into an old-style checkpoint.
    state_keys = tuple(head.state_dict())
    assert not any("position_embedding" in key for key in state_keys)
    assert not any("latent_summary_projection" in key for key in state_keys)
    Gr00tN1d7A2AActionHead(config).load_state_dict(head.state_dict(), strict=True)


def test_dit_backend_trains_and_infers_with_temporal_latent_tokens():
    config = _config(
        a2a_flow_backbone="dit",
        a2a_dit_token_dim=16,
        a2a_dit_num_layers=2,
        a2a_dit_num_heads=4,
        a2a_dit_mlp_ratio=2,
        a2a_dit_dropout=0.0,
    )
    head = Gr00tN1d7A2AActionHead(config).train()
    backbone = _backbone(config)
    inputs = _inputs(config)
    output = head(backbone, inputs)
    output["loss"].backward()

    assert output["z0"].shape == (2, 4, 16)
    assert output["z1"].shape == (2, 4, 16)
    assert output["z1_inferred"].shape == (2, 4, 16)
    assert output["action_pred"].shape == (2, 4, 7)
    assert torch.isfinite(output["loss"])
    assert any(parameter.grad is not None for parameter in head.flow_net.parameters())
    assert any(parameter.grad is not None for parameter in head.trajectory_encoder.parameters())

    head.eval()
    first = head.get_action(backbone, inputs)["action_pred"]
    second = head.get_action(backbone, inputs)["action_pred"]
    torch.testing.assert_close(first, second)


def test_dit_backend_respects_partial_cold_start_history_mask():
    config = _config(
        a2a_flow_backbone="dit",
        a2a_dit_token_dim=16,
        a2a_dit_num_layers=2,
        a2a_dit_num_heads=4,
        a2a_dit_mlp_ratio=2,
    )
    head = Gr00tN1d7A2AActionHead(config).eval()
    inputs = _inputs(config)
    inputs["history_action_mask"][:, :3] = 0
    output = head.get_action(_backbone(config), inputs)

    torch.testing.assert_close(output["z0"][:, :3], torch.zeros_like(output["z0"][:, :3]))
    assert output["action_pred"].shape == (2, 4, 7)
    assert torch.isfinite(output["action_pred"]).all()


def test_dit_large_default_profile_smoke():
    """Exercise the documented 8-token, 256-wide, 8-layer comparison profile."""
    config = _config(
        action_horizon=8,
        a2a_history_horizon=8,
        a2a_future_horizon=8,
        a2a_flow_backbone="dit",
    )
    head = Gr00tN1d7A2AActionHead(config).eval()
    output = head.get_action(_backbone(config, batch_size=1), _inputs(config, batch_size=1))

    assert output["z0"].shape == (1, 8, 256)
    assert output["z1_inferred"].shape == (1, 8, 256)
    assert output["action_pred"].shape == (1, 8, 7)
    assert len(head.flow_net.blocks) == 8


def test_inference_is_deterministic_without_history_noise():
    config = _config()
    head = Gr00tN1d7A2AActionHead(config).eval()
    backbone = _backbone(config)
    inputs = _inputs(config)
    first = head.get_action(backbone, inputs)["action_pred"]
    second = head.get_action(backbone, inputs)["action_pred"]
    torch.testing.assert_close(first, second)


def test_source_latent_depends_on_executed_history():
    config = _config()
    head = Gr00tN1d7A2AActionHead(config).eval()
    backbone = _backbone(config)
    first_inputs = _inputs(config)
    second_inputs = BatchFeature(data=dict(first_inputs))
    second_inputs["history_action_canonical"] = (
        first_inputs["history_action_canonical"] + first_inputs["history_action_mask"] * 0.5
    )
    first = head.get_action(backbone, first_inputs)["z0"]
    second = head.get_action(backbone, second_inputs)["z0"]
    assert not torch.allclose(first, second)


def test_optional_current_state_condition_is_an_explicit_nonstrict_ablation():
    config = _config(a2a_include_current_state_condition=True)
    head = Gr00tN1d7A2AActionHead(config).eval()
    backbone = _backbone(config)
    first_inputs = _inputs(config)
    second_inputs = BatchFeature(data=dict(first_inputs))
    second_inputs["state"] = first_inputs["state"] + 0.5

    first = head.get_action(backbone, first_inputs)["action_pred"]
    second = head.get_action(backbone, second_inputs)["action_pred"]
    assert not torch.allclose(first, second)


def test_unassigned_future_channel_is_rejected():
    config = _config()
    head = Gr00tN1d7A2AActionHead(config).eval()
    inputs = _inputs(config)
    inputs["auxiliary_action_mask"][..., -1] = 0
    try:
        head.get_action(_backbone(config), inputs)
    except ValueError as error:
        assert "no generation head" in str(error)
    else:
        raise AssertionError("unassigned action channel should fail")


def test_inference_rejects_history_future_channel_mismatch():
    config = _config()
    head = Gr00tN1d7A2AActionHead(config).eval()
    inputs = _inputs(config)
    inputs["history_action_mask"][..., 0] = 0
    with pytest.raises(ValueError, match="same channels"):
        head.get_action(_backbone(config), inputs)


def test_strict_paper_config_accepts_exact_architecture():
    config = _config(**_strict_paper_overrides())
    assert tuple(config.a2a_encoder_channels) == (128, 256, 512)
    assert config.a2a_flow_mlp_ratio == 4
    assert config.a2a_decoder_res_blocks == 4


def test_joint_stage_rejects_frozen_random_trajectory_autoencoder():
    config = _config(tune_projector=False, a2a_training_stage="joint")

    with pytest.raises(ValueError, match="trajectory encoder/decoder"):
        Gr00tN1d7A2AActionHead(config)


def test_autoencoder_stage_rejects_unused_trainable_backbone():
    config = _config(a2a_training_stage="autoencoder", tune_llm=True)

    with pytest.raises(ValueError, match="bypasses the VLM backbone"):
        Gr00tN1d7A2AActionHead(config)


def test_checkpoint_load_overrides_saved_autoencoder_stage_for_joint_training():
    from gr00t.model.gr00t_n1d7_a2a.setup import Gr00tN1d7A2APipeline

    pipeline = object.__new__(Gr00tN1d7A2APipeline)
    pipeline.config = SimpleNamespace(model=_config(a2a_training_stage="joint"))

    overrides = pipeline._get_model_extra_kwargs()
    assert overrides["a2a_training_stage"] == "joint"
    assert overrides["a2a_num_inference_steps"] == 1
    assert overrides["a2a_history_noise_train_std"] == 0.0
    assert overrides["a2a_flow_backbone"] == "mlp"
    assert overrides["a2a_dit_num_layers"] == 8


def test_dit_config_rejects_invalid_head_width_and_strict_paper_claim():
    with pytest.raises(ValueError, match="divisible"):
        _config(
            a2a_flow_backbone="dit",
            a2a_dit_token_dim=18,
            a2a_dit_num_heads=4,
        )

    with pytest.raises(ValueError, match="flow_backbone"):
        _config(
            **_strict_paper_overrides(),
            a2a_flow_backbone="dit",
        )


def test_full_checkpoint_cannot_silently_switch_mlp_to_dit(monkeypatch):
    from gr00t.model.gr00t_n1d7_a2a import setup as a2a_setup

    saved_config = _config(a2a_flow_backbone="mlp")
    requested_config = _config(
        a2a_flow_backbone="dit",
        a2a_dit_token_dim=16,
        a2a_dit_num_layers=2,
        a2a_dit_num_heads=4,
        a2a_dit_mlp_ratio=2,
    )
    pipeline = object.__new__(a2a_setup.Gr00tN1d7A2APipeline)
    pipeline.config = SimpleNamespace(
        model=requested_config,
        training=SimpleNamespace(start_from_checkpoint="mlp-checkpoint"),
    )
    pipeline.transformers_loading_kwargs = {}
    monkeypatch.setattr(
        a2a_setup.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: saved_config,
    )

    with pytest.raises(ValueError, match="cannot change its MLP/DiT module graph"):
        pipeline._normalize_checkpoint_sources()


@pytest.mark.parametrize(
    ("override", "mismatch_name"),
    (
        ({"action_horizon": 7}, "action_horizon"),
        ({"a2a_history_horizon": 7}, "history_horizon"),
        ({"a2a_future_horizon": 7}, "future_horizon"),
        ({"a2a_latent_dim": 256}, "latent_dim"),
        ({"a2a_encoder_channels": (64, 256, 512)}, "encoder_channels"),
        ({"a2a_encoder_kernel_size": 3}, "encoder_kernel"),
        ({"a2a_flow_blocks": 3}, "flow_blocks"),
        ({"a2a_flow_mlp_ratio": 2}, "flow_mlp_ratio"),
        ({"a2a_decoder_res_blocks": 3}, "decoder_blocks"),
        ({"a2a_include_current_state_condition": True}, "include_current_state_condition"),
    ),
)
def test_strict_paper_config_rejects_architecture_drift(override, mismatch_name):
    with pytest.raises(ValueError, match=mismatch_name):
        _config(**_strict_paper_overrides(**override))
