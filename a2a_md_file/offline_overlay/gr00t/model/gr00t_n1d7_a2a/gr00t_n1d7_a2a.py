# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""GR00T N1.7 with vector-MLP or temporal-token DiT A2A flow matching."""

from __future__ import annotations

import logging
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, PreTrainedModel
from transformers.feature_extraction_utils import BatchFeature
import tree

from gr00t.configs.model.gr00t_n1d7_a2a import Gr00tN1d7A2AConfig
from gr00t.model.gr00t_n1d7.gr00t_n1d7 import get_backbone_cls
from gr00t.model.modules.a2a_latent import (
    ActionTrajectoryDecoder,
    ActionTrajectoryEncoder,
    ActionTrajectoryTokenDecoder,
    ActionTrajectoryTokenEncoder,
    LatentDiTFlowNet,
    LatentFlowNet,
    VLMConditionPooler,
    euler_integrate,
    masked_l1_loss,
)
from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificMLP


logger = logging.getLogger(__name__)


def _masked_bce_with_logits(
    logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    valid = mask.to(device=logits.device, dtype=torch.bool)
    safe_logits = torch.where(valid, logits, torch.zeros_like(logits))
    safe_target = torch.where(valid, target, torch.zeros_like(target))
    loss = F.binary_cross_entropy_with_logits(safe_logits, safe_target, reduction="none")
    weights = valid.to(dtype=loss.dtype)
    return torch.where(valid, loss, torch.zeros_like(loss)).sum() / weights.sum().clamp_min(1.0)


def _masked_categorical_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    group_index: torch.Tensor,
) -> torch.Tensor:
    """Cross entropy for variable-width categorical groups in a padded action tensor."""
    if group_index.shape != logits.shape:
        raise ValueError("categorical_group_index must match the padded action shape")
    losses = []
    for batch_index in range(logits.shape[0]):
        groups = torch.unique(group_index[batch_index][group_index[batch_index] >= 0])
        for group in groups.tolist():
            channel_mask = (group_index[batch_index] == group).any(dim=0)
            if not torch.any(channel_mask):
                continue
            group_mask = mask[batch_index, :, channel_mask].bool()
            step_mask = group_mask.any(dim=-1)
            if not torch.any(step_mask):
                continue
            if not torch.all(group_mask[step_mask]):
                raise ValueError("A categorical group must mask all of its class channels together")
            group_target = target[batch_index, :, channel_mask][step_mask]
            if not torch.allclose(
                group_target.sum(dim=-1),
                torch.ones(group_target.shape[0], device=target.device, dtype=target.dtype),
                atol=1e-5,
                rtol=0,
            ):
                raise ValueError("Categorical A2A targets must be one-hot on valid timesteps")
            class_index = group_target.argmax(dim=-1)
            group_logits = logits[batch_index, :, channel_mask][step_mask]
            losses.append(F.cross_entropy(group_logits, class_index))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


class Gr00tN1d7A2AActionHead(nn.Module):
    """Latent A2A generator with a paper-core MLP or temporal-token DiT backend."""

    supports_gradient_checkpointing = True

    def __init__(self, config: Gr00tN1d7A2AConfig):
        super().__init__()
        self.config = config
        self.action_dim = int(config.max_action_dim)
        self.action_horizon = int(config.a2a_future_horizon)
        self.latent_dim = int(config.a2a_latent_dim)
        self.flow_backbone = str(config.a2a_flow_backbone)
        self.num_inference_timesteps = int(config.a2a_num_inference_steps)
        self.training_stage = str(config.a2a_training_stage)
        if self.training_stage not in {"autoencoder", "flow_only", "joint"}:
            raise ValueError(f"Unsupported A2A training stage: {self.training_stage}")
        if self.training_stage in {"autoencoder", "joint"} and not config.tune_projector:
            raise ValueError(
                f"A2A {self.training_stage} stage requires tune_projector=True so the "
                "trajectory encoder/decoder are trainable; use flow_only only after "
                "loading a trained A2A checkpoint"
            )
        if self.training_stage == "autoencoder" and (config.tune_llm or config.tune_visual):
            raise ValueError(
                "A2A autoencoder stage bypasses the VLM backbone, so tune_llm and "
                "tune_visual must both be False"
            )
        if self.training_stage in {"flow_only", "joint"} and not config.tune_diffusion_model:
            raise ValueError(f"A2A {self.training_stage} stage requires tune_diffusion_model=True")
        if int(config.a2a_num_inference_steps) <= 0 or int(config.a2a_ic_train_steps) <= 0:
            raise ValueError("A2A Euler step counts must be positive")
        if config.a2a_history_noise_train_std < 0 or config.a2a_history_noise_infer_std < 0:
            raise ValueError("A2A history noise standard deviations must be non-negative")
        loss_weights = (
            config.a2a_lambda_fm,
            config.a2a_lambda_ae,
            config.a2a_lambda_ic,
            config.a2a_lambda_ic_action,
            config.a2a_lambda_aux,
        )
        if any(weight < 0 for weight in loss_weights):
            raise ValueError("A2A loss weights must be non-negative")
        # Revalidate here as callers may mutate a config after construction.
        config.validate_paper_architecture()
        self.mask_token = None  # Compatibility with the base pipeline's loader checks.

        self.vlln = (
            nn.LayerNorm(config.backbone_embedding_dim) if config.use_vlln else nn.Identity()
        )
        self.condition_pooler = VLMConditionPooler(
            input_dim=config.backbone_embedding_dim,
            condition_dim=self.latent_dim,
        )
        self.current_state_projection = None
        if config.a2a_include_current_state_condition:
            self.current_state_projection = CategorySpecificMLP(
                num_categories=config.max_num_embodiments,
                input_dim=config.max_state_dim,
                hidden_dim=self.latent_dim,
                output_dim=self.latent_dim,
            )
        if self.flow_backbone == "mlp":
            # Original paper-core path: pool the complete trajectory to one
            # latent vector and predict its velocity with four AdaLN MLPs.
            self.trajectory_encoder = ActionTrajectoryEncoder(
                action_dim=self.action_dim,
                latent_dim=self.latent_dim,
                num_embodiments=config.max_num_embodiments,
                conv_channels=tuple(config.a2a_encoder_channels),
                kernel_size=config.a2a_encoder_kernel_size,
            )
            self.trajectory_decoder = ActionTrajectoryDecoder(
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
                latent_dim=self.latent_dim,
                hidden_dim=self.latent_dim,
                num_embodiments=config.max_num_embodiments,
                num_blocks=config.a2a_decoder_res_blocks,
                expansion_factor=config.a2a_flow_mlp_ratio,
            )
            self.flow_net = LatentFlowNet(
                latent_dim=self.latent_dim,
                condition_dim=self.latent_dim,
                hidden_dim=self.latent_dim * config.a2a_flow_mlp_ratio,
                num_blocks=config.a2a_flow_blocks,
            )
            self.latent_summary_projection = nn.Identity()
        else:
            # Temporal-token ablation: Conv1d retains H tokens instead of
            # pooling them. Attention now has a genuine trajectory time axis.
            token_dim = int(config.a2a_dit_token_dim)
            self.trajectory_encoder = ActionTrajectoryTokenEncoder(
                action_dim=self.action_dim,
                trajectory_horizon=self.action_horizon,
                token_dim=token_dim,
                num_embodiments=config.max_num_embodiments,
                conv_channels=tuple(config.a2a_encoder_channels),
                kernel_size=config.a2a_encoder_kernel_size,
            )
            self.trajectory_decoder = ActionTrajectoryTokenDecoder(
                action_dim=self.action_dim,
                action_horizon=self.action_horizon,
                token_dim=token_dim,
                hidden_dim=self.latent_dim,
                num_embodiments=config.max_num_embodiments,
                num_blocks=config.a2a_decoder_res_blocks,
                expansion_factor=config.a2a_flow_mlp_ratio,
            )
            self.flow_net = LatentDiTFlowNet(
                token_dim=token_dim,
                num_tokens=self.action_horizon,
                condition_dim=self.latent_dim,
                num_layers=config.a2a_dit_num_layers,
                num_heads=config.a2a_dit_num_heads,
                mlp_ratio=config.a2a_dit_mlp_ratio,
                dropout=config.a2a_dit_dropout,
            )
            # Auxiliary action types still use the established 512-D interface.
            # Pooling happens only after the DiT has produced future tokens.
            self.latent_summary_projection = nn.Sequential(
                nn.LayerNorm(token_dim),
                nn.Linear(token_dim, self.latent_dim),
            )
        self.auxiliary_head = CategorySpecificMLP(
            num_categories=config.max_num_embodiments,
            input_dim=2 * self.latent_dim,
            hidden_dim=self.latent_dim,
            output_dim=self.action_horizon * self.action_dim,
        )
        self.set_trainable_parameters(
            config.tune_projector, config.tune_diffusion_model, config.tune_vlln
        )

    def set_trainable_parameters(
        self, tune_projector: bool, tune_flow: bool, tune_vlln: bool
    ) -> None:
        projector_modules = (
            self.condition_pooler,
            self.trajectory_encoder,
            self.trajectory_decoder,
            self.latent_summary_projection,
            self.auxiliary_head,
        )
        for module in projector_modules:
            module.requires_grad_(tune_projector)
        if self.current_state_projection is not None:
            self.current_state_projection.requires_grad_(tune_projector)
        self.flow_net.requires_grad_(tune_flow)
        self.vlln.requires_grad_(tune_vlln)
        if self.training_stage == "autoencoder":
            self.condition_pooler.requires_grad_(False)
            if self.current_state_projection is not None:
                self.current_state_projection.requires_grad_(False)
            self.auxiliary_head.requires_grad_(False)
            self.flow_net.requires_grad_(False)
            self.vlln.requires_grad_(False)
        elif self.training_stage == "flow_only":
            self.trajectory_encoder.requires_grad_(False)
            self.trajectory_decoder.requires_grad_(False)

    def set_frozen_modules_to_eval_mode(self) -> None:
        for module in (
            self.vlln,
            self.condition_pooler,
            self.trajectory_encoder,
            self.trajectory_decoder,
            self.latent_summary_projection,
            self.flow_net,
            self.auxiliary_head,
            self.current_state_projection,
        ):
            if module is not None and not any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                module.eval()

    def prepare_input(self, batch: dict) -> BatchFeature:
        return BatchFeature(data=batch)

    def _condition(self, backbone_output: BatchFeature, action_input: BatchFeature) -> torch.Tensor:
        features = self.vlln(backbone_output["backbone_features"])
        attention_mask = backbone_output.get("backbone_attention_mask")
        image_mask = backbone_output.get("image_mask")
        text_mask = None
        if image_mask is not None:
            if attention_mask is None:
                text_mask = ~image_mask.bool()
            else:
                text_mask = attention_mask.bool() & ~image_mask.bool()
        condition = self.condition_pooler(features, attention_mask, image_mask, text_mask)
        if self.current_state_projection is not None:
            state = action_input["state"]
            embodiment_id = action_input["embodiment_id"].long()
            state_condition = self.current_state_projection(state, embodiment_id).mean(dim=1)
            condition = condition + state_condition
        return condition

    def _masks(self, action_input: BatchFeature) -> tuple[torch.Tensor, ...]:
        masks = tuple(
            action_input[name].to(dtype=action_input["history_action_canonical"].dtype)
            for name in (
                "continuous_action_mask",
                "auxiliary_action_mask",
                "binary_action_mask",
                "categorical_action_mask",
            )
        )
        continuous, auxiliary, binary, categorical = masks
        for name, mask in zip(
            ("continuous", "auxiliary", "binary", "categorical"), masks, strict=True
        ):
            if not torch.all(torch.isfinite(mask)) or not torch.all((mask == 0) | (mask == 1)):
                raise ValueError(f"A2A {name} channel mask must contain only finite 0/1 values")
        covered = continuous + auxiliary + binary + categorical
        if torch.any(covered > 1.0 + 1e-6):
            raise ValueError("A2A channel masks overlap")
        future_mask = action_input["future_action_mask"].to(dtype=covered.dtype)
        if not torch.all(torch.isfinite(future_mask)) or not torch.all(
            (future_mask == 0) | (future_mask == 1)
        ):
            raise ValueError("A2A future_action_mask must contain only finite 0/1 values")
        if torch.any(future_mask.bool() & ~covered.bool()):
            raise ValueError("A2A future action contains channels with no generation head")
        if torch.any(covered.bool() & ~future_mask.bool()):
            raise ValueError("A2A generation masks must be zero outside future_action_mask padding")
        continuous_per_sample = continuous.flatten(1).sum(dim=-1)
        if torch.any(continuous_per_sample <= 0):
            bad = torch.nonzero(continuous_per_sample <= 0).flatten().tolist()
            raise ValueError(
                "Latent A2A requires continuous mapped channels for every sample; "
                f"missing batch indices={bad}"
            )
        return masks

    def _validate_action_input(self, action_input: BatchFeature, require_future: bool) -> None:
        embodiment_id = action_input["embodiment_id"]
        if embodiment_id.ndim != 1:
            raise ValueError("embodiment_id must have shape [B]")
        batch_size = embodiment_id.shape[0]
        if torch.any(embodiment_id < 0) or torch.any(
            embodiment_id >= self.config.max_num_embodiments
        ):
            raise ValueError("embodiment_id is outside the configured category range")
        history_shape = (
            batch_size,
            int(self.config.a2a_history_horizon),
            self.action_dim,
        )
        future_shape = (batch_size, self.action_horizon, self.action_dim)
        for name in ("history_action_canonical", "history_action_mask"):
            if tuple(action_input[name].shape) != history_shape:
                raise ValueError(
                    f"{name} must have shape {history_shape}, got {tuple(action_input[name].shape)}"
                )
        future_names = (
            "future_action_mask",
            "continuous_action_mask",
            "auxiliary_action_mask",
            "binary_action_mask",
            "categorical_action_mask",
        )
        if require_future:
            future_names = ("future_action_canonical", *future_names)
        for name in future_names:
            if tuple(action_input[name].shape) != future_shape:
                raise ValueError(
                    f"{name} must have shape {future_shape}, got {tuple(action_input[name].shape)}"
                )
        history_mask = action_input["history_action_mask"]
        if not torch.all(torch.isfinite(history_mask)) or not torch.all(
            (history_mask == 0) | (history_mask == 1)
        ):
            raise ValueError("history_action_mask must contain only finite 0/1 values")
        valid_history = history_mask.bool()
        history = action_input["history_action_canonical"]
        if torch.any(valid_history & ~torch.isfinite(history)):
            raise ValueError("history_action_canonical is non-finite under its active mask")
        if torch.any(valid_history.flatten(1).sum(dim=-1) <= 0):
            raise ValueError("Every A2A sample must contain at least one real history value")
        if require_future:
            valid_future = action_input["future_action_mask"].bool()
            future = action_input["future_action_canonical"]
            if torch.any(valid_future & ~torch.isfinite(future)):
                raise ValueError("future_action_canonical is non-finite under its active mask")
        if self.current_state_projection is not None:
            expected_state_shape = (batch_size, 1, int(self.config.max_state_dim))
            state = action_input.get("state")
            if state is None or tuple(state.shape) != expected_state_shape:
                raise ValueError(
                    f"Current-state A2A ablation requires state with shape {expected_state_shape}"
                )
            if not torch.all(torch.isfinite(state)):
                raise ValueError("Current-state A2A condition contains NaN or infinity")

    @staticmethod
    def _validate_continuous_channel_support(
        history_mask: torch.Tensor, continuous_mask: torch.Tensor
    ) -> None:
        history_support = history_mask.bool().any(dim=1)
        continuous_support = continuous_mask.bool().any(dim=1)
        if not torch.equal(history_support, continuous_support):
            raise ValueError(
                "A2A history and future continuous masks must describe the same channels"
            )

    def _auxiliary_logits(
        self, condition: torch.Tensor, latent: torch.Tensor, embodiment_id: torch.Tensor
    ) -> torch.Tensor:
        if latent.ndim == 2:
            latent_summary = latent
        elif latent.ndim == 3 and self.flow_backbone == "dit":
            latent_summary = self.latent_summary_projection(latent.mean(dim=1))
        else:
            raise ValueError(f"Unexpected A2A latent shape: {tuple(latent.shape)}")
        features = torch.cat((condition, latent_summary), dim=-1).unsqueeze(1)
        output = self.auxiliary_head(features, embodiment_id).squeeze(1)
        return output.reshape(-1, self.action_horizon, self.action_dim)

    def _flow_velocity(
        self,
        latent: torch.Tensor,
        timesteps: torch.Tensor,
        condition: torch.Tensor,
        history_time_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Dispatch to the selected velocity network without changing FM semantics."""
        if self.flow_backbone == "dit":
            return self.flow_net(
                latent,
                timesteps,
                condition,
                token_mask=history_time_mask,
            )
        return self.flow_net(latent, timesteps, condition)

    @staticmethod
    def _latent_pair_distance(z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """Report one Euclidean history/future distance per batch item."""
        return (z1 - z0).flatten(1).norm(dim=-1).mean()

    def _decode_and_merge(
        self,
        latent: torch.Tensor,
        condition: torch.Tensor,
        embodiment_id: torch.Tensor,
        masks: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        continuous_mask, auxiliary_mask, binary_mask, categorical_mask = masks
        continuous = self.trajectory_decoder(latent, embodiment_id, continuous_mask)
        auxiliary_logits = self._auxiliary_logits(condition, latent, embodiment_id)
        regression = torch.tanh(auxiliary_logits)
        discrete = torch.sigmoid(auxiliary_logits)
        action = (
            continuous * continuous_mask
            + regression * auxiliary_mask
            + discrete * (binary_mask + categorical_mask)
        )
        return action, continuous, auxiliary_logits

    def forward(self, backbone_output: BatchFeature, action_input: BatchFeature) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        self._validate_action_input(action_input, require_future=True)
        embodiment_id = action_input["embodiment_id"].long()
        history = action_input["history_action_canonical"]
        future = action_input["future_action_canonical"]
        history_mask = action_input["history_action_mask"]
        masks = self._masks(action_input)
        continuous_mask, auxiliary_mask, binary_mask, categorical_mask = masks
        self._validate_continuous_channel_support(history_mask, continuous_mask)

        noise_std = float(self.config.a2a_history_noise_train_std)
        if self.training and noise_std > 0:
            history = history + noise_std * torch.randn_like(history) * history_mask

        # The very same trajectory encoder maps executed history and future
        # demonstration into z0/z1, as required by latent A2A.
        z1 = self.trajectory_encoder(future, embodiment_id, continuous_mask)
        future_reconstruction = self.trajectory_decoder(z1, embodiment_id, continuous_mask)
        loss_ae = masked_l1_loss(future_reconstruction, future, continuous_mask)
        if self.training_stage == "autoencoder":
            z0 = self.trajectory_encoder(history, embodiment_id, history_mask)
            zero = loss_ae * 0.0
            return BatchFeature(
                data={
                    "loss": loss_ae,
                    "loss_fm": zero,
                    "loss_ae": loss_ae,
                    "loss_ic": zero,
                    "loss_ic_latent": zero,
                    "loss_ic_action": zero,
                    "loss_aux": zero,
                    "loss_aux_regression": zero,
                    "loss_aux_binary": zero,
                    "loss_aux_categorical": zero,
                    "action_pred": future_reconstruction,
                    "z0": z0,
                    "z1": z1,
                    "z1_inferred": z1,
                    "latent_std": torch.cat((z0, z1), dim=0).std(unbiased=False),
                    "latent_pair_distance": self._latent_pair_distance(z0, z1),
                }
            )

        condition = self._condition(backbone_output, action_input)
        z0 = self.trajectory_encoder(history, embodiment_id, history_mask)
        tau = torch.rand(z0.shape[0], device=z0.device, dtype=torch.float32)
        path_tau = tau.to(dtype=z0.dtype)
        # Broadcast one sampled flow time over either [D] (MLP) or [H, D]
        # (DiT). The linear OT path and target velocity are otherwise identical.
        path_tau = path_tau.reshape((z0.shape[0],) + (1,) * (z0.ndim - 1))
        z_tau = (1.0 - path_tau) * z0 + path_tau * z1
        target_velocity = z1 - z0
        history_time_mask = history_mask.bool().any(dim=-1)
        predicted_velocity = self._flow_velocity(
            z_tau,
            tau,
            condition,
            history_time_mask,
        )
        loss_fm = F.mse_loss(predicted_velocity, target_velocity)

        def velocity_field(latent, timesteps, flow_condition):
            # Capture the real-history validity pattern for every differentiable
            # Euler step. Invalid cold-start queries still generate future tokens.
            return self._flow_velocity(
                latent,
                timesteps,
                flow_condition,
                history_time_mask,
            )

        z1_inferred = euler_integrate(
            velocity_field,
            z0,
            condition,
            num_steps=int(self.config.a2a_ic_train_steps),
        )
        action_inferred, _, auxiliary_logits = self._decode_and_merge(
            z1_inferred, condition, embodiment_id, masks
        )
        loss_ic_latent = F.l1_loss(z1_inferred, z1)
        loss_ic_action = masked_l1_loss(action_inferred, future, continuous_mask)
        loss_ic = loss_ic_latent + self.config.a2a_lambda_ic_action * loss_ic_action

        loss_aux_regression = masked_l1_loss(torch.tanh(auxiliary_logits), future, auxiliary_mask)
        loss_aux_binary = _masked_bce_with_logits(auxiliary_logits, future, binary_mask)
        categorical_group_index = action_input.get("categorical_group_index")
        if categorical_group_index is None:
            if categorical_mask.sum() > 0:
                raise ValueError("categorical_group_index is required for categorical actions")
            categorical_group_index = torch.full_like(categorical_mask, -1, dtype=torch.long)
        loss_aux_categorical = _masked_categorical_cross_entropy(
            auxiliary_logits,
            future,
            categorical_mask,
            categorical_group_index.long(),
        )
        loss_aux = loss_aux_regression + loss_aux_binary + loss_aux_categorical
        full_joint_loss = (
            self.config.a2a_lambda_fm * loss_fm
            + self.config.a2a_lambda_ae * loss_ae
            + self.config.a2a_lambda_ic * loss_ic
            + self.config.a2a_lambda_aux * loss_aux
        )
        if self.training_stage == "autoencoder":
            loss = loss_ae
        elif self.training_stage == "flow_only":
            loss = (
                self.config.a2a_lambda_fm * loss_fm
                + self.config.a2a_lambda_ic * loss_ic
                + self.config.a2a_lambda_aux * loss_aux
            )
        else:
            loss = full_joint_loss

        return BatchFeature(
            data={
                "loss": loss,
                "loss_fm": loss_fm,
                "loss_ae": loss_ae,
                "loss_ic": loss_ic,
                "loss_ic_latent": loss_ic_latent,
                "loss_ic_action": loss_ic_action,
                "loss_aux": loss_aux,
                "loss_aux_regression": loss_aux_regression,
                "loss_aux_binary": loss_aux_binary,
                "loss_aux_categorical": loss_aux_categorical,
                "action_pred": action_inferred,
                "z0": z0,
                "z1": z1,
                "z1_inferred": z1_inferred,
                "latent_std": torch.cat((z0, z1), dim=0).std(unbiased=False),
                "latent_pair_distance": self._latent_pair_distance(z0, z1),
            }
        )

    @torch.no_grad()
    def get_action(
        self,
        backbone_output: BatchFeature,
        action_input: BatchFeature,
        options: dict[str, Any] | None = None,
    ) -> BatchFeature:
        self.set_frozen_modules_to_eval_mode()
        self._validate_action_input(action_input, require_future=False)
        condition = self._condition(backbone_output, action_input)
        embodiment_id = action_input["embodiment_id"].long()
        history = action_input["history_action_canonical"]
        history_mask = action_input["history_action_mask"]
        masks = self._masks(action_input)
        self._validate_continuous_channel_support(history_mask, masks[0])
        noise_std = float(self.config.a2a_history_noise_infer_std)
        if noise_std > 0:
            history = history + noise_std * torch.randn_like(history) * history_mask
        z0 = self.trajectory_encoder(history, embodiment_id, history_mask)
        history_time_mask = history_mask.bool().any(dim=-1)
        num_steps = self.num_inference_timesteps
        if options is not None and "num_inference_steps" in options:
            num_steps = int(options["num_inference_steps"])

        def velocity_field(latent, timesteps, flow_condition):
            return self._flow_velocity(
                latent,
                timesteps,
                flow_condition,
                history_time_mask,
            )

        z1_inferred = euler_integrate(velocity_field, z0, condition, num_steps=num_steps)
        action, _, _ = self._decode_and_merge(z1_inferred, condition, embodiment_id, masks)
        return BatchFeature(
            data={
                "action_pred": action,
                "z0": z0,
                "z1_inferred": z1_inferred,
                "num_inference_steps": num_steps,
            }
        )


class Gr00tN1d7A2A(PreTrainedModel):
    """GR00T VLM backbone plus the latent A2A action head."""

    config_class = Gr00tN1d7A2AConfig
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: Gr00tN1d7A2AConfig,
        transformers_loading_kwargs: dict = {"trust_remote_code": True},
    ):
        super().__init__(config)
        self.config = config
        backbone_cls = get_backbone_cls(config)
        self.backbone = backbone_cls(
            model_name=config.model_name,
            tune_llm=config.tune_llm,
            tune_visual=config.tune_visual,
            select_layer=config.select_layer,
            reproject_vision=config.reproject_vision,
            use_flash_attention=config.use_flash_attention,
            load_bf16=config.load_bf16,
            tune_top_llm_layers=config.tune_top_llm_layers,
            trainable_params_fp32=config.backbone_trainable_params_fp32,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )
        self.action_head = Gr00tN1d7A2AActionHead(config)
        from gr00t.model.gr00t_n1d7.processing_gr00t_n1d7 import Gr00tN1d7DataCollator

        self.collator = Gr00tN1d7DataCollator(
            model_name=config.model_name,
            model_type=config.backbone_model_type,
            transformers_loading_kwargs=transformers_loading_kwargs,
        )

    def prepare_input(self, inputs: dict) -> tuple[BatchFeature, BatchFeature]:
        inputs = dict(inputs)
        if "vlm_content" in inputs:
            contents = inputs.pop("vlm_content")
            if not isinstance(contents, list):
                contents = [contents]
            inputs.update(
                self.collator([{"vlm_content": content} for content in contents])["inputs"]
            )
        backbone_inputs = self.backbone.prepare_input(inputs)
        action_inputs = self.action_head.prepare_input(inputs)

        def move(value):
            if torch.is_floating_point(value):
                return value.to(self.device, dtype=self.dtype)
            return value.to(self.device)

        return (
            tree.map_structure(move, backbone_inputs),
            tree.map_structure(move, action_inputs),
        )

    def forward(self, inputs: dict) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        if self.action_head.training_stage == "autoencoder":
            return self.action_head(None, action_inputs)
        return self.action_head(self.backbone(backbone_inputs), action_inputs)

    @torch.no_grad()
    def get_action(self, inputs: dict, options: dict[str, Any] | None = None) -> BatchFeature:
        backbone_inputs, action_inputs = self.prepare_input(inputs)
        return self.action_head.get_action(self.backbone(backbone_inputs), action_inputs, options)

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype


AutoConfig.register("Gr00tN1d7A2A", Gr00tN1d7A2AConfig)
AutoModel.register(Gr00tN1d7A2AConfig, Gr00tN1d7A2A)
