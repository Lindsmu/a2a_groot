# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Building blocks for latent action-to-action (A2A) flow matching.

The modules in this file deliberately do not depend on a particular GR00T action
head.  They provide the trajectory autoencoder, VLM condition pooling, latent
velocity field, and differentiable ODE integration needed by an A2A head while
remaining small enough to unit test independently.
"""

from __future__ import annotations

from collections.abc import Callable
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from gr00t.model.modules.embodiment_conditioned_mlp import CategorySpecificLinear


def _trajectory_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    """Return a boolean mask broadcast to ``[batch, time, dimension]``."""
    if mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(-1)
    try:
        return torch.broadcast_to(
            mask.to(device=reference.device, dtype=torch.bool), reference.shape
        )
    except RuntimeError as error:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} is not broadcastable to "
            f"trajectory shape {tuple(reference.shape)}"
        ) from error


def _token_mask(mask: Tensor | None, reference: Tensor) -> Tensor:
    """Return a boolean token mask for a ``[batch, sequence, feature]`` tensor."""
    shape = reference.shape[:2]
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=reference.device)
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim != 2:
        raise ValueError(f"expected a [batch, sequence] mask, got shape {tuple(mask.shape)}")
    try:
        return torch.broadcast_to(mask.to(device=reference.device, dtype=torch.bool), shape)
    except RuntimeError as error:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} is not broadcastable to token shape {shape}"
        ) from error


def _masked_token_mean(hidden_states: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    masked_states = torch.where(mask.unsqueeze(-1), hidden_states, torch.zeros_like(hidden_states))
    total = masked_states.sum(dim=1)
    count = weights.sum(dim=1).clamp_min(1.0)
    return total / count


def masked_l1_loss(
    prediction: Tensor,
    target: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute elementwise L1 averaged only over valid entries.

    An all-false mask produces a differentiable zero instead of a NaN.  Masks
    may omit trailing singleton dimensions as long as they are broadcastable to
    the prediction.
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction and target must have the same shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if mask is None:
        return (prediction - target).abs().mean()
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    try:
        valid = torch.broadcast_to(
            mask.to(device=prediction.device, dtype=torch.bool), prediction.shape
        )
    except RuntimeError as runtime_error:
        raise ValueError(
            f"mask with shape {tuple(mask.shape)} is not broadcastable to loss shape "
            f"{tuple(prediction.shape)}"
        ) from runtime_error
    difference = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    weights = valid.to(dtype=prediction.dtype)
    return difference.abs().sum() / weights.sum().clamp_min(1.0)


class SinusoidalTimeEmbedding(nn.Module):
    """Continuous sinusoidal embedding for flow times in ``[0, 1]``."""

    def __init__(self, embedding_dim: int, max_period: float = 10_000.0):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive")
        if max_period <= 0:
            raise ValueError("max_period must be positive")
        self.embedding_dim = embedding_dim
        self.max_period = max_period

    def forward(self, timesteps: Tensor) -> Tensor:
        if timesteps.ndim == 2 and timesteps.shape[-1] == 1:
            timesteps = timesteps.squeeze(-1)
        if timesteps.ndim != 1:
            raise ValueError(
                f"expected one continuous timestep per batch item, got {tuple(timesteps.shape)}"
            )

        half_dim = self.embedding_dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(-1)
        exponent = (
            -math.log(self.max_period)
            * torch.arange(half_dim, device=timesteps.device, dtype=torch.float32)
            / max(half_dim, 1)
        )
        angles = timesteps.float().unsqueeze(-1) * exponent.exp().unsqueeze(0)
        embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if self.embedding_dim % 2:
            embedding = F.pad(embedding, (0, 1))
        return embedding


class ActionTrajectoryEncoder(nn.Module):
    """Encode a masked action/proprioception trajectory into one latent vector.

    The same encoder instance is intended to encode both the executed history
    ``z0`` and future demonstration trajectory ``z1``.  A category-specific
    input projection accommodates different embodiment semantics while the
    three temporal convolutions and latent projection are shared.
    """

    def __init__(
        self,
        action_dim: int,
        latent_dim: int = 512,
        num_embodiments: int = 1,
        conv_channels: tuple[int, int, int] = (128, 256, 512),
        kernel_size: int = 5,
    ):
        super().__init__()
        if action_dim <= 0 or latent_dim <= 0 or num_embodiments <= 0:
            raise ValueError("action_dim, latent_dim, and num_embodiments must be positive")
        if len(conv_channels) != 3 or any(channel <= 0 for channel in conv_channels):
            raise ValueError("conv_channels must contain exactly three positive channel sizes")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.num_embodiments = num_embodiments
        self.input_adapter = CategorySpecificLinear(num_embodiments, action_dim, conv_channels[0])
        in_channels = (conv_channels[0], *conv_channels[:-1])
        padding = kernel_size // 2
        self.convolutions = nn.ModuleList(
            nn.Conv1d(input_channel, output_channel, kernel_size, padding=padding)
            for input_channel, output_channel in zip(in_channels, conv_channels, strict=True)
        )
        # Per-token normalization does not let synthetic/masked timesteps alter
        # the statistics of valid timesteps (unlike a temporal GroupNorm).
        self.normalizations = nn.ModuleList(nn.LayerNorm(channel) for channel in conv_channels)
        self.latent_projection = nn.Linear(conv_channels[-1], latent_dim)

    def forward(
        self,
        trajectory: Tensor,
        embodiment_ids: Tensor,
        trajectory_mask: Tensor | None = None,
    ) -> Tensor:
        if trajectory.ndim != 3 or trajectory.shape[-1] != self.action_dim:
            raise ValueError(
                f"expected trajectory [batch, time, {self.action_dim}], got "
                f"{tuple(trajectory.shape)}"
            )
        batch_size = trajectory.shape[0]
        if embodiment_ids.shape != (batch_size,):
            raise ValueError(
                f"expected embodiment_ids shape ({batch_size},), got {tuple(embodiment_ids.shape)}"
            )

        element_mask = _trajectory_mask(trajectory_mask, trajectory)
        time_mask = element_mask.any(dim=-1)
        masked_trajectory = torch.where(element_mask, trajectory, torch.zeros_like(trajectory))

        hidden_states = self.input_adapter(masked_trajectory, embodiment_ids)
        time_weights = time_mask.to(dtype=hidden_states.dtype).unsqueeze(1)
        hidden_states = hidden_states.transpose(1, 2) * time_weights
        for convolution, normalization in zip(self.convolutions, self.normalizations, strict=True):
            hidden_states = convolution(hidden_states).transpose(1, 2)
            hidden_states = normalization(hidden_states).transpose(1, 2)
            hidden_states = F.silu(hidden_states) * time_weights

        pooled = hidden_states.sum(dim=-1) / time_weights.sum(dim=-1).clamp_min(1.0)
        latent = self.latent_projection(pooled)
        has_valid_token = time_mask.any(dim=-1, keepdim=True)
        return latent * has_valid_token.to(dtype=latent.dtype)


class ActionTrajectoryTokenEncoder(nn.Module):
    """Encode a trajectory into one latent token per relative action timestep.

    This is the sequence-latent alternative to :class:`ActionTrajectoryEncoder`.
    The original paper-style MLP backend pools the three Conv1d outputs into one
    vector.  A Transformer needs more than one token for temporal attention to
    be useful, so this encoder keeps the time axis and projects every timestep
    to ``token_dim``.  History and future must still share this *same* encoder.

    Masked values are replaced by zero before the category adapter and are
    zeroed again after every temporal convolution.  Consequently, synthetic
    cold-start padding cannot leak into a valid token through a convolution.
    """

    def __init__(
        self,
        action_dim: int,
        trajectory_horizon: int,
        token_dim: int,
        num_embodiments: int = 1,
        conv_channels: tuple[int, int, int] = (128, 256, 512),
        kernel_size: int = 5,
    ):
        super().__init__()
        if min(action_dim, trajectory_horizon, token_dim, num_embodiments) <= 0:
            raise ValueError(
                "action_dim, trajectory_horizon, token_dim, and num_embodiments must be positive"
            )
        if len(conv_channels) != 3 or any(channel <= 0 for channel in conv_channels):
            raise ValueError("conv_channels must contain exactly three positive channel sizes")
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.action_dim = action_dim
        self.trajectory_horizon = trajectory_horizon
        self.token_dim = token_dim
        self.num_embodiments = num_embodiments
        self.input_adapter = CategorySpecificLinear(num_embodiments, action_dim, conv_channels[0])
        in_channels = (conv_channels[0], *conv_channels[:-1])
        padding = kernel_size // 2
        self.convolutions = nn.ModuleList(
            nn.Conv1d(input_channel, output_channel, kernel_size, padding=padding)
            for input_channel, output_channel in zip(in_channels, conv_channels, strict=True)
        )
        self.normalizations = nn.ModuleList(nn.LayerNorm(channel) for channel in conv_channels)
        self.token_projection = nn.Linear(conv_channels[-1], token_dim)

    def forward(
        self,
        trajectory: Tensor,
        embodiment_ids: Tensor,
        trajectory_mask: Tensor | None = None,
    ) -> Tensor:
        if trajectory.ndim != 3:
            raise ValueError(
                f"expected trajectory [batch, {self.trajectory_horizon}, {self.action_dim}], "
                f"got {tuple(trajectory.shape)}"
            )
        expected_shape = (trajectory.shape[0], self.trajectory_horizon, self.action_dim)
        if tuple(trajectory.shape) != expected_shape:
            raise ValueError(
                f"expected trajectory [batch, {self.trajectory_horizon}, {self.action_dim}], "
                f"got {tuple(trajectory.shape)}"
            )
        batch_size = trajectory.shape[0]
        if embodiment_ids.shape != (batch_size,):
            raise ValueError(
                f"expected embodiment_ids shape ({batch_size},), got {tuple(embodiment_ids.shape)}"
            )

        element_mask = _trajectory_mask(trajectory_mask, trajectory)
        time_mask = element_mask.any(dim=-1)
        masked_trajectory = torch.where(element_mask, trajectory, torch.zeros_like(trajectory))

        hidden_states = self.input_adapter(masked_trajectory, embodiment_ids)
        time_weights = time_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
        hidden_states = hidden_states * time_weights
        for convolution, normalization in zip(self.convolutions, self.normalizations, strict=True):
            hidden_states = convolution(hidden_states.transpose(1, 2)).transpose(1, 2)
            hidden_states = F.silu(normalization(hidden_states)) * time_weights

        tokens = self.token_projection(hidden_states) * time_weights
        return tokens


class ResidualMLPBlock(nn.Module):
    """Pre-normalized residual MLP block used by the A2A decoder."""

    def __init__(self, hidden_dim: int, expansion_factor: int = 4):
        super().__init__()
        if hidden_dim <= 0 or expansion_factor <= 0:
            raise ValueError("hidden_dim and expansion_factor must be positive")
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion_factor),
            nn.SiLU(),
            nn.Linear(hidden_dim * expansion_factor, hidden_dim),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return hidden_states + self.mlp(self.norm(hidden_states))


class ActionTrajectoryDecoder(nn.Module):
    """Decode one latent into a future action trajectory with four residual MLPs."""

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        latent_dim: int = 512,
        hidden_dim: int = 512,
        num_embodiments: int = 1,
        num_blocks: int = 4,
        expansion_factor: int = 4,
    ):
        super().__init__()
        if min(action_dim, action_horizon, latent_dim, hidden_dim, num_embodiments) <= 0:
            raise ValueError("decoder dimensions and num_embodiments must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, expansion_factor) for _ in range(num_blocks)
        )
        self.output_projection = CategorySpecificLinear(
            num_embodiments, hidden_dim, action_horizon * action_dim
        )

    def forward(
        self,
        latent: Tensor,
        embodiment_ids: Tensor,
        trajectory_mask: Tensor | None = None,
    ) -> Tensor:
        if latent.ndim != 2:
            raise ValueError(f"expected latent [batch, dimension], got {tuple(latent.shape)}")
        batch_size = latent.shape[0]
        if embodiment_ids.shape != (batch_size,):
            raise ValueError(
                f"expected embodiment_ids shape ({batch_size},), got {tuple(embodiment_ids.shape)}"
            )

        hidden_states = self.latent_projection(latent)
        for block in self.blocks:
            hidden_states = block(hidden_states)
        trajectory = self.output_projection(hidden_states.unsqueeze(1), embodiment_ids)
        trajectory = trajectory.reshape(batch_size, self.action_horizon, self.action_dim)
        if trajectory_mask is not None:
            trajectory = trajectory * _trajectory_mask(trajectory_mask, trajectory).to(
                dtype=trajectory.dtype
            )
        return trajectory


class ActionTrajectoryTokenDecoder(nn.Module):
    """Decode a temporal latent-token sequence into one future action chunk.

    The token sequence is flattened only at the decoder boundary.  Keeping the
    four residual MLP blocks global preserves the existing decoder's ability to
    model correlations across the complete action chunk, while the FlowNet can
    operate on explicit temporal tokens before this point.
    """

    def __init__(
        self,
        action_dim: int,
        action_horizon: int,
        token_dim: int,
        hidden_dim: int = 512,
        num_embodiments: int = 1,
        num_blocks: int = 4,
        expansion_factor: int = 4,
    ):
        super().__init__()
        if min(action_dim, action_horizon, token_dim, hidden_dim, num_embodiments) <= 0:
            raise ValueError("decoder dimensions and num_embodiments must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.token_dim = token_dim
        self.latent_projection = nn.Linear(action_horizon * token_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(hidden_dim, expansion_factor) for _ in range(num_blocks)
        )
        self.output_projection = CategorySpecificLinear(
            num_embodiments, hidden_dim, action_horizon * action_dim
        )

    def forward(
        self,
        latent_tokens: Tensor,
        embodiment_ids: Tensor,
        trajectory_mask: Tensor | None = None,
    ) -> Tensor:
        if latent_tokens.ndim != 3:
            raise ValueError(
                f"expected latent tokens [batch, {self.action_horizon}, {self.token_dim}], "
                f"got {tuple(latent_tokens.shape)}"
            )
        expected_shape = (
            latent_tokens.shape[0],
            self.action_horizon,
            self.token_dim,
        )
        if tuple(latent_tokens.shape) != expected_shape:
            raise ValueError(
                f"expected latent tokens [batch, {self.action_horizon}, {self.token_dim}], "
                f"got {tuple(latent_tokens.shape)}"
            )
        batch_size = latent_tokens.shape[0]
        if embodiment_ids.shape != (batch_size,):
            raise ValueError(
                f"expected embodiment_ids shape ({batch_size},), got {tuple(embodiment_ids.shape)}"
            )

        hidden_states = self.latent_projection(latent_tokens.flatten(1))
        for block in self.blocks:
            hidden_states = block(hidden_states)
        trajectory = self.output_projection(hidden_states.unsqueeze(1), embodiment_ids)
        trajectory = trajectory.reshape(batch_size, self.action_horizon, self.action_dim)
        if trajectory_mask is not None:
            trajectory = trajectory * _trajectory_mask(trajectory_mask, trajectory).to(
                dtype=trajectory.dtype
            )
        return trajectory


class VLMConditionPooler(nn.Module):
    """Pool masked VLM tokens while preserving separate image/text summaries."""

    def __init__(
        self,
        input_dim: int,
        condition_dim: int = 512,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        if input_dim <= 0 or condition_dim <= 0:
            raise ValueError("input_dim and condition_dim must be positive")
        hidden_dim = condition_dim if hidden_dim is None else hidden_dim
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        self.projection = nn.Sequential(
            nn.Linear(3 * input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, condition_dim),
        )

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor | None = None,
        image_mask: Tensor | None = None,
        text_mask: Tensor | None = None,
    ) -> Tensor:
        if hidden_states.ndim != 3:
            raise ValueError(
                f"expected hidden_states [batch, sequence, feature], got "
                f"{tuple(hidden_states.shape)}"
            )
        valid_mask = _token_mask(attention_mask, hidden_states)
        global_summary = _masked_token_mean(hidden_states, valid_mask)

        if image_mask is None:
            image_summary = torch.zeros_like(global_summary)
        else:
            valid_image_mask = valid_mask & _token_mask(image_mask, hidden_states)
            image_summary = _masked_token_mean(hidden_states, valid_image_mask)

        if text_mask is None:
            text_summary = torch.zeros_like(global_summary)
        else:
            valid_text_mask = valid_mask & _token_mask(text_mask, hidden_states)
            text_summary = _masked_token_mean(hidden_states, valid_text_mask)

        return self.projection(torch.cat((global_summary, image_summary, text_summary), dim=-1))


class AdaLNResidualMLPBlock(nn.Module):
    """Residual MLP whose pre-normalization is modulated by time and condition."""

    def __init__(
        self,
        latent_dim: int,
        conditioning_dim: int,
        hidden_dim: int | None = None,
    ):
        super().__init__()
        hidden_dim = 4 * latent_dim if hidden_dim is None else hidden_dim
        if min(latent_dim, conditioning_dim, hidden_dim) <= 0:
            raise ValueError("block dimensions must be positive")
        self.norm = nn.LayerNorm(latent_dim, elementwise_affine=False)
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, 3 * latent_dim),
        )
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, latent: Tensor, conditioning: Tensor) -> Tensor:
        shift, scale, gate = self.modulation(conditioning).chunk(3, dim=-1)
        modulated = self.norm(latent) * (1.0 + scale) + shift
        return latent + gate * self.mlp(modulated)


class LatentFlowNet(nn.Module):
    """Four-block AdaLN MLP velocity field in the A2A latent space."""

    def __init__(
        self,
        latent_dim: int = 512,
        condition_dim: int = 512,
        time_embedding_dim: int = 128,
        hidden_dim: int | None = None,
        num_blocks: int = 4,
    ):
        super().__init__()
        if min(latent_dim, condition_dim, time_embedding_dim, num_blocks) <= 0:
            raise ValueError("flow dimensions and num_blocks must be positive")
        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.conditioning_projection = nn.Sequential(
            nn.Linear(condition_dim + time_embedding_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.blocks = nn.ModuleList(
            AdaLNResidualMLPBlock(latent_dim, condition_dim, hidden_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(latent_dim)
        self.output_projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, latent: Tensor, timesteps: Tensor, condition: Tensor) -> Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"expected latent [batch, {self.latent_dim}], got {tuple(latent.shape)}"
            )
        batch_size = latent.shape[0]
        if condition.shape != (batch_size, self.condition_dim):
            raise ValueError(
                f"expected condition shape ({batch_size}, {self.condition_dim}), got "
                f"{tuple(condition.shape)}"
            )
        time_embedding = self.time_embedding(timesteps).to(dtype=latent.dtype)
        conditioning = self.conditioning_projection(
            torch.cat((condition.to(dtype=latent.dtype), time_embedding), dim=-1)
        )
        hidden_states = latent
        for block in self.blocks:
            hidden_states = block(hidden_states, conditioning)
        return self.output_projection(self.output_norm(hidden_states))


def _modulate_tokens(normalized: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply one sample-wise AdaLN shift/scale pair to all temporal tokens."""
    return normalized * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TemporalSelfAttention(nn.Module):
    """ONNX-friendly multi-head self-attention over trajectory time tokens.

    ``nn.MultiheadAttention`` can dispatch to PyTorch's fused
    ``aten::_native_multi_head_attention`` during legacy ONNX tracing, which is
    not exportable in the deployment toolchain used by this repository.  The
    explicit QKV/MatMul/Softmax form below is mathematically equivalent and
    lowers to standard ONNX operators understood by ONNX Runtime and TensorRT.
    """

    def __init__(self, token_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if token_dim <= 0 or num_heads <= 0:
            raise ValueError("attention token_dim and num_heads must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("attention token_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("attention dropout must be in [0, 1)")
        self.token_dim = token_dim
        self.num_heads = num_heads
        self.head_dim = token_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.qkv_projection = nn.Linear(token_dim, 3 * token_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(token_dim, token_dim)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim != 3 or hidden_states.shape[-1] != self.token_dim:
            raise ValueError(
                f"expected attention input [batch, tokens, {self.token_dim}], got "
                f"{tuple(hidden_states.shape)}"
            )
        batch_size, num_tokens, _ = hidden_states.shape
        qkv = self.qkv_projection(hidden_states)
        qkv = qkv.reshape(batch_size, num_tokens, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_probabilities = self.attention_dropout(attention_scores.softmax(dim=-1))
        attended = torch.matmul(attention_probabilities, value)
        attended = attended.transpose(1, 2).reshape(batch_size, num_tokens, self.token_dim)
        return self.output_dropout(self.output_projection(attended))


class AdaLNDiTBlock(nn.Module):
    """A temporal DiT block with AdaLN-modulated attention and feed-forward paths.

    Unlike :class:`AdaLNResidualMLPBlock`, this block contains multi-head
    self-attention.  Its tokens therefore exchange information across relative
    action timesteps.  Flow time and the pooled VLM condition jointly produce
    the two AdaLN shift/scale/gate triplets.
    """

    def __init__(
        self,
        token_dim: int,
        conditioning_dim: int,
        num_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(token_dim, conditioning_dim, num_heads, mlp_ratio) <= 0:
            raise ValueError("DiT block dimensions, heads, and MLP ratio must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("token_dim must be divisible by num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.norm_attention = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.attention = TemporalSelfAttention(token_dim, num_heads, dropout)
        self.norm_mlp = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * mlp_ratio),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout),
            nn.Linear(token_dim * mlp_ratio, token_dim),
            nn.Dropout(dropout),
        )
        self.modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(conditioning_dim, 6 * token_dim),
        )

    def forward(self, hidden_states: Tensor, conditioning: Tensor) -> Tensor:
        (
            attention_shift,
            attention_scale,
            attention_gate,
            mlp_shift,
            mlp_scale,
            mlp_gate,
        ) = self.modulation(conditioning).chunk(6, dim=-1)
        attention_input = _modulate_tokens(
            self.norm_attention(hidden_states), attention_shift, attention_scale
        )
        attention_output = self.attention(attention_input)
        hidden_states = hidden_states + attention_gate.unsqueeze(1) * attention_output
        mlp_input = _modulate_tokens(self.norm_mlp(hidden_states), mlp_shift, mlp_scale)
        return hidden_states + mlp_gate.unsqueeze(1) * self.mlp(mlp_input)


class LatentDiTFlowNet(nn.Module):
    """DiT velocity field over a sequence of action-trajectory latent tokens.

    ``token_mask`` describes which history positions contain real proprioceptive
    feedback during cold start.  Invalid positions are *not* removed with a key
    padding mask: every position must eventually predict a future action token.
    Instead, a learned validity embedding tells the DiT whether a zero token is
    real or synthetic padding while still allowing that query to be generated.
    """

    def __init__(
        self,
        token_dim: int,
        num_tokens: int,
        condition_dim: int = 512,
        time_embedding_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if min(token_dim, num_tokens, condition_dim, time_embedding_dim, num_layers) <= 0:
            raise ValueError("DiT flow dimensions, token count, and layer count must be positive")
        if token_dim % num_heads != 0:
            raise ValueError("a2a_dit_token_dim must be divisible by a2a_dit_num_heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("a2a_dit_dropout must be in [0, 1)")

        self.token_dim = token_dim
        self.num_tokens = num_tokens
        self.condition_dim = condition_dim
        self.time_embedding = SinusoidalTimeEmbedding(time_embedding_dim)
        self.conditioning_projection = nn.Sequential(
            nn.Linear(condition_dim + time_embedding_dim, condition_dim),
            nn.SiLU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.position_embedding = nn.Parameter(torch.empty(1, num_tokens, token_dim))
        self.validity_embedding = nn.Embedding(2, token_dim)
        self.blocks = nn.ModuleList(
            AdaLNDiTBlock(
                token_dim=token_dim,
                conditioning_dim=condition_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(token_dim, elementwise_affine=False)
        self.output_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, 2 * token_dim),
        )
        self.output_projection = nn.Linear(token_dim, token_dim)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.normal_(self.validity_embedding.weight, std=0.02)

    def forward(
        self,
        latent_tokens: Tensor,
        timesteps: Tensor,
        condition: Tensor,
        token_mask: Tensor | None = None,
    ) -> Tensor:
        if latent_tokens.ndim != 3:
            raise ValueError(
                f"expected latent tokens [batch, {self.num_tokens}, {self.token_dim}], "
                f"got {tuple(latent_tokens.shape)}"
            )
        expected_shape = (latent_tokens.shape[0], self.num_tokens, self.token_dim)
        if tuple(latent_tokens.shape) != expected_shape:
            raise ValueError(
                f"expected latent tokens [batch, {self.num_tokens}, {self.token_dim}], "
                f"got {tuple(latent_tokens.shape)}"
            )
        batch_size = latent_tokens.shape[0]
        if condition.shape != (batch_size, self.condition_dim):
            raise ValueError(
                f"expected condition shape ({batch_size}, {self.condition_dim}), got "
                f"{tuple(condition.shape)}"
            )
        if token_mask is None:
            token_mask = torch.ones(
                batch_size,
                self.num_tokens,
                dtype=torch.bool,
                device=latent_tokens.device,
            )
        else:
            token_mask = _token_mask(token_mask, latent_tokens)

        time_embedding = self.time_embedding(timesteps).to(dtype=latent_tokens.dtype)
        conditioning = self.conditioning_projection(
            torch.cat((condition.to(dtype=latent_tokens.dtype), time_embedding), dim=-1)
        )
        hidden_states = (
            latent_tokens
            + self.position_embedding.to(dtype=latent_tokens.dtype)
            + self.validity_embedding(token_mask.long()).to(dtype=latent_tokens.dtype)
        )
        for block in self.blocks:
            hidden_states = block(hidden_states, conditioning)
        output_shift, output_scale = self.output_modulation(conditioning).chunk(2, dim=-1)
        hidden_states = _modulate_tokens(
            self.output_norm(hidden_states), output_shift, output_scale
        )
        return self.output_projection(hidden_states)


def euler_integrate(
    velocity_field: Callable[..., Tensor],
    initial_latent: Tensor,
    condition: Tensor | None = None,
    num_steps: int = 1,
    start_time: float = 0.0,
    end_time: float = 1.0,
    return_trajectory: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """Integrate a latent velocity field with differentiable explicit Euler steps.

    ``velocity_field`` is called as ``field(z, tau, condition)`` when a condition
    is supplied and as ``field(z, tau)`` otherwise.  No ``no_grad`` context or
    in-place update is used, so inferred-latent consistency losses can backprop
    through every solver step.
    """
    if initial_latent.ndim < 2:
        raise ValueError(
            "expected initial_latent with a batch dimension and at least one latent "
            f"dimension, got {tuple(initial_latent.shape)}"
        )
    if not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_steps must be a positive integer")
    if not end_time > start_time:
        raise ValueError("end_time must be greater than start_time")

    step_size = (end_time - start_time) / num_steps
    latent = initial_latent
    trajectory = [latent] if return_trajectory else None
    for step in range(num_steps):
        timesteps = torch.full(
            (latent.shape[0],),
            start_time + step * step_size,
            device=latent.device,
            dtype=torch.float32,
        )
        if condition is None:
            velocity = velocity_field(latent, timesteps)
        else:
            velocity = velocity_field(latent, timesteps, condition)
        if velocity.shape != latent.shape:
            raise ValueError(
                f"velocity field returned shape {tuple(velocity.shape)}, expected "
                f"{tuple(latent.shape)}"
            )
        latent = latent + step_size * velocity
        if trajectory is not None:
            trajectory.append(latent)

    if trajectory is None:
        return latent
    return latent, torch.stack(trajectory, dim=1)
