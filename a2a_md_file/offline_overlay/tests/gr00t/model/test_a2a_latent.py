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

"""CPU tests for the model-independent latent A2A building blocks."""

from gr00t.model.modules.a2a_latent import (
    ActionTrajectoryDecoder,
    ActionTrajectoryEncoder,
    ActionTrajectoryTokenDecoder,
    ActionTrajectoryTokenEncoder,
    LatentDiTFlowNet,
    LatentFlowNet,
    SinusoidalTimeEmbedding,
    VLMConditionPooler,
    euler_integrate,
    masked_l1_loss,
)
import pytest
import torch


def _small_encoder() -> ActionTrajectoryEncoder:
    return ActionTrajectoryEncoder(
        action_dim=6,
        latent_dim=16,
        num_embodiments=3,
        conv_channels=(8, 12, 16),
        kernel_size=5,
    )


class TestActionTrajectoryAutoencoder:
    def test_encoder_decoder_shapes_and_gradients(self):
        encoder = _small_encoder()
        decoder = ActionTrajectoryDecoder(
            action_dim=6,
            action_horizon=5,
            latent_dim=16,
            hidden_dim=24,
            num_embodiments=3,
            num_blocks=4,
        )
        actions = torch.randn(3, 5, 6, requires_grad=True)
        embodiment_ids = torch.tensor([0, 1, 2])
        mask = torch.ones_like(actions, dtype=torch.bool)
        mask[1, -1] = False

        latent = encoder(actions, embodiment_ids, mask)
        reconstruction = decoder(latent, embodiment_ids, mask)
        loss = masked_l1_loss(reconstruction, actions, mask)
        loss.backward()

        assert latent.shape == (3, 16)
        assert reconstruction.shape == (3, 5, 6)
        assert torch.count_nonzero(reconstruction[1, -1]) == 0
        assert actions.grad is not None
        assert torch.isfinite(actions.grad).all()
        assert all(parameter.grad is not None for parameter in encoder.parameters())
        assert all(parameter.grad is not None for parameter in decoder.parameters())

    def test_masked_values_cannot_change_latent(self):
        torch.manual_seed(7)
        encoder = _small_encoder().eval()
        actions = torch.randn(2, 7, 6)
        mask = torch.ones_like(actions, dtype=torch.bool)
        mask[:, -2:] = False
        perturbed = actions.clone()
        perturbed[~mask] = torch.nan
        embodiment_ids = torch.tensor([0, 2])

        actual = encoder(actions, embodiment_ids, mask)
        changed_only_under_mask = encoder(perturbed, embodiment_ids, mask)

        torch.testing.assert_close(actual, changed_only_under_mask, atol=0, rtol=0)

    def test_explicit_padding_matches_shorter_trajectory(self):
        torch.manual_seed(11)
        encoder = _small_encoder().eval()
        short = torch.randn(2, 5, 6)
        padded = torch.cat((short, torch.randn(2, 3, 6)), dim=1)
        short_mask = torch.ones_like(short, dtype=torch.bool)
        padded_mask = torch.zeros_like(padded, dtype=torch.bool)
        padded_mask[:, : short.shape[1]] = True
        embodiment_ids = torch.tensor([1, 2])

        short_latent = encoder(short, embodiment_ids, short_mask)
        padded_latent = encoder(padded, embodiment_ids, padded_mask)

        torch.testing.assert_close(short_latent, padded_latent, atol=1e-6, rtol=1e-6)

    def test_all_masked_trajectory_maps_to_zero(self):
        encoder = _small_encoder().eval()
        actions = torch.randn(2, 4, 6)
        latent = encoder(
            actions,
            torch.tensor([0, 1]),
            torch.zeros_like(actions, dtype=torch.bool),
        )
        torch.testing.assert_close(latent, torch.zeros_like(latent), atol=0, rtol=0)


class TestVLMConditionPooler:
    def test_shape_and_mask_invariance(self):
        torch.manual_seed(3)
        pooler = VLMConditionPooler(input_dim=12, condition_dim=16, hidden_dim=20).eval()
        hidden_states = torch.randn(2, 8, 12)
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.bool
        )
        image_mask = torch.tensor(
            [[1, 1, 0, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool
        )
        text_mask = attention_mask & ~image_mask
        perturbed = hidden_states.clone()
        perturbed[~attention_mask] = torch.nan

        condition = pooler(hidden_states, attention_mask, image_mask, text_mask)
        perturbed_condition = pooler(perturbed, attention_mask, image_mask, text_mask)

        assert condition.shape == (2, 16)
        torch.testing.assert_close(condition, perturbed_condition, atol=0, rtol=0)

    def test_missing_modality_masks_are_supported(self):
        pooler = VLMConditionPooler(input_dim=4, condition_dim=6)
        result = pooler(torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))
        assert result.shape == (2, 6)


class TestTemporalTokenAutoencoder:
    def test_token_encoder_decoder_shapes_masks_and_gradients(self):
        encoder = ActionTrajectoryTokenEncoder(
            action_dim=6,
            trajectory_horizon=5,
            token_dim=16,
            num_embodiments=3,
            conv_channels=(8, 12, 16),
            kernel_size=5,
        )
        decoder = ActionTrajectoryTokenDecoder(
            action_dim=6,
            action_horizon=5,
            token_dim=16,
            hidden_dim=24,
            num_embodiments=3,
            num_blocks=4,
        )
        actions = torch.randn(3, 5, 6, requires_grad=True)
        embodiment_ids = torch.tensor([0, 1, 2])
        mask = torch.ones_like(actions, dtype=torch.bool)
        mask[1, -1] = False

        tokens = encoder(actions, embodiment_ids, mask)
        reconstruction = decoder(tokens, embodiment_ids, mask)
        masked_l1_loss(reconstruction, actions, mask).backward()

        assert tokens.shape == (3, 5, 16)
        assert reconstruction.shape == (3, 5, 6)
        assert torch.count_nonzero(tokens[1, -1]) == 0
        assert torch.count_nonzero(reconstruction[1, -1]) == 0
        assert actions.grad is not None and torch.isfinite(actions.grad).all()
        assert all(parameter.grad is not None for parameter in encoder.parameters())
        assert all(parameter.grad is not None for parameter in decoder.parameters())

    def test_masked_token_values_cannot_leak_through_temporal_convolutions(self):
        torch.manual_seed(17)
        encoder = ActionTrajectoryTokenEncoder(
            action_dim=6,
            trajectory_horizon=5,
            token_dim=16,
            num_embodiments=2,
            conv_channels=(8, 12, 16),
            kernel_size=5,
        ).eval()
        actions = torch.randn(2, 5, 6)
        mask = torch.ones_like(actions, dtype=torch.bool)
        mask[:, :2] = False
        perturbed = actions.clone()
        perturbed[~mask] = torch.nan
        embodiment_ids = torch.tensor([0, 1])

        expected = encoder(actions, embodiment_ids, mask)
        actual = encoder(perturbed, embodiment_ids, mask)

        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(actual[:, :2], torch.zeros_like(actual[:, :2]))


class TestLatentFlow:
    def test_time_embedding_shape_for_odd_dimension(self):
        embedding = SinusoidalTimeEmbedding(7)(torch.tensor([0.0, 0.5, 1.0]))
        assert embedding.shape == (3, 7)
        assert torch.isfinite(embedding).all()

    def test_four_block_flownet_shape_and_gradient(self):
        flow = LatentFlowNet(
            latent_dim=16,
            condition_dim=12,
            time_embedding_dim=8,
            hidden_dim=32,
            num_blocks=4,
        )
        latent = torch.randn(3, 16, requires_grad=True)
        condition = torch.randn(3, 12, requires_grad=True)
        velocity = flow(latent, torch.tensor([0.0, 0.4, 1.0]), condition)
        velocity.square().mean().backward()

        assert velocity.shape == latent.shape
        assert latent.grad is not None and torch.isfinite(latent.grad).all()
        assert condition.grad is not None and torch.isfinite(condition.grad).all()
        assert len(flow.blocks) == 4

    def test_differentiable_euler_matches_constant_velocity_oracle(self):
        initial = torch.randn(2, 3, requires_grad=True)
        condition = torch.randn(2, 3, requires_grad=True)

        def constant_velocity(latent, timesteps, flow_condition):
            del latent, timesteps
            return flow_condition

        result, trajectory = euler_integrate(
            constant_velocity,
            initial,
            condition,
            num_steps=4,
            return_trajectory=True,
        )
        torch.testing.assert_close(result, initial + condition)
        assert trajectory.shape == (2, 5, 3)

        result.sum().backward()
        torch.testing.assert_close(initial.grad, torch.ones_like(initial))
        torch.testing.assert_close(condition.grad, torch.ones_like(condition))

    def test_euler_rejects_wrong_velocity_shape(self):
        with pytest.raises(ValueError, match="velocity field returned shape"):
            euler_integrate(lambda latent, time: latent[:, :-1], torch.randn(2, 4))

    def test_temporal_dit_shape_gradient_and_time_attention(self):
        torch.manual_seed(23)
        flow = LatentDiTFlowNet(
            token_dim=16,
            num_tokens=5,
            condition_dim=12,
            time_embedding_dim=8,
            num_layers=2,
            num_heads=4,
            mlp_ratio=2,
        )
        latent = torch.randn(3, 5, 16, requires_grad=True)
        condition = torch.randn(3, 12, requires_grad=True)
        token_mask = torch.tensor(
            [[0, 0, 1, 1, 1], [1, 1, 1, 1, 1], [0, 1, 1, 1, 1]],
            dtype=torch.bool,
        )
        velocity = flow(
            latent,
            torch.tensor([0.0, 0.4, 1.0]),
            condition,
            token_mask,
        )
        velocity.square().mean().backward()

        assert velocity.shape == latent.shape
        assert latent.grad is not None and torch.isfinite(latent.grad).all()
        assert condition.grad is not None and torch.isfinite(condition.grad).all()
        assert len(flow.blocks) == 2
        assert all(block.attention.num_heads == 4 for block in flow.blocks)

    def test_temporal_dit_uses_validity_embedding_for_cold_start(self):
        torch.manual_seed(29)
        flow = LatentDiTFlowNet(
            token_dim=8,
            num_tokens=4,
            condition_dim=8,
            time_embedding_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2,
        ).eval()
        latent = torch.zeros(1, 4, 8)
        time = torch.zeros(1)
        condition = torch.randn(1, 8)

        full = flow(latent, time, condition, torch.ones(1, 4, dtype=torch.bool))
        cold = flow(
            latent,
            time,
            condition,
            torch.tensor([[0, 0, 0, 1]], dtype=torch.bool),
        )

        assert not torch.allclose(full, cold)

    def test_temporal_attention_couples_different_action_timesteps(self):
        torch.manual_seed(37)
        flow = LatentDiTFlowNet(
            token_dim=8,
            num_tokens=4,
            condition_dim=8,
            time_embedding_dim=8,
            num_layers=1,
            num_heads=2,
            mlp_ratio=2,
        ).eval()
        latent = torch.randn(1, 4, 8)
        changed = latent.clone()
        changed[:, 0] += torch.linspace(0.1, 0.8, 8)
        time = torch.tensor([0.3])
        condition = torch.randn(1, 8)

        baseline_velocity = flow(latent, time, condition)
        changed_velocity = flow(changed, time, condition)

        # Token 3 was not modified directly. A changed output there proves that
        # token 0 communicated through temporal self-attention.
        assert not torch.allclose(baseline_velocity[:, 3], changed_velocity[:, 3])

    def test_euler_integrates_temporal_token_latents(self):
        initial = torch.randn(2, 4, 8, requires_grad=True)
        condition = torch.randn(2, 4, 8, requires_grad=True)

        def constant_velocity(latent, timesteps, flow_condition):
            del latent, timesteps
            return flow_condition

        result, trajectory = euler_integrate(
            constant_velocity,
            initial,
            condition,
            num_steps=2,
            return_trajectory=True,
        )

        torch.testing.assert_close(result, initial + condition)
        assert trajectory.shape == (2, 3, 4, 8)


class TestMaskedL1:
    def test_uses_only_valid_entries(self):
        prediction = torch.tensor([[[1.0, 100.0], [3.0, 100.0]]])
        target = torch.zeros_like(prediction)
        mask = torch.tensor([[[1, 0], [1, 0]]], dtype=torch.bool)
        assert masked_l1_loss(prediction, target, mask).item() == 2.0

    def test_all_false_mask_is_differentiable_zero(self):
        prediction = torch.randn(2, 3, requires_grad=True)
        loss = masked_l1_loss(
            prediction, torch.zeros_like(prediction), torch.zeros_like(prediction)
        )
        loss.backward()
        assert loss.item() == 0.0
        torch.testing.assert_close(prediction.grad, torch.zeros_like(prediction))
