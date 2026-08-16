# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Configuration for the latent action-to-action GR00T N1.7 variant."""

from dataclasses import dataclass, field
from typing import Any, Literal

from . import register_model_config
from .gr00t_n1d7 import Gr00tN1d7Config


@dataclass(init=False)
class Gr00tN1d7A2AConfig(Gr00tN1d7Config):
    """GR00T N1.7 with selectable latent A2A flow-matching heads.

    The VLM backbone is unchanged.  The raw-action Gaussian flow head is replaced
    by a trajectory autoencoder and either the paper-core four-block AdaLN MLP or
    an experimental temporal-token DiT velocity network.
    """

    model_type: str = "Gr00tN1d7A2A"

    # The paper-core migration profile uses equal eight-step source/target
    # windows.  It deliberately keeps GR00T's configured visual horizon and VLM
    # backbone instead of claiming a bit-for-bit reproduction of ResNet18/m=8.
    action_horizon: int = 8
    a2a_history_horizon: int = 8
    a2a_future_horizon: int = 8
    a2a_latent_dim: int = 512
    a2a_encoder_channels: tuple[int, ...] = (128, 256, 512)
    a2a_encoder_kernel_size: int = 5
    a2a_flow_blocks: int = 4
    a2a_flow_mlp_ratio: int = 4
    a2a_decoder_res_blocks: int = 4

    # Select only the latent velocity-field/autoencoder layout. ``mlp`` is the
    # existing paper-core implementation and remains the default so old config
    # files and checkpoints keep exactly the same module graph. ``dit`` is an
    # explicit GR00T research ablation: it preserves the trajectory time axis as
    # [B, H, token_dim] and applies temporal self-attention.
    a2a_flow_backbone: Literal["mlp", "dit"] = "mlp"
    a2a_dit_token_dim: int = 256
    a2a_dit_num_layers: int = 8
    a2a_dit_num_heads: int = 8
    a2a_dit_mlp_ratio: int = 4
    a2a_dit_dropout: float = 0.0

    # Paper losses: L = L_FM + .5 L_AE + L_IC and
    # L_IC = latent_L1 + .5 action_L1.
    a2a_lambda_fm: float = 1.0
    a2a_lambda_ae: float = 0.5
    a2a_lambda_ic: float = 1.0
    a2a_lambda_ic_action: float = 0.5
    a2a_lambda_aux: float = 1.0

    a2a_history_noise_train_std: float = 0.02
    a2a_history_noise_infer_std: float = 0.0
    a2a_num_inference_steps: int = 1
    a2a_ic_train_steps: int = 1
    a2a_include_current_state_condition: bool = False

    # Staged migration support. ``autoencoder`` isolates trajectory
    # reconstruction, ``flow_only`` freezes the learned trajectory space, and
    # ``joint`` optimizes the complete paper objective.
    a2a_training_stage: Literal["autoencoder", "flow_only", "joint"] = "joint"
    # Retained public field name for checkpoint/CLI compatibility. Semantically
    # this now means the fixed paper-core migration profile described above.
    a2a_strict_paper_architecture: bool = True

    # Explicit per-embodiment channel contract. Strict mode requires it;
    # dimension-based inference exists only behind the opt-out exploratory path.
    a2a_channel_specs: dict[str, list[dict[str, Any]]] | None = None
    a2a_require_explicit_channel_specs: bool = True
    a2a_require_semantic_metadata: bool = True

    # JSON emitted by scripts/a2a/build_canonical_stats.py. Strict training
    # requires this artifact so history and future never silently use separate
    # state/action scales. Full A2A checkpoints embed the same data in their
    # processor config and do not depend on the original path at inference.
    a2a_canonical_statistics_path: str | None = None
    a2a_expected_contract_sha256: str | None = None
    a2a_require_canonical_statistics: bool = True

    # A base N1.7 checkpoint only seeds the VLM-side allowlist.  Full A2A resume
    # continues to use training.start_from_checkpoint.
    a2a_base_vla_checkpoint: str | None = None
    a2a_cold_start: str = "repeat_first_state"
    a2a_max_time_gap_s: float | None = None

    # Original GR00T field names are retained for CLI/config compatibility.
    # ``tune_diffusion_model`` controls whichever A2A FlowNet backend is selected.
    tune_diffusion_model: bool = True
    state_dropout_prob: float = 0.0

    # Human-readable architecture metadata saved in config.json.
    a2a_architecture: dict[str, Any] = field(
        default_factory=lambda: {
            "source": "executed_proprio_history",
            "path": "linear_latent_interpolation",
            "encoder": "conv1d_channels_128_256_512_kernel5",
            "flow": "4_block_adaln_mlp_ratio4",
            "decoder": "4_block_residual_mlp_ratio4",
        }
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.validate_paper_architecture()
        architecture = dict(self.a2a_architecture)
        architecture["flow_backbone"] = self.a2a_flow_backbone
        if self.a2a_flow_backbone == "mlp":
            architecture["latent_layout"] = f"vector_{self.a2a_latent_dim}"
            architecture["flow"] = (
                f"{self.a2a_flow_blocks}_block_adaln_mlp_ratio{self.a2a_flow_mlp_ratio}"
            )
        else:
            architecture["latent_layout"] = (
                f"{self.a2a_history_horizon}_temporal_tokens_x_{self.a2a_dit_token_dim}"
            )
            architecture["flow"] = (
                f"{self.a2a_dit_num_layers}_block_adaln_dit_"
                f"{self.a2a_dit_num_heads}_heads_ratio{self.a2a_dit_mlp_ratio}"
            )
        self.a2a_architecture = architecture

    def validate_paper_architecture(self) -> None:
        """Reject drift from the repository's fixed paper-core migration profile.

        The paper specifies H=8, a 512-D space, 3 Conv1d layers with kernel 5,
        four AdaLN-MLP FlowNet blocks, and four decoder MLP blocks. It does not
        publish the intermediate Conv widths, MLP ratio, activations, or
        normalization details. Those are fixed repository implementation
        choices, not claims about undisclosed author code.
        """
        if self.a2a_flow_backbone not in {"mlp", "dit"}:
            raise ValueError(
                f"a2a_flow_backbone must be 'mlp' or 'dit', got {self.a2a_flow_backbone!r}"
            )
        dit_integer_fields = {
            "a2a_dit_token_dim": self.a2a_dit_token_dim,
            "a2a_dit_num_layers": self.a2a_dit_num_layers,
            "a2a_dit_num_heads": self.a2a_dit_num_heads,
            "a2a_dit_mlp_ratio": self.a2a_dit_mlp_ratio,
        }
        nonpositive = {name: value for name, value in dit_integer_fields.items() if value <= 0}
        if nonpositive:
            raise ValueError(f"A2A DiT dimensions must be positive: {nonpositive}")
        if self.a2a_dit_token_dim % self.a2a_dit_num_heads != 0:
            raise ValueError("a2a_dit_token_dim must be divisible by a2a_dit_num_heads")
        if not 0.0 <= self.a2a_dit_dropout < 1.0:
            raise ValueError("a2a_dit_dropout must be in [0, 1)")
        if self.a2a_flow_backbone == "dit" and (
            self.a2a_history_horizon != self.a2a_future_horizon
        ):
            raise ValueError("Temporal-token A2A DiT requires equal history/future horizons")

        if not self.a2a_strict_paper_architecture:
            return

        try:
            encoder_channels = tuple(self.a2a_encoder_channels)
        except TypeError:
            encoder_channels = self.a2a_encoder_channels

        strict_values = {
            "flow_backbone": (self.a2a_flow_backbone, "mlp"),
            "action_horizon": (self.action_horizon, 8),
            "history_horizon": (self.a2a_history_horizon, 8),
            "future_horizon": (self.a2a_future_horizon, 8),
            "latent_dim": (self.a2a_latent_dim, 512),
            "encoder_channels": (encoder_channels, (128, 256, 512)),
            "encoder_kernel": (self.a2a_encoder_kernel_size, 5),
            "flow_blocks": (self.a2a_flow_blocks, 4),
            "flow_mlp_ratio": (self.a2a_flow_mlp_ratio, 4),
            "decoder_blocks": (self.a2a_decoder_res_blocks, 4),
            "include_current_state_condition": (
                self.a2a_include_current_state_condition,
                False,
            ),
        }
        mismatches = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in strict_values.items()
            if actual != expected
        }
        if mismatches:
            raise ValueError(
                "Strict A2A migration profile requires the paper-core MLP backend, H=8, "
                "latent=512, Conv1d channels (128, 256, 512) with kernel 5, four "
                "FlowNet/decoder blocks, and MLP ratio 4; "
                f"mismatches={mismatches}"
            )


register_model_config("Gr00tN1d7A2A", Gr00tN1d7A2AConfig)
