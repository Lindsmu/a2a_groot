# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Training pipeline and strict base-VLA migration for latent A2A."""

from copy import deepcopy
import gc
import json
import logging

from transformers import AutoConfig, AutoModel

from gr00t.configs.model.gr00t_n1d7_a2a import Gr00tN1d7A2AConfig
from gr00t.model.gr00t_n1d7.setup import Gr00tN1d7Pipeline
from gr00t.model.registry import register_model
from gr00t.utils.dist_utils import run_or_wait_on_rank0

from .gr00t_n1d7_a2a import Gr00tN1d7A2A
from .processing_gr00t_n1d7_a2a import Gr00tN1d7A2AProcessor


class Gr00tN1d7A2APipeline(Gr00tN1d7Pipeline):
    model_class = Gr00tN1d7A2A
    processor_class = Gr00tN1d7A2AProcessor

    def setup(self):
        self._configure_a2a_windows()
        self._normalize_checkpoint_sources()
        if (
            self.model_config.a2a_training_stage == "flow_only"
            and self.config.training.start_from_checkpoint is None
        ):
            raise ValueError(
                "a2a_training_stage='flow_only' requires a full A2A checkpoint in "
                "training.start_from_checkpoint; freezing a randomly initialized AE is invalid"
            )
        super().setup()

    def _normalize_checkpoint_sources(self) -> None:
        """Distinguish a full A2A resume from a base N1.7 initialization."""
        checkpoint = self.config.training.start_from_checkpoint
        if checkpoint is None:
            return
        checkpoint_config = AutoConfig.from_pretrained(
            checkpoint, **self.transformers_loading_kwargs
        )
        checkpoint_type = getattr(checkpoint_config, "model_type", None)
        if checkpoint_type == "Gr00tN1d7A2A":
            # A full A2A resume must keep the serialized module graph. In
            # particular, silently loading an MLP checkpoint into a DiT head
            # would leave most of the new action head randomly initialized.
            common_fields = (
                "action_horizon",
                "a2a_history_horizon",
                "a2a_future_horizon",
                "a2a_latent_dim",
                "a2a_encoder_channels",
                "a2a_encoder_kernel_size",
                "a2a_decoder_res_blocks",
                "a2a_flow_backbone",
            )
            backend = getattr(checkpoint_config, "a2a_flow_backbone", "mlp")
            backend_fields = (
                ("a2a_flow_blocks", "a2a_flow_mlp_ratio")
                if backend == "mlp"
                else (
                    "a2a_dit_token_dim",
                    "a2a_dit_num_layers",
                    "a2a_dit_num_heads",
                    "a2a_dit_mlp_ratio",
                    "a2a_dit_dropout",
                )
            )
            mismatches = {}
            for field in (*common_fields, *backend_fields):
                checkpoint_value = getattr(
                    checkpoint_config,
                    field,
                    "mlp" if field == "a2a_flow_backbone" else None,
                )
                requested_value = getattr(self.model_config, field)
                if field == "a2a_encoder_channels":
                    checkpoint_value = tuple(checkpoint_value)
                    requested_value = tuple(requested_value)
                if checkpoint_value != requested_value:
                    mismatches[field] = {
                        "checkpoint": checkpoint_value,
                        "requested": requested_value,
                    }
            if mismatches:
                raise ValueError(
                    "A full A2A checkpoint cannot change its MLP/DiT module graph. "
                    "Start the comparison from the same original Gr00tN1d7 base "
                    f"checkpoint instead; mismatches={mismatches}"
                )
            return
        if checkpoint_type != "Gr00tN1d7":
            raise ValueError(
                "A2A start_from_checkpoint must be a full Gr00tN1d7A2A checkpoint or "
                f"an original Gr00tN1d7 base checkpoint, got {checkpoint_type!r}"
            )
        configured_base = self.model_config.a2a_base_vla_checkpoint
        if configured_base is not None and str(configured_base) != str(checkpoint):
            raise ValueError(
                "Conflicting original N1.7 checkpoints were supplied through "
                "start_from_checkpoint and a2a_base_vla_checkpoint"
            )
        self.model_config.a2a_base_vla_checkpoint = str(checkpoint)
        self.config.training.start_from_checkpoint = None
        logging.info(
            "Treating original N1.7 start_from_checkpoint=%s as A2A base initialization",
            checkpoint,
        )

    def _configure_a2a_windows(self) -> None:
        history = int(self.model_config.a2a_history_horizon)
        future = int(self.model_config.a2a_future_horizon)
        if history != future:
            raise ValueError("Strict latent A2A requires equal history and future horizons")
        if int(self.model_config.action_horizon) != future:
            raise ValueError(
                "action_horizon must equal a2a_future_horizon for the strict A2A variant"
            )
        if int(self.model_config.state_history_length) != 1:
            raise ValueError(
                "state_history_length must remain 1; A2A history is carried separately"
            )

        updated = {}
        for embodiment, modalities in self.config.data.modality_configs.items():
            modalities = deepcopy(modalities)
            modalities["state"].delta_indices = list(range(-(history - 1), 1))
            modalities["action"].delta_indices = list(range(future))
            updated[embodiment] = modalities
        self.config.data.modality_configs = updated

    def _get_processor_extra_kwargs(self) -> dict:
        kwargs = {
            "a2a_history_horizon": self.model_config.a2a_history_horizon,
            "a2a_future_horizon": self.model_config.a2a_future_horizon,
            "a2a_cold_start": self.model_config.a2a_cold_start,
            "a2a_require_canonical_statistics": (
                self.model_config.a2a_require_canonical_statistics
            ),
            "a2a_require_explicit_channel_specs": (
                self.model_config.a2a_require_explicit_channel_specs
            ),
            "a2a_require_semantic_metadata": self.model_config.a2a_require_semantic_metadata,
        }
        if self.model_config.a2a_expected_contract_sha256 is not None:
            kwargs["a2a_expected_contract_sha256"] = self.model_config.a2a_expected_contract_sha256
        if self.model_config.a2a_channel_specs is not None:
            kwargs["a2a_channel_specs"] = self.model_config.a2a_channel_specs
        # A resumed A2A processor already embeds the canonical statistics. Do
        # not re-open a possibly machine-local source path from the old run.
        if (
            self.config.training.start_from_checkpoint is None
            and self.model_config.a2a_canonical_statistics_path is not None
        ):
            kwargs["a2a_canonical_statistics_path"] = (
                self.model_config.a2a_canonical_statistics_path
            )
        return kwargs

    def _get_model_extra_kwargs(self) -> dict:
        """Apply the requested A2A experiment stage when loading a full checkpoint.

        In particular, an AE warmup checkpoint stores ``autoencoder`` in its
        config. Loading it for the next ``joint`` stage must not silently retain
        the old stage and freeze the FlowNet path.
        """
        fields = (
            "action_horizon",
            "a2a_history_horizon",
            "a2a_future_horizon",
            "a2a_latent_dim",
            "a2a_encoder_channels",
            "a2a_encoder_kernel_size",
            "a2a_flow_blocks",
            "a2a_flow_mlp_ratio",
            "a2a_decoder_res_blocks",
            "a2a_flow_backbone",
            "a2a_dit_token_dim",
            "a2a_dit_num_layers",
            "a2a_dit_num_heads",
            "a2a_dit_mlp_ratio",
            "a2a_dit_dropout",
            "a2a_lambda_fm",
            "a2a_lambda_ae",
            "a2a_lambda_ic",
            "a2a_lambda_ic_action",
            "a2a_lambda_aux",
            "a2a_history_noise_train_std",
            "a2a_history_noise_infer_std",
            "a2a_num_inference_steps",
            "a2a_ic_train_steps",
            "a2a_include_current_state_condition",
            "a2a_training_stage",
            "a2a_strict_paper_architecture",
            "a2a_channel_specs",
            "a2a_require_explicit_channel_specs",
            "a2a_require_semantic_metadata",
            "a2a_expected_contract_sha256",
            "a2a_require_canonical_statistics",
            "a2a_cold_start",
            "a2a_max_time_gap_s",
        )
        return {field: getattr(self.model_config, field) for field in fields}

    def _create_model(self):
        model = super()._create_model()
        if not isinstance(model, Gr00tN1d7A2A):
            raise TypeError(
                "A2A pipeline loaded a non-A2A model. Use a2a_base_vla_checkpoint for an "
                "original N1.7 initializer and start_from_checkpoint only for full A2A resume."
            )
        base_checkpoint = self.model_config.a2a_base_vla_checkpoint
        if base_checkpoint is None or self.config.training.start_from_checkpoint is not None:
            return model

        logging.info("Loading base GR00T VLA allowlist from %s", base_checkpoint)
        base_model = AutoModel.from_pretrained(
            base_checkpoint,
            transformers_loading_kwargs=self.transformers_loading_kwargs,
            **self.transformers_loading_kwargs,
        )
        if getattr(base_model.config, "model_type", None) != "Gr00tN1d7":
            raise ValueError(
                "a2a_base_vla_checkpoint must contain the original Gr00tN1d7 model, got "
                f"{getattr(base_model.config, 'model_type', None)!r}"
            )

        # Strictly copy only the documented allowlist. Any backbone drift fails.
        backbone_result = model.backbone.load_state_dict(
            base_model.backbone.state_dict(), strict=True
        )
        reused = ["backbone.*"]
        if hasattr(base_model.action_head, "vlln") and hasattr(model.action_head, "vlln"):
            model.action_head.vlln.load_state_dict(
                base_model.action_head.vlln.state_dict(), strict=True
            )
            reused.append("action_head.vlln.*")
        report = {
            "source": str(base_checkpoint),
            "source_model_type": base_model.config.model_type,
            "reused_allowlist": reused,
            "ignored": [
                "action_head.model.*",
                "action_head.state_encoder.*",
                "action_head.action_encoder.*",
                "action_head.action_decoder.*",
            ],
            "backbone_missing": list(backbone_result.missing_keys),
            "backbone_unexpected": list(backbone_result.unexpected_keys),
        }
        report_path = self.save_cfg_dir / "base_vla_migration_report.json"
        with run_or_wait_on_rank0(label="A2A base migration report") as is_rank0:
            if is_rank0:
                with report_path.open("w", encoding="utf-8") as handle:
                    json.dump(report, handle, indent=2)
        del base_model
        gc.collect()
        logging.info("Base VLA migration report written to %s", report_path)
        return model


register_model(Gr00tN1d7A2AConfig, Gr00tN1d7A2APipeline)
