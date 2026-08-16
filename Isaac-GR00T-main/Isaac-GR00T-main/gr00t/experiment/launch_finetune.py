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

# Launch finetuning for N1.7 on "single node".
# This script tries to provide a similar user experience as current OSS.

import json
import os
from pathlib import Path

import tyro

from gr00t.configs.base_config import get_default_config
from gr00t.configs.finetune_config import FinetuneConfig
from gr00t.experiment.experiment import run


# Make sure the user provided modality config is registered.
def load_modality_config(modality_config_path: str):
    import importlib
    import sys

    path = Path(modality_config_path)
    if path.exists() and path.suffix == ".py":
        sys.path.append(str(path.parent))
        importlib.import_module(path.stem)
        print(f"Loaded modality config: {path}")
    else:
        raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")


if __name__ == "__main__":
    # Set LOGURU_LEVEL environment variable if not already set (default: INFO)
    if "LOGURU_LEVEL" not in os.environ:
        os.environ["LOGURU_LEVEL"] = "INFO"
    # Use tyro for clean CLI
    ft_config = tyro.cli(FinetuneConfig, description=__doc__)
    from gr00t.data.embodiment_tags import EmbodimentTag

    ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
    embodiment_tag = ft_config.embodiment_tag.value

    # all rank workers should register for the modality config
    if ft_config.modality_config_path is not None:
        load_modality_config(ft_config.modality_config_path)

    dataset_paths = [path for path in ft_config.dataset_path.split(os.pathsep) if path]

    model_overrides = {"model_type": ft_config.model_type}
    a2a_options = {
        "a2a_canonical_statistics_path": ft_config.a2a_canonical_statistics_path,
        "a2a_expected_contract_sha256": ft_config.a2a_expected_contract_sha256,
        "a2a_training_stage": ft_config.a2a_training_stage,
        "a2a_cold_start": ft_config.a2a_cold_start,
        "a2a_num_inference_steps": ft_config.a2a_num_inference_steps,
        "a2a_history_noise_train_std": ft_config.a2a_history_noise_train_std,
        "a2a_max_time_gap_s": ft_config.a2a_max_time_gap_s,
        "a2a_flow_backbone": ft_config.a2a_flow_backbone,
        "a2a_dit_token_dim": ft_config.a2a_dit_token_dim,
        "a2a_dit_num_layers": ft_config.a2a_dit_num_layers,
        "a2a_dit_num_heads": ft_config.a2a_dit_num_heads,
        "a2a_dit_mlp_ratio": ft_config.a2a_dit_mlp_ratio,
        "a2a_dit_dropout": ft_config.a2a_dit_dropout,
        "a2a_strict_paper_architecture": ft_config.a2a_strict_paper_architecture,
    }
    is_a2a = ft_config.model_type == "Gr00tN1d7A2A"
    if is_a2a:
        if ft_config.a2a_channel_specs_path is None:
            raise ValueError(
                "Gr00tN1d7A2A requires --a2a-channel-specs-path; equal state/action "
                "dimensions are not a physical-semantics contract"
            )
        specs_path = Path(ft_config.a2a_channel_specs_path)
        with specs_path.open("r", encoding="utf-8") as handle:
            channel_specs = json.load(handle)
        if not isinstance(channel_specs, dict) or embodiment_tag not in channel_specs:
            raise ValueError(
                "A2A channel specs must be a JSON object containing the selected "
                f"embodiment {embodiment_tag!r}"
            )
        if ft_config.a2a_canonical_statistics_path is None:
            raise ValueError(
                "Gr00tN1d7A2A requires --a2a-canonical-statistics-path built from "
                "unnormalized canonical history/future trajectories"
            )
        if ft_config.a2a_expected_contract_sha256 is None:
            raise ValueError(
                "Gr00tN1d7A2A requires --a2a-expected-contract-sha256 to bind the "
                "training configuration to its data/controller contract"
            )
        model_overrides["a2a_channel_specs"] = channel_specs
        model_overrides.update(a2a_options)
    elif (
        ft_config.a2a_channel_specs_path is not None
        or ft_config.a2a_canonical_statistics_path is not None
        or ft_config.a2a_expected_contract_sha256 is not None
        or ft_config.a2a_training_stage != "joint"
        or ft_config.a2a_cold_start != "repeat_first_state"
        or ft_config.a2a_num_inference_steps != 1
        or ft_config.a2a_history_noise_train_std != 0.02
        or ft_config.a2a_max_time_gap_s is not None
        or ft_config.a2a_flow_backbone != "mlp"
        or ft_config.a2a_dit_token_dim != 256
        or ft_config.a2a_dit_num_layers != 8
        or ft_config.a2a_dit_num_heads != 8
        or ft_config.a2a_dit_mlp_ratio != 4
        or ft_config.a2a_dit_dropout != 0.0
        or not ft_config.a2a_strict_paper_architecture
    ):
        raise ValueError("A2A-specific options require --model-type Gr00tN1d7A2A")

    config = get_default_config().load_dict(
        {
            "model": model_overrides,
            "data": {
                "download_cache": False,
                "datasets": [
                    {
                        "dataset_paths": dataset_paths,
                        "mix_ratio": 1.0,
                        "embodiment_tag": embodiment_tag,
                    }
                ],
            },
        }
    )
    config.load_config_path = None

    # overwrite with finetune config supplied by the user
    config.model.tune_llm = ft_config.tune_llm
    config.model.tune_visual = ft_config.tune_visual
    config.model.tune_projector = ft_config.tune_projector
    config.model.tune_diffusion_model = ft_config.tune_diffusion_model
    config.model.state_dropout_prob = ft_config.state_dropout_prob
    config.model.random_rotation_angle = ft_config.random_rotation_angle
    config.model.color_jitter_params = ft_config.color_jitter_params
    config.model.use_percentiles = ft_config.use_percentiles
    if (ft_config.shortest_image_edge is None) != (ft_config.crop_fraction is None):
        raise ValueError("shortest_image_edge and crop_fraction must be set together")
    if ft_config.shortest_image_edge is not None:
        config.model.shortest_image_edge = ft_config.shortest_image_edge
        config.model.crop_fraction = ft_config.crop_fraction
        config.model.image_crop_size = None
        config.model.image_target_size = None
    if ft_config.extra_augmentation_config:
        config.model.extra_augmentation_config = json.loads(ft_config.extra_augmentation_config)
    else:
        config.model.extra_augmentation_config = None

    config.model.load_bf16 = False
    config.model.reproject_vision = False
    config.model.model_name = "nvidia/Cosmos-Reason2-2B"
    config.model.backbone_trainable_params_fp32 = True
    # The latent A2A processor builds its absolute canonical trajectories from
    # raw states/actions before the legacy relative-action transform. Keep the
    # old launcher's behaviour only for the original N1.7 model.
    config.model.use_relative_action = not is_a2a

    config.training.experiment_name = ft_config.experiment_name
    config.training.start_from_checkpoint = ft_config.base_model_path
    config.training.optim = "adamw_torch"
    config.training.global_batch_size = ft_config.global_batch_size
    config.training.dataloader_num_workers = ft_config.dataloader_num_workers
    config.training.learning_rate = ft_config.learning_rate
    config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
    config.training.output_dir = ft_config.output_dir
    config.training.save_steps = ft_config.save_steps
    config.training.save_total_limit = ft_config.save_total_limit
    config.training.num_gpus = ft_config.num_gpus
    config.training.use_wandb = ft_config.use_wandb
    config.training.max_steps = ft_config.max_steps
    config.training.weight_decay = ft_config.weight_decay
    config.training.warmup_ratio = ft_config.warmup_ratio
    config.training.wandb_project = ft_config.wandb_project

    config.data.shard_size = ft_config.shard_size
    config.data.episode_sampling_rate = ft_config.episode_sampling_rate
    config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch
    config.data.ds_weights_alpha = ft_config.ds_weights_alpha

    config.training.save_only_model = ft_config.save_only_model
    config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
    config.training.skip_weight_loading = ft_config.skip_weight_loading

    run(config)
