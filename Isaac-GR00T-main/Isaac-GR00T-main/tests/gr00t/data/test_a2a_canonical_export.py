# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig, VLAStepData
import numpy as np
import pandas as pd
import pytest
from scripts.a2a.export_canonical_windows import (
    collect_identity_preprocessed_windows,
    load_modality_config,
    pack_identity_preprocessed_window,
    save_identity_preprocessed_windows,
    validate_identity_preprocessed_contract,
)


def _contract(*, embodiment: str | None = None) -> dict:
    embodiment = embodiment or EmbodimentTag.NEW_EMBODIMENT.value
    return {
        "version": 1,
        "embodiment": embodiment,
        "feature_dim": 4,
        "history_horizon": 2,
        "future_horizon": 2,
        "channels": [
            {
                "action_key": "arm_canonical",
                "source_state_key": "arm_canonical",
                "kind": "continuous",
                "start": 0,
                "end": 2,
                "canonical_format": "joint_position",
                "source_format": "joint_position",
                "target_format": "joint_position",
                "source_unit": "radian",
                "target_unit": "radian",
                "source_frame": "joint",
                "target_frame": "joint",
            },
            {
                "action_key": "gripper_command",
                "source_state_key": None,
                "kind": "binary",
                "start": 2,
                "end": 3,
                "canonical_format": "scalar",
                "source_format": None,
                "target_format": None,
                "source_unit": None,
                "target_unit": None,
                "source_frame": None,
                "target_frame": None,
            },
        ],
        "provenance": {
            "dataset": "preprocessed_unit_test",
            "dataset_revision": "1",
            "dataset_fingerprint_sha256": "0" * 64,
            "source_schema_sha256": "1" * 64,
            "canonicalizer": "identity_preprocessed",
            "canonicalizer_version": "1",
            "canonicalizer_sha256": "2" * 64,
            "exporter_version": "1",
            "target_definition": "absolute_joint_target",
            "time_alignment": "history_t_minus_1_to_t__future_t_to_t_plus_1",
        },
        "controller": {
            "type": "joint_position",
            "version": "1",
            "frame": "joint",
            "control_delta": False,
            "rotation_composition": "none",
            "scale": [1.0, 1.0],
        },
    }


def _step() -> VLAStepData:
    return VLAStepData(
        images={},
        states={
            "arm_canonical": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        },
        actions={
            "arm_canonical": np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32),
            "gripper_command": np.array([[0.0], [1.0]], dtype=np.float32),
        },
        text="task",
        embodiment=EmbodimentTag.NEW_EMBODIMENT,
    )


def test_pack_preserves_unnormalized_values_and_contract_masks():
    packed = pack_identity_preprocessed_window(_step(), _contract())

    np.testing.assert_array_equal(
        packed["history"][:, :2], _step().states["arm_canonical"]
    )
    np.testing.assert_array_equal(
        packed["future"][:, :2], _step().actions["arm_canonical"]
    )
    np.testing.assert_array_equal(
        packed["future"][:, 2:3], _step().actions["gripper_command"]
    )
    assert packed["history_mask"][:, :2].all()
    assert not packed["history_mask"][:, 2:].any()
    assert packed["future_mask"][:, :3].all()
    assert not packed["future_mask"][:, 3:].any()


def test_contract_rejects_nonidentity_and_raw_libero_pseudo_canonical():
    nonidentity = _contract()
    nonidentity["provenance"]["canonicalizer"] = "osc_delta_to_absolute"
    with pytest.raises(ValueError, match="only supports"):
        validate_identity_preprocessed_contract(nonidentity)

    raw_libero = _contract(embodiment="libero_sim")
    raw_libero["feature_dim"] = 1
    raw_libero["channels"] = [
        {
            "action_key": "x",
            "source_state_key": "x",
            "kind": "continuous",
            "start": 0,
            "end": 1,
            "canonical_format": "scalar",
            "source_format": "scalar",
            "target_format": "scalar",
            "source_unit": "meter",
            "target_unit": "meter",
            "source_frame": "world",
            "target_frame": "world",
        }
    ]
    with pytest.raises(ValueError, match="Raw LIBERO action keys"):
        validate_identity_preprocessed_contract(raw_libero)


def test_pack_rejects_nonfinite_and_nonbinary_active_values():
    nonfinite = _step()
    nonfinite.states["arm_canonical"][0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        pack_identity_preprocessed_window(nonfinite, _contract())

    nonbinary = _step()
    nonbinary.actions["gripper_command"][0, 0] = 0.5
    with pytest.raises(ValueError, match="only 0/1"):
        pack_identity_preprocessed_window(nonbinary, _contract())


class _FakeEpisodeLoader:
    def __init__(self, episode: pd.DataFrame):
        self.episode = episode
        self.episode_lengths = [len(episode)]

    def __getitem__(self, index: int) -> pd.DataFrame:
        assert index == 0
        return self.episode


def test_collect_traverses_every_strict_anchor_and_embeds_provenance(tmp_path: Path):
    episode = pd.DataFrame(
        {
            "state.arm_canonical": [
                np.array([index, index + 0.1], dtype=np.float32) for index in range(4)
            ],
            "action.arm_canonical": [
                np.array([index + 0.2, index + 0.3], dtype=np.float32)
                for index in range(4)
            ],
            "action.gripper_command": [
                np.array([index % 2], dtype=np.float32) for index in range(4)
            ],
            "language.task": ["task"] * 4,
        }
    )
    modality_configs = {
        "state": ModalityConfig(delta_indices=[-1, 0], modality_keys=["arm_canonical"]),
        "action": ModalityConfig(
            delta_indices=[0, 1], modality_keys=["arm_canonical", "gripper_command"]
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    dataset = SimpleNamespace(
        episode_loader=_FakeEpisodeLoader(episode),
        modality_configs=modality_configs,
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )

    arrays = collect_identity_preprocessed_windows(dataset, _contract())
    assert arrays["history"].shape == (2, 2, 4)
    assert arrays["future"].shape == (2, 2, 4)
    np.testing.assert_array_equal(arrays["episode_index"], [0, 0])
    np.testing.assert_array_equal(arrays["base_step_index"], [1, 2])

    output_path = save_identity_preprocessed_windows(
        arrays, _contract(), tmp_path / "canonical_windows.npz"
    )
    with np.load(output_path, allow_pickle=False) as payload:
        assert payload["input_space"].item() == "unnormalized_canonical_physical"
        assert payload["canonicalizer"].item() == "identity_preprocessed"
        assert len(payload["contract_sha256"].item()) == 64
        saved_contract = json.loads(payload["contract_json"].item())
        assert saved_contract["provenance"]["dataset"] == "preprocessed_unit_test"


def test_collect_rejects_noncontiguous_timeline_even_when_shapes_match():
    episode = pd.DataFrame(
        {
            "state.arm_canonical": [np.zeros(2, dtype=np.float32)] * 4,
            "action.arm_canonical": [np.zeros(2, dtype=np.float32)] * 4,
            "action.gripper_command": [np.zeros(1, dtype=np.float32)] * 4,
            "language.task": ["task"] * 4,
        }
    )
    dataset = SimpleNamespace(
        episode_loader=_FakeEpisodeLoader(episode),
        modality_configs={
            "state": ModalityConfig(
                delta_indices=[-2, 0], modality_keys=["arm_canonical"]
            ),
            "action": ModalityConfig(
                delta_indices=[0, 1],
                modality_keys=["arm_canonical", "gripper_command"],
            ),
            "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        },
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
    )

    with pytest.raises(ValueError, match="exact contract timeline"):
        collect_identity_preprocessed_windows(dataset, _contract())


def test_save_rejects_mask_layout_that_disagrees_with_contract(tmp_path: Path):
    packed = pack_identity_preprocessed_window(_step(), _contract())
    arrays = {key: value[None, ...] for key, value in packed.items()}
    arrays["episode_index"] = np.array([0], dtype=np.int64)
    arrays["base_step_index"] = np.array([1], dtype=np.int64)
    arrays["history_mask"][0, 0, 2] = 1

    with pytest.raises(ValueError, match="continuous channel layout"):
        save_identity_preprocessed_windows(
            arrays, _contract(), tmp_path / "invalid_windows.npz"
        )


def test_load_modality_config_uses_explicit_local_python_registration(tmp_path: Path):
    module_name = f"a2a_export_config_{abs(hash(str(tmp_path)))}"
    registry_key = f"{module_name}_embodiment"
    config_path = tmp_path / f"{module_name}.py"
    config_path.write_text(
        "from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS\n"
        f"MODALITY_CONFIGS[{registry_key!r}] = {{'registered': True}}\n",
        encoding="utf-8",
    )
    try:
        assert load_modality_config(config_path) == config_path
        assert MODALITY_CONFIGS[registry_key] == {"registered": True}
    finally:
        MODALITY_CONFIGS.pop(registry_key, None)
        sys.modules.pop(module_name, None)
        if str(tmp_path) in sys.path:
            sys.path.remove(str(tmp_path))
