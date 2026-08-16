# SPDX-License-Identifier: Apache-2.0

"""Canonical-space tests for the latent A2A processor.

The fixture copies a convenient seven-key layout into ``new_embodiment``. Its
numeric values are synthetic, *already-canonical* scalar trajectories. These
tests deliberately do not claim that raw LIBERO absolute EEF observations and
OSC delta commands share a physical space.
"""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from gr00t.data.a2a_contract import canonical_contract_sha256, canonical_statistics_sha256
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import MessageType, VLAStepData
import numpy as np
import pytest


FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "processor_config"
EMBODIMENT = EmbodimentTag.NEW_EMBODIMENT.value


def _channel_specs():
    specs = []
    for key, unit in zip(
        ("x", "y", "z", "roll", "pitch", "yaw"),
        ("meter", "meter", "meter", "radian", "radian", "radian"),
        strict=True,
    ):
        specs.append(
            {
                "action_key": key,
                "source_state_key": key,
                "kind": "continuous",
                "canonical_format": "scalar",
                "source_format": "scalar",
                "target_format": "scalar",
                "source_unit": unit,
                "target_unit": unit,
                "source_frame": "robot_base",
                "target_frame": "robot_base",
            }
        )
    specs.append(
        {
            "action_key": "gripper",
            "source_state_key": "gripper",
            "kind": "regression",
            "canonical_format": "scalar",
        }
    )
    return {EMBODIMENT: specs}


def _canonical_statistics(statistics):
    feature_dim = 128
    lower = np.zeros(feature_dim, dtype=np.float32)
    upper = np.zeros(feature_dim, dtype=np.float32)
    active = np.zeros(feature_dim, dtype=bool)
    for cursor, key in enumerate(("x", "y", "z", "roll", "pitch", "yaw")):
        state = statistics[EMBODIMENT]["state"][key]
        action = statistics[EMBODIMENT]["action"][key]
        lower[cursor] = min(state["min"][0], action["min"][0])
        upper[cursor] = max(state["max"][0], action["max"][0])
        active[cursor] = True
    result = {
        "version": 1,
        "normalization": "shared_history_future_continuous_intersection",
        "input_space": "unnormalized_canonical_physical",
        "feature_dim": feature_dim,
        "history_horizon": 8,
        "future_horizon": 8,
        "num_windows": 1,
        "nonfinite_active_count": 0,
        "mean": np.zeros(feature_dim, dtype=np.float32).tolist(),
        "std": np.ones(feature_dim, dtype=np.float32).tolist(),
        "min": lower.tolist(),
        "max": upper.tolist(),
        "q01": lower.tolist(),
        "q99": upper.tolist(),
        "q_low": lower.tolist(),
        "q_high": upper.tolist(),
        "active_channel_mask": active.tolist(),
        "valid_count": np.where(active, 16, 0).tolist(),
        "history_valid_count": np.where(active, 8, 0).tolist(),
        "future_valid_count": np.where(active, 8, 0).tolist(),
        "constant_mask": np.zeros(feature_dim, dtype=bool).tolist(),
    }
    channels = []
    cursor = 0
    for spec in _channel_specs()[EMBODIMENT]:
        dim = len(statistics[EMBODIMENT]["action"][spec["action_key"]]["mean"])
        channels.append(
            {
                "action_key": spec["action_key"],
                "source_state_key": spec.get("source_state_key"),
                "kind": spec["kind"],
                "start": cursor,
                "end": cursor + dim,
                "canonical_format": spec.get("canonical_format", "auto"),
                "source_format": spec.get("source_format"),
                "target_format": spec.get("target_format"),
                "source_unit": spec.get("source_unit"),
                "target_unit": spec.get("target_unit"),
                "source_frame": spec.get("source_frame"),
                "target_frame": spec.get("target_frame"),
            }
        )
        cursor += dim
    result["channel_contract"] = {
        "version": 1,
        "embodiment": EMBODIMENT,
        "feature_dim": feature_dim,
        "history_horizon": 8,
        "future_horizon": 8,
        "channels": channels,
        "provenance": {
            "dataset": "synthetic_processor_fixture",
            "dataset_revision": "1",
            "dataset_fingerprint_sha256": "0" * 64,
            "source_schema_sha256": "1" * 64,
            "canonicalizer": "identity_test_only",
            "canonicalizer_version": "1",
            "canonicalizer_sha256": "2" * 64,
            "exporter_version": "1",
            "target_definition": "absolute_physical_target",
            "time_alignment": "history_t_minus_7_to_t__future_t_to_t_plus_7",
        },
        "controller": {
            "type": "preprocessed_absolute",
            "version": "1",
            "frame": "robot_base",
            "control_delta": False,
            "rotation_composition": "none_scalar_fixture",
            "scale": [1.0],
        },
    }
    result["channel_contract"]["sha256"] = canonical_contract_sha256(
        result["channel_contract"]
    )
    result["statistics_sha256"] = canonical_statistics_sha256(result)
    return result


@pytest.fixture
def processor_and_stats():
    from gr00t.model.gr00t_n1d7_a2a.processing_gr00t_n1d7_a2a import Gr00tN1d7A2AProcessor

    with (FIXTURE_DIR / "processor_config.json").open(encoding="utf-8") as handle:
        kwargs = json.load(handle)["processor_kwargs"]
    with (FIXTURE_DIR / "statistics.json").open(encoding="utf-8") as handle:
        fixture_statistics = json.load(handle)
    statistics = {EMBODIMENT: deepcopy(fixture_statistics["libero_sim"])}
    modality_configs = {
        EMBODIMENT: deepcopy(kwargs["modality_configs"]["libero_sim"])
    }
    modality_configs[EMBODIMENT]["state"]["delta_indices"] = list(range(-7, 1))
    modality_configs[EMBODIMENT]["action"]["delta_indices"] = list(range(8))
    mock_vlm = MagicMock()
    mock_vlm.tokenizer.padding_side = "left"
    mock_vlm.apply_chat_template.return_value = "mock"
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        canonical_statistics = _canonical_statistics(statistics)
        processor = Gr00tN1d7A2AProcessor(
            modality_configs=modality_configs,
            statistics=statistics,
            max_state_dim=kwargs["max_state_dim"],
            max_action_dim=kwargs["max_action_dim"],
            max_action_horizon=8,
            use_percentiles=kwargs["use_percentiles"],
            use_relative_action=kwargs["use_relative_action"],
            apply_sincos_state_encoding=kwargs["apply_sincos_state_encoding"],
            image_crop_size=[224, 224],
            image_target_size=[224, 224],
            use_albumentations=False,
            a2a_canonical_statistics=canonical_statistics,
            a2a_require_canonical_statistics=True,
            a2a_channel_specs=_channel_specs(),
            a2a_require_explicit_channel_specs=True,
            a2a_require_semantic_metadata=True,
            a2a_expected_contract_sha256=canonical_statistics["channel_contract"]["sha256"],
        )
    processor.eval()
    return processor, statistics


def _midpoint(stat):
    return (np.asarray(stat["min"], dtype=np.float32) + np.asarray(stat["max"], dtype=np.float32)) / 2


def _step(processor, statistics):
    modality = processor.modality_configs[EMBODIMENT]
    states = {
        key: np.repeat(
            _midpoint(statistics[EMBODIMENT]["state"][key])[None], 8, axis=0
        )
        for key in modality["state"].modality_keys
    }
    actions = {
        key: np.repeat(
            _midpoint(statistics[EMBODIMENT]["action"][key])[None], 8, axis=0
        )
        for key in modality["action"].modality_keys
    }
    images = {
        key: [np.zeros((16, 16, 3), dtype=np.uint8)]
        for key in modality["video"].modality_keys
    }
    return VLAStepData(
        images=images,
        states=states,
        actions=actions,
        text="do the task",
        embodiment=EmbodimentTag(EMBODIMENT),
    )


def test_builds_shared_canonical_history_and_future(processor_and_stats):
    processor, statistics = processor_and_stats
    step = _step(processor, statistics)
    result = processor._build_a2a_inputs(step)
    assert result["history_action_canonical"].shape == (8, 128)
    assert result["future_action_canonical"].shape == (8, 128)
    # Six synthetic already-canonical scalar groups map one-to-one. The
    # dimension-mismatched auxiliary fixture is excluded from the latent.
    assert result["continuous_action_mask"].sum().item() == 8 * 6
    assert result["auxiliary_action_mask"].sum().item() == 8
    assert result["history_action_mask"].sum().item() == 8 * 6


def test_continuous_absolute_pose_to_delta_command_semantics_are_rejected(
    processor_and_stats,
):
    processor, statistics = processor_and_stats
    mismatched = _channel_specs()
    mismatched[EMBODIMENT][0].update(
        {
            "source_format": "absolute_eef_position",
            "target_format": "normalized_osc_delta_command",
            "source_unit": "meter",
            "target_unit": "dimensionless",
        }
    )
    processor._configured_channel_specs = mismatched

    with pytest.raises(ValueError, match="mismatched (format|unit)"):
        processor._resolve_specs(EMBODIMENT, statistics)


def test_strict_processor_rejects_known_raw_libero_keys_even_if_metadata_lies(
    processor_and_stats,
):
    processor, statistics = processor_and_stats
    raw_embodiment = EmbodimentTag.LIBERO_PANDA.value
    processor.modality_configs[raw_embodiment] = deepcopy(
        processor.modality_configs[EMBODIMENT]
    )
    processor._configured_channel_specs[raw_embodiment] = _channel_specs()[EMBODIMENT]
    raw_statistics = dict(statistics)
    raw_statistics[raw_embodiment] = deepcopy(statistics[EMBODIMENT])

    with pytest.raises(ValueError, match="Raw LIBERO state columns"):
        processor._resolve_specs(raw_embodiment, raw_statistics)


def test_same_physical_value_has_same_shared_normalization(processor_and_stats):
    processor, _ = processor_and_stats
    params = processor.a2a_norm_params[EMBODIMENT]["x"]
    value = ((params["min"] + params["max"]) / 2)[None]
    history_value = processor._normalize(value, EMBODIMENT, "x")
    future_value = processor._normalize(value, EMBODIMENT, "x")
    np.testing.assert_allclose(history_value, future_value)


def test_policy_cold_start_valid_mask_reaches_encoder_contract(processor_and_stats):
    processor, statistics = processor_and_stats
    step = _step(processor, statistics)
    step.metadata["a2a_history_valid_mask"] = np.array(
        [0, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32
    )
    result = processor._build_a2a_inputs(step)
    assert result["history_action_mask"][:-1].sum().item() == 0
    assert result["history_action_mask"][-1].sum().item() == 6


def test_full_call_keeps_current_state_and_emits_a2a_contract(processor_and_stats):
    processor, statistics = processor_and_stats
    step = _step(processor, statistics)
    with patch.object(processor, "_get_vlm_inputs", return_value={"vlm_content": {}}):
        result = processor(
            [{"type": MessageType.EPISODE_STEP.value, "content": step}]
        )
    assert result["state"].shape[0] == 1
    assert "action" not in result
    assert "history_action_canonical" in result
    assert "future_action_canonical" in result


def test_canonical_decode_round_trip(processor_and_stats):
    processor, statistics = processor_and_stats
    step = _step(processor, statistics)
    encoded = processor._build_a2a_inputs(step)["future_action_canonical"].numpy()
    decoded = processor.decode_action(encoded, EmbodimentTag(EMBODIMENT))
    for key, expected in step.actions.items():
        np.testing.assert_allclose(decoded[key][0], expected[0], atol=1e-5)


def test_processor_save_load_embeds_canonical_statistics(processor_and_stats, tmp_path):
    processor, _ = processor_and_stats
    processor.save_pretrained(tmp_path)
    saved = json.loads((tmp_path / "processor_config.json").read_text(encoding="utf-8"))
    assert saved["processor_kwargs"]["a2a_canonical_statistics"]["feature_dim"] == 128

    mock_vlm = MagicMock()
    mock_vlm.tokenizer.padding_side = "left"
    with patch(
        "gr00t.model.gr00t_n1d7.processing_gr00t_n1d7.build_processor",
        return_value=mock_vlm,
    ):
        loaded = type(processor).from_pretrained(tmp_path)
    np.testing.assert_allclose(
        loaded.a2a_norm_params[EMBODIMENT]["x"]["min"],
        processor.a2a_norm_params[EMBODIMENT]["x"]["min"],
    )
