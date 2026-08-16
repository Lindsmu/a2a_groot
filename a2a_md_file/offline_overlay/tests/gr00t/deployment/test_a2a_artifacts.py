# SPDX-License-Identifier: Apache-2.0

import json

from gr00t.deployment.a2a_artifacts import (
    canonical_json_sha256,
    canonical_statistics_sha256,
    checkpoint_sha256,
    sha256_file,
)
import pytest


def test_sha256_file_and_canonical_json_are_deterministic(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"latent-a2a")
    assert sha256_file(artifact) == sha256_file(artifact)
    assert canonical_json_sha256({"b": 2, "a": 1}) == canonical_json_sha256(
        {"a": 1, "b": 2}
    )


def test_checkpoint_sha_binds_config_weights_and_relative_name(tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text(json.dumps({"model_type": "A2A"}))
    weights = checkpoint / "model.safetensors"
    weights.write_bytes(b"weights-v1")
    first = checkpoint_sha256(checkpoint)
    weights.write_bytes(b"weights-v2")
    assert checkpoint_sha256(checkpoint) != first


def test_checkpoint_sha_requires_inference_weights(tmp_path):
    (tmp_path / "config.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="weight files"):
        checkpoint_sha256(tmp_path)


def test_canonical_statistics_sha_rejects_empty_payload():
    with pytest.raises(ValueError, match="absent or empty"):
        canonical_statistics_sha256({})
