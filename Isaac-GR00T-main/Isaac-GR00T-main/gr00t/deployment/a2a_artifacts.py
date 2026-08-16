# SPDX-License-Identifier: Apache-2.0

"""Deterministic identities for latent-A2A deployment artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    """Hash a JSON-compatible payload independently of key order/whitespace."""

    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def checkpoint_sha256(checkpoint_directory: str | Path) -> str:
    """Bind a deployment graph to the exact checkpoint config and weight files.

    Optimizer/RNG state is intentionally excluded because it is not consumed by
    ``AutoModel.from_pretrained``.  Relative file names are included in the
    digest, so replacing or renaming one shard changes the identity.
    """

    checkpoint_directory = Path(checkpoint_directory).resolve()
    if not checkpoint_directory.is_dir():
        raise FileNotFoundError(f"A2A checkpoint directory does not exist: {checkpoint_directory}")

    files: set[Path] = set()
    config_path = checkpoint_directory / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"A2A checkpoint is missing {config_path}")
    files.add(config_path)
    for pattern in ("*.safetensors", "pytorch_model*.bin", "model*.bin"):
        files.update(path for path in checkpoint_directory.rglob(pattern) if path.is_file())
    for name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
        index_path = checkpoint_directory / name
        if index_path.is_file():
            files.add(index_path)
    weight_files = [
        path
        for path in files
        if path.suffix in {".safetensors", ".bin"}
    ]
    if not weight_files:
        raise FileNotFoundError(
            "A2A checkpoint contains no .safetensors or model .bin weight files"
        )

    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(checkpoint_directory).as_posix()):
        relative = path.relative_to(checkpoint_directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def canonical_statistics_sha256(processor_or_payload: Any) -> str:
    """Hash the complete canonical-statistics payload used by the processor."""

    payload = getattr(processor_or_payload, "a2a_canonical_statistics", processor_or_payload)
    if not isinstance(payload, dict) or not payload:
        raise ValueError("A2A canonical statistics are absent or empty")
    return canonical_json_sha256(payload)


__all__ = [
    "canonical_json_sha256",
    "canonical_statistics_sha256",
    "checkpoint_sha256",
    "sha256_file",
]
