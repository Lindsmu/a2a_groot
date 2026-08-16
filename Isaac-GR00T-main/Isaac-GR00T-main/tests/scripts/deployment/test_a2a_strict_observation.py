# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
import sys
from types import SimpleNamespace

from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
import numpy as np
import pandas as pd


_DEPLOYMENT_DIRECTORY = Path(__file__).resolve().parents[3] / "scripts" / "deployment"
if str(_DEPLOYMENT_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(_DEPLOYMENT_DIRECTORY))

from scripts.deployment.export_onnx_n1d7 import prepare_observation  # noqa: E402


def test_a2a_prepare_observation_uses_first_complete_history_anchor():
    config = {
        "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
        "state": ModalityConfig(delta_indices=list(range(-2, 1)), modality_keys=["joint"]),
        "action": ModalityConfig(delta_indices=[0, 1], modality_keys=["joint"]),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
    }
    episode = pd.DataFrame(
        {
            "video.front": [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(6)],
            "state.joint": [np.asarray([value], dtype=np.float32) for value in range(6)],
            "action.joint": [np.asarray([value + 10], dtype=np.float32) for value in range(6)],
            "language.task": ["move"] * 6,
        }
    )

    class Dataset:
        def __getitem__(self, index):
            assert index == 0
            return episode

    policy = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(model_type="Gr00tN1d7A2A")),
        embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
        get_modality_config=lambda: config,
    )
    observation = prepare_observation(policy, Dataset(), traj_idx=0)
    assert observation["state"]["joint"].reshape(-1).tolist() == [0.0, 1.0, 2.0]
