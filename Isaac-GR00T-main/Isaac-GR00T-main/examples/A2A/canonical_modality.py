# SPDX-License-Identifier: Apache-2.0

"""Example modality registration for an already-canonical A2A dataset.

The dataset columns named below are placeholders. They must be produced by an
offline, versioned robot/controller adapter; renaming raw controller deltas is
not canonicalization.
"""

from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


canonical_a2a_config = {
    "video": ModalityConfig(delta_indices=[0], modality_keys=["front"]),
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["eef_pose_canonical"],
    ),
    "action": ModalityConfig(
        delta_indices=list(range(8)),
        modality_keys=["eef_pose_canonical", "gripper_command"],
        action_configs=[
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.EEF,
                format=ActionFormat.XYZ_ROT6D,
                state_key="eef_pose_canonical",
            ),
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(
    canonical_a2a_config,
    embodiment_tag=EmbodimentTag.NEW_EMBODIMENT,
)
