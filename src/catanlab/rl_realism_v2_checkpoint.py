from __future__ import annotations

from pathlib import Path

import torch

from catanlab.rl_model import (
    RealismV2ActorCritic,
)


REALISM_V2_TRAINING_PROTOCOL = (
    "realism-v2-bc-v1"
)


def load_realism_v2_checkpoint(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> RealismV2ActorCritic:
    path = Path(path)

    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    protocol = checkpoint.get(
        "training_protocol"
    )

    if (
        protocol
        != REALISM_V2_TRAINING_PROTOCOL
    ):
        raise ValueError(
            "Checkpoint is not a realism-v2 BC "
            "checkpoint: "
            f"training_protocol={protocol!r}"
        )

    required = (
        "observation_dim",
        "action_dim",
        "hidden_dim",
        "model_state_dict",
    )

    missing = [
        key
        for key in required
        if key not in checkpoint
    ]

    if missing:
        raise ValueError(
            "Checkpoint is missing required keys: "
            f"{missing}"
        )

    model = RealismV2ActorCritic(
        observation_dim=checkpoint[
            "observation_dim"
        ],
        action_dim=checkpoint[
            "action_dim"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)
    model.eval()

    return model
