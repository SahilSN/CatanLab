from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from catanlab.action_space import (
    build_action_vocabulary,
)
from catanlab.board import build_random_board
from catanlab.rl_model import (
    CatanActorCritic,
    FactorizedCatanActorCritic,
    mask_policy_logits,
)


def load_model(path: Path):
    checkpoint = torch.load(
        path,
        map_location="cpu",
    )

    state_dict = checkpoint[
        "model_state_dict"
    ]

    model_class = checkpoint.get(
        "model_class"
    )

    # Backward compatibility with older checkpoints
    # that did not store model_class explicitly.
    if model_class is None:
        if "type_head.weight" in state_dict:
            model_class = "factorized"
        else:
            model_class = "flat"

    if model_class == "factorized":
        model = FactorizedCatanActorCritic(
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
    elif model_class == "flat":
        model = CatanActorCritic(
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
    else:
        raise ValueError(
            "Unknown checkpoint model_class: "
            f"{model_class!r}"
        )

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--data",
        type=Path,
        required=True,
    )

    args = parser.parse_args()

    model = load_model(args.model)

    data = np.load(args.data)

    observations = torch.tensor(
        data["observations"],
        dtype=torch.float32,
    )

    masks = torch.tensor(
        data["legal_masks"],
        dtype=torch.bool,
    )

    targets = torch.tensor(
        data["action_ids"],
        dtype=torch.long,
    )

    board = build_random_board(seed=0)

    vocabulary = build_action_vocabulary(
        len(board.vertices),
        board.edges,
    )

    with torch.no_grad():
        logits, _ = model(
            observations
        )

        masked_logits = mask_policy_logits(
            logits,
            masks,
        )

        predictions = masked_logits.argmax(
            dim=-1
        )

        top3 = torch.topk(
            masked_logits,
            k=3,
            dim=-1,
        ).indices

    teacher_types = Counter()
    predicted_types = Counter()

    confusion = defaultdict(Counter)

    exact_by_type = defaultdict(
        lambda: [0, 0]
    )

    top3_correct = 0

    for i in range(len(targets)):
        target_id = int(
            targets[i].item()
        )

        pred_id = int(
            predictions[i].item()
        )

        target_type = (
            vocabulary[
                target_id
            ].action_type.value
        )

        pred_type = (
            vocabulary[
                pred_id
            ].action_type.value
        )

        teacher_types[
            target_type
        ] += 1

        predicted_types[
            pred_type
        ] += 1

        confusion[
            target_type
        ][
            pred_type
        ] += 1

        exact_by_type[
            target_type
        ][1] += 1

        if pred_id == target_id:
            exact_by_type[
                target_type
            ][0] += 1

        if target_id in [
            int(x)
            for x in top3[i]
        ]:
            top3_correct += 1

    total = len(targets)

    print(
        "=== TEST POLICY DIAGNOSTIC ==="
    )

    print(
        f"examples: {total}"
    )

    print(
        f"exact accuracy: "
        f"{float((predictions == targets).float().mean()):.3%}"
    )

    print(
        f"top-3 exact accuracy: "
        f"{top3_correct / total:.3%}"
    )

    print()
    print(
        "teacher vs predicted action-type frequency:"
    )

    action_types = [
        action_type.value
        for action_type in dict.fromkeys(
            action.action_type
            for action in vocabulary
        )
    ]

    for action_type in action_types:
        teacher_count = teacher_types[
            action_type
        ]

        predicted_count = predicted_types[
            action_type
        ]

        print(
            f"{action_type:20s} "
            f"teacher="
            f"{teacher_count:4d} "
            f"{teacher_count / total:7.2%} "
            f"pred="
            f"{predicted_count:4d} "
            f"{predicted_count / total:7.2%}"
        )

    print()
    print(
        "exact accuracy by teacher action type:"
    )

    for action_type in action_types:
        correct, count = exact_by_type[
            action_type
        ]

        accuracy = (
            correct / count
            if count
            else 0.0
        )

        print(
            f"{action_type:20s} "
            f"{correct:4d}/{count:4d} "
            f"{accuracy:7.2%}"
        )

    print()
    print(
        "action-type confusion:"
    )

    print(
        f"{'teacher':20s}",
        end="",
    )

    for predicted in action_types:
        print(
            f"{predicted[:8]:>10s}",
            end="",
        )

    print()

    for teacher in action_types:
        print(
            f"{teacher:20s}",
            end="",
        )

        for predicted in action_types:
            print(
                f"{confusion[teacher][predicted]:10d}",
                end="",
            )

        print()


if __name__ == "__main__":
    main()
