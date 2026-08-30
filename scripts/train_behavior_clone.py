from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)

from catanlab.action_space import (
    build_action_vocabulary,
)
from catanlab.board import (
    build_random_board,
)
from catanlab.rl_model import (
    CatanActorCritic,
    mask_policy_logits,
)


def build_type_weights(
    train_dataset,
    action_type_ids,
):
    """
    Build moderate inverse-square-root weights by
    action type.

    This reduces PASS dominance without making rare
    action types overwhelmingly expensive.
    """
    targets = train_dataset.tensors[2]

    target_types = action_type_ids[
        targets
    ]

    num_types = int(
        action_type_ids.max().item()
    ) + 1

    counts = torch.bincount(
        target_types,
        minlength=num_types,
    ).float()

    frequencies = (
        counts / counts.sum()
    )

    weights = (
        1.0
        / torch.sqrt(
            frequencies.clamp_min(
                1e-8
            )
        )
    )

    # Keep the overall scale close to the original
    # unweighted loss.
    weights = (
        weights
        / weights.mean()
    )

    # Avoid extreme weighting for the rarest type.
    weights = torch.clamp(
        weights,
        min=0.5,
        max=2.5,
    )

    print(
        "type counts:",
        counts.tolist(),
    )

    print(
        "type weights:",
        weights.tolist(),
    )

    return weights


def load_dataset(path: Path):
    data = np.load(path)

    observations = torch.tensor(
        data["observations"],
        dtype=torch.float32,
    )

    legal_masks = torch.tensor(
        data["legal_masks"],
        dtype=torch.bool,
    )

    action_ids = torch.tensor(
        data["action_ids"],
        dtype=torch.long,
    )

    return TensorDataset(
        observations,
        legal_masks,
        action_ids,
    )


def build_action_type_ids():
    board = build_random_board(
        seed=0
    )

    vocabulary = build_action_vocabulary(
        len(board.vertices),
        board.edges,
    )

    type_names = []

    type_to_id = {}

    type_ids = []

    for action in vocabulary:
        name = action.action_type.value

        if name not in type_to_id:
            type_to_id[name] = len(
                type_to_id
            )

        type_ids.append(
            type_to_id[name]
        )

        type_names.append(name)

    return (
        torch.tensor(
            type_ids,
            dtype=torch.long,
        ),
        type_to_id,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    action_type_ids,
):
    model.eval()

    total = 0
    correct = 0

    pass_total = 0
    pass_correct = 0

    non_pass_total = 0
    non_pass_correct = 0

    type_correct = 0

    total_loss = 0.0

    loss_fn = nn.CrossEntropyLoss()

    action_type_ids = (
        action_type_ids.to(device)
    )

    for (
        observations,
        masks,
        targets,
    ) in loader:
        observations = (
            observations.to(device)
        )

        masks = masks.to(device)
        targets = targets.to(device)

        logits, _ = model(
            observations
        )

        masked_logits = (
            mask_policy_logits(
                logits,
                masks,
            )
        )

        loss = loss_fn(
            masked_logits,
            targets,
        )

        predictions = (
            masked_logits.argmax(
                dim=-1
            )
        )

        batch_size = (
            targets.shape[0]
        )

        total += batch_size

        total_loss += (
            loss.item()
            * batch_size
        )

        matches = (
            predictions == targets
        )

        correct += int(
            matches.sum().item()
        )

        # PASS is action ID 0 in the fixed vocabulary.
        is_pass = targets == 0

        pass_total += int(
            is_pass.sum().item()
        )

        pass_correct += int(
            (
                matches
                & is_pass
            ).sum().item()
        )

        is_non_pass = ~is_pass

        non_pass_total += int(
            is_non_pass.sum().item()
        )

        non_pass_correct += int(
            (
                matches
                & is_non_pass
            ).sum().item()
        )

        predicted_types = (
            action_type_ids[
                predictions
            ]
        )

        target_types = (
            action_type_ids[
                targets
            ]
        )

        type_correct += int(
            (
                predicted_types
                == target_types
            ).sum().item()
        )

    return {
        "loss": (
            total_loss / total
        ),
        "accuracy": (
            correct / total
        ),
        "pass_accuracy": (
            pass_correct / pass_total
            if pass_total
            else 0.0
        ),
        "non_pass_accuracy": (
            non_pass_correct
            / non_pass_total
            if non_pass_total
            else 0.0
        ),
        "action_type_accuracy": (
            type_correct / total
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(
            "results/rl_teacher/"
            "split_100"
        ),
    )

    parser.add_argument(
        "--dagger-val",
        type=Path,
        default=None,
        help=(
            "Optional held-out DAgger validation "
            "dataset used for checkpoint selection."
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )

    parser.add_argument(
        "--weighted",
        action="store_true",
        help=(
            "Use inverse-square-root action-type "
            "weights during training."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/rl_teacher/"
            "bc_model.pt"
        ),
    )

    args = parser.parse_args()

    torch.manual_seed(
        args.seed
    )

    np.random.seed(
        args.seed
    )

    if torch.backends.mps.is_available():
        device = torch.device(
            "mps"
        )
    else:
        device = torch.device(
            "cpu"
        )

    print(
        "device:",
        device,
    )

    train_dataset = load_dataset(
        args.data_dir / "train.npz"
    )

    val_dataset = load_dataset(
        args.data_dir / "val.npz"
    )

    dagger_val_dataset = (
        load_dataset(
            args.dagger_val
        )
        if args.dagger_val
        is not None
        else None
    )

    test_dataset = load_dataset(
        args.data_dir / "test.npz"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    dagger_val_loader = (
        DataLoader(
            dagger_val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
        )
        if dagger_val_dataset
        is not None
        else None
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = CatanActorCritic(
        observation_dim=1138,
        action_dim=202,
        hidden_dim=256,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    loss_fn = nn.CrossEntropyLoss(
        reduction="none"
    )

    (
        action_type_ids,
        _,
    ) = build_action_type_ids()

    action_type_ids_device = (
        action_type_ids.to(device)
    )

    if args.weighted:
        type_weights = build_type_weights(
            train_dataset,
            action_type_ids,
        ).to(device)
    else:
        type_weights = None

        print(
            "training loss: unweighted"
        )

    best_val_loss = float(
        "inf"
    )

    best_epoch = None

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        model.train()

        running_loss = 0.0
        rows_seen = 0

        for (
            observations,
            masks,
            targets,
        ) in train_loader:
            observations = (
                observations.to(device)
            )

            masks = masks.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            logits, _ = model(
                observations
            )

            masked_logits = (
                mask_policy_logits(
                    logits,
                    masks,
                )
            )

            per_example_loss = loss_fn(
                masked_logits,
                targets,
            )

            if args.weighted:
                target_types = (
                    action_type_ids_device[
                        targets
                    ]
                )

                example_weights = (
                    type_weights[
                        target_types
                    ]
                )

                loss = (
                    per_example_loss
                    * example_weights
                ).mean()
            else:
                loss = (
                    per_example_loss.mean()
                )

            loss.backward()

            optimizer.step()

            batch_size = (
                targets.shape[0]
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            rows_seen += (
                batch_size
            )

        train_loss = (
            running_loss
            / rows_seen
        )

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            action_type_ids,
        )

        dagger_val_metrics = (
            evaluate(
                model,
                dagger_val_loader,
                device,
                action_type_ids,
            )
            if dagger_val_loader
            is not None
            else None
        )

        if dagger_val_metrics is None:
            selection_loss = (
                val_metrics["loss"]
            )
        else:
            selection_loss = (
                0.5
                * val_metrics["loss"]
                + 0.5
                * dagger_val_metrics[
                    "loss"
                ]
            )

        print(
            f"epoch={epoch:02d} "
            f"train_loss="
            f"{train_loss:.4f} "
            f"val_loss="
            f"{val_metrics['loss']:.4f} "
            f"val_acc="
            f"{val_metrics['accuracy']:.3%} "
            f"val_pass="
            f"{val_metrics['pass_accuracy']:.3%} "
            f"val_nonpass="
            f"{val_metrics['non_pass_accuracy']:.3%} "
            f"val_type="
            f"{val_metrics['action_type_accuracy']:.3%} "
            + (
                (
                    f"dagger_loss="
                    f"{dagger_val_metrics['loss']:.4f} "
                    f"dagger_acc="
                    f"{dagger_val_metrics['accuracy']:.3%} "
                    f"dagger_nonpass="
                    f"{dagger_val_metrics['non_pass_accuracy']:.3%}"
                )
                if dagger_val_metrics
                is not None
                else ""
            ),
            flush=True,
        )

        if (
            selection_loss
            < best_val_loss
        ):
            best_val_loss = (
                selection_loss
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "observation_dim": 1138,
                    "action_dim": 202,
                    "hidden_dim": 256,
                    "epoch": epoch,
                    "val_metrics": (
                        val_metrics
                    ),
                },
                args.output,
            )

    print()
    print(
        "best epoch:",
        best_epoch,
    )

    checkpoint = torch.load(
        args.output,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        action_type_ids,
    )

    print()
    print(
        "=== TEST METRICS ==="
    )

    for key, value in (
        test_metrics.items()
    ):
        print(
            f"{key:22s} "
            f"{value:.4f}"
        )

    print()
    print(
        "saved:",
        args.output,
    )


if __name__ == "__main__":
    main()
