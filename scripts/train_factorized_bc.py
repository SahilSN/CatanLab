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

from catanlab.rl_model import (
    FactorizedCatanActorCritic,
    mask_policy_logits,
)


def load_dataset(path: Path):
    data = np.load(path)

    return TensorDataset(
        torch.tensor(
            data["observations"],
            dtype=torch.float32,
        ),
        torch.tensor(
            data["legal_masks"],
            dtype=torch.bool,
        ),
        torch.tensor(
            data["action_ids"],
            dtype=torch.long,
        ),
    )


def action_type_and_parameter(
    action_ids,
):
    model = (
        FactorizedCatanActorCritic
    )

    action_types = torch.empty_like(
        action_ids
    )

    parameters = torch.full_like(
        action_ids,
        -1,
    )

    pass_mask = (
        action_ids == model.PASS_START
    )

    settlement_mask = (
        (action_ids >= model.SETTLEMENT_START)
        & (
            action_ids
            < model.SETTLEMENT_START
            + model.SETTLEMENT_COUNT
        )
    )

    city_mask = (
        (action_ids >= model.CITY_START)
        & (
            action_ids
            < model.CITY_START
            + model.CITY_COUNT
        )
    )

    road_mask = (
        (action_ids >= model.ROAD_START)
        & (
            action_ids
            < model.ROAD_START
            + model.ROAD_COUNT
        )
    )

    dev_mask = (
        action_ids == model.DEV_START
    )

    trade_mask = (
        (action_ids >= model.TRADE_START)
        & (
            action_ids
            < model.TRADE_START
            + model.TRADE_COUNT
        )
    )

    action_types[
        pass_mask
    ] = model.TYPE_PASS

    action_types[
        settlement_mask
    ] = model.TYPE_SETTLEMENT

    action_types[
        city_mask
    ] = model.TYPE_CITY

    action_types[
        road_mask
    ] = model.TYPE_ROAD

    action_types[
        dev_mask
    ] = model.TYPE_DEV

    action_types[
        trade_mask
    ] = model.TYPE_TRADE

    parameters[
        settlement_mask
    ] = (
        action_ids[
            settlement_mask
        ]
        - model.SETTLEMENT_START
    )

    parameters[
        city_mask
    ] = (
        action_ids[
            city_mask
        ]
        - model.CITY_START
    )

    parameters[
        road_mask
    ] = (
        action_ids[
            road_mask
        ]
        - model.ROAD_START
    )

    parameters[
        trade_mask
    ] = (
        action_ids[
            trade_mask
        ]
        - model.TRADE_START
    )

    return (
        action_types,
        parameters,
    )


def parameter_loss(
    settlement_logits,
    city_logits,
    road_logits,
    trade_logits,
    action_types,
    parameters,
):
    model = (
        FactorizedCatanActorCritic
    )

    losses = []

    for (
        type_id,
        logits,
    ) in (
        (
            model.TYPE_SETTLEMENT,
            settlement_logits,
        ),
        (
            model.TYPE_CITY,
            city_logits,
        ),
        (
            model.TYPE_ROAD,
            road_logits,
        ),
        (
            model.TYPE_TRADE,
            trade_logits,
        ),
    ):
        mask = (
            action_types == type_id
        )

        if not mask.any():
            continue

        losses.append(
            nn.functional.cross_entropy(
                logits[mask],
                parameters[mask],
            )
        )

    if not losses:
        return torch.tensor(
            0.0,
            device=action_types.device,
        )

    return torch.stack(
        losses
    ).mean()


@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
):
    model.eval()

    total = 0
    correct = 0

    pass_total = 0
    pass_correct = 0

    nonpass_total = 0
    nonpass_correct = 0

    type_total = 0
    type_correct = 0

    total_loss = 0.0

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

        (
            type_logits,
            settlement_logits,
            city_logits,
            road_logits,
            trade_logits,
            _,
        ) = model.policy_components(
            observations
        )

        (
            target_types,
            parameters,
        ) = action_type_and_parameter(
            targets
        )

        type_loss = (
            nn.functional.cross_entropy(
                type_logits,
                target_types,
            )
        )

        param_loss = parameter_loss(
            settlement_logits,
            city_logits,
            road_logits,
            trade_logits,
            target_types,
            parameters,
        )

        loss = (
            type_loss
            + param_loss
        )

        flat_logits, _ = model(
            observations
        )

        flat_logits = mask_policy_logits(
            flat_logits,
            masks,
        )

        predictions = (
            flat_logits.argmax(
                dim=-1
            )
        )

        predicted_types, _ = (
            action_type_and_parameter(
                predictions
            )
        )

        matches = (
            predictions == targets
        )

        batch_size = len(targets)

        total += batch_size

        total_loss += (
            loss.item()
            * batch_size
        )

        correct += int(
            matches.sum().item()
        )

        is_pass = (
            targets == 0
        )

        pass_total += int(
            is_pass.sum().item()
        )

        pass_correct += int(
            (
                matches
                & is_pass
            ).sum().item()
        )

        is_nonpass = ~is_pass

        nonpass_total += int(
            is_nonpass.sum().item()
        )

        nonpass_correct += int(
            (
                matches
                & is_nonpass
            ).sum().item()
        )

        type_total += batch_size

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
            pass_correct
            / pass_total
            if pass_total
            else 0.0
        ),
        "non_pass_accuracy": (
            nonpass_correct
            / nonpass_total
            if nonpass_total
            else 0.0
        ),
        "action_type_accuracy": (
            type_correct
            / type_total
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
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
        "--parameter-weight",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
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

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    model = FactorizedCatanActorCritic(
        observation_dim=1138,
        action_dim=202,
        hidden_dim=256,
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
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

            (
                type_logits,
                settlement_logits,
                city_logits,
                road_logits,
                trade_logits,
                _,
            ) = model.policy_components(
                observations
            )

            (
                target_types,
                parameters,
            ) = action_type_and_parameter(
                targets
            )

            type_loss = (
                nn.functional.cross_entropy(
                    type_logits,
                    target_types,
                )
            )

            param_loss = parameter_loss(
                settlement_logits,
                city_logits,
                road_logits,
                trade_logits,
                target_types,
                parameters,
            )

            loss = (
                type_loss
                + args.parameter_weight
                * param_loss
            )

            loss.backward()

            optimizer.step()

            batch_size = len(
                targets
            )

            running_loss += (
                loss.item()
                * batch_size
            )

            rows_seen += batch_size

        train_loss = (
            running_loss / rows_seen
        )

        metrics = evaluate(
            model,
            val_loader,
            device,
        )

        print(
            f"epoch={epoch:02d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={metrics['loss']:.4f} "
            f"val_acc={metrics['accuracy']:.3%} "
            f"val_pass="
            f"{metrics['pass_accuracy']:.3%} "
            f"val_nonpass="
            f"{metrics['non_pass_accuracy']:.3%} "
            f"val_type="
            f"{metrics['action_type_accuracy']:.3%}",
            flush=True,
        )

        if metrics["loss"] < best_val_loss:
            best_val_loss = (
                metrics["loss"]
            )

            best_epoch = epoch

            torch.save(
                {
                    "model_state_dict": (
                        model.state_dict()
                    ),
                    "model_class": (
                        "factorized"
                    ),
                    "observation_dim": 1138,
                    "action_dim": 202,
                    "hidden_dim": 256,
                    "epoch": epoch,
                    "val_metrics": metrics,
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
