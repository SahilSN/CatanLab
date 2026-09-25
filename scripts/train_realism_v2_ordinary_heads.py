from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from catanlab.rl_model import (
    RealismV2ActorCritic,
)
from catanlab.rl_teacher_training import (
    evaluate_teacher_v2_batches,
    iter_shuffled_teacher_v2_batches,
    iter_teacher_v2_batches,
    train_teacher_v2_batches,
)


ORDINARY_HEAD_PREFIXES = (
    "type_head.",
    "settlement_head.",
    "city_head.",
    "road_head.",
    "trade_head.",
)

BACKBONE_PREFIX = "backbone."


def choose_device(value):
    if value != "auto":
        return torch.device(value)

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


def configure_ordinary_only(
    model,
):
    trainable = []
    frozen = []

    for name, parameter in (
        model.named_parameters()
    ):
        should_train = (
            name.startswith(
                BACKBONE_PREFIX
            )
            or any(
                name.startswith(prefix)
                for prefix in (
                    ORDINARY_HEAD_PREFIXES
                )
            )
        )

        parameter.requires_grad = (
            should_train
        )

        if should_train:
            trainable.append(name)
        else:
            frozen.append(name)

    if not trainable:
        raise RuntimeError(
            "No ordinary-policy parameters "
            "were made trainable."
        )

    return trainable, frozen


def count_trainable(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def spread_extra_streams(
    base_stream,
    extra_streams,
    *,
    extra_every,
):
    """
    Emit one DAgger batch after approximately every
    `extra_every` base batches.

    This keeps the small on-policy corpus distributed
    throughout the much larger frozen teacher epoch
    instead of front-loading it.
    """
    if extra_every <= 0:
        raise ValueError(
            "extra_every must be positive."
        )

    extra_stream = itertools.chain.from_iterable(
        extra_streams
    )

    extra_exhausted = False
    base_since_extra = 0

    for batch in base_stream:
        yield batch
        base_since_extra += 1

        if (
            not extra_exhausted
            and base_since_extra >= extra_every
        ):
            try:
                yield next(extra_stream)
            except StopIteration:
                extra_exhausted = True

            base_since_extra = 0

    if not extra_exhausted:
        yield from extra_stream

def evaluate(
    model,
    path,
    batch_size,
    device,
):
    return (
        evaluate_teacher_v2_batches(
            model,
            iter_teacher_v2_batches(
                path,
                batch_size=batch_size,
                drop_last=False,
            ),
            device,
        )
    )



def selection_metric(
    metrics,
):
    """
    Use unweighted macro accuracy across decision
    families so frequent families cannot dominate
    checkpoint selection.
    """
    return metrics["macro_accuracy"]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--base-train",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--extra-train",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--extra-repeat",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--extra-every",
        type=int,
        default=27,
        help=(
            "Insert one extra/DAgger batch after this "
            "many base batches."
        ),
    )

    parser.add_argument(
        "--validation",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
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
        help="Learning rate for ordinary-action heads.",
    )

    parser.add_argument(
        "--backbone-lr",
        type=float,
        default=1e-4,
        help="Learning rate for the shared backbone.",
    )

    parser.add_argument(
        "--shuffle-buffer-batches",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--device",
        default="auto",
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if args.extra_repeat < 0:
        raise ValueError(
            "--extra-repeat cannot be negative."
        )

    if (
        args.extra_repeat > 0
        and args.extra_train is None
    ):
        raise ValueError(
            "--extra-train is required when "
            "--extra-repeat > 0."
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_path = (
        args.output_dir / "best.pt"
    )

    latest_path = (
        args.output_dir / "latest.pt"
    )

    history_path = (
        args.output_dir / "history.json"
    )

    checkpoint = torch.load(
        args.init_checkpoint,
        map_location="cpu",
        weights_only=False,
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

    trainable, frozen = (
        configure_ordinary_only(
            model
        )
    )

    device = choose_device(
        args.device
    )

    model.to(device)

    backbone_parameters = []
    ordinary_head_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):
        if not parameter.requires_grad:
            continue

        if name.startswith(
            BACKBONE_PREFIX
        ):
            backbone_parameters.append(
                parameter
            )
        else:
            ordinary_head_parameters.append(
                parameter
            )

    optimizer = torch.optim.Adam(
        [
            {
                "params": (
                    ordinary_head_parameters
                ),
                "lr": args.lr,
            },
            {
                "params": (
                    backbone_parameters
                ),
                "lr": args.backbone_lr,
            },
        ]
    )

    print(
        f"device: {device}",
        flush=True,
    )

    print(
        "trainable parameter tensors:",
        len(trainable),
        flush=True,
    )

    print(
        "trainable parameters:",
        count_trainable(model),
        flush=True,
    )

    for name in trainable:
        print(
            f"  TRAIN {name}",
            flush=True,
        )

    print(
        f"frozen parameter tensors: "
        f"{len(frozen)}",
        flush=True,
    )

    initial = evaluate(
        model,
        args.validation,
        args.batch_size,
        device,
    )

    initial_score = (
        initial["macro_accuracy"]
    )

    print(
        "initial validation: "
        f"overall="
        f"{initial['overall']['accuracy']:.4f} "
        f"macro={initial_score:.4f}",
        flush=True,
    )

    best_score = initial_score

    history = {
        "initial_validation": initial,
        "epochs": [],
    }

    initial_out = dict(
        checkpoint
    )

    initial_out[
        "ordinary_head_finetune"
    ] = {
        "epoch": 0,
        "best_macro_accuracy": (
            best_score
        ),
        "extra_repeat": (
            args.extra_repeat
        ),
    }

    torch.save(
        initial_out,
        best_path,
    )

    for epoch in range(
        1,
        args.epochs + 1
    ):
        base = (
            iter_shuffled_teacher_v2_batches(
                args.base_train,
                batch_size=(
                    args.batch_size
                ),
                shuffle_buffer_batches=(
                    args.shuffle_buffer_batches
                ),
                seed=(
                    args.seed
                    + epoch * 1000
                ),
                drop_last=False,
            )
        )

        extra_streams = []

        for repeat in range(
            args.extra_repeat
        ):
            extra_streams.append(
                iter_shuffled_teacher_v2_batches(
                    args.extra_train,
                    batch_size=(
                        args.batch_size
                    ),
                    shuffle_buffer_batches=(
                        args.shuffle_buffer_batches
                    ),
                    seed=(
                        args.seed
                        + epoch * 10000
                        + repeat
                    ),
                    drop_last=False,
                )
            )

        batches = (
            base
            if not extra_streams
            else spread_extra_streams(
                base,
                extra_streams,
                extra_every=(
                    args.extra_every
                ),
            )
        )

        metrics = (
            train_teacher_v2_batches(
                model,
                batches,
                optimizer,
                device,
                family_loss_weights=None,
                log_every=args.log_every,
            )
        )

        validation = evaluate(
            model,
            args.validation,
            args.batch_size,
            device,
        )

        score = (
            validation["macro_accuracy"]
        )

        print(
            f"epoch {epoch}: "
            f"train_acc="
            f"{metrics['overall']['accuracy']:.4f} "
            f"train_macro="
            f"{metrics['macro_accuracy']:.4f} "
            f"val_acc="
            f"{validation['overall']['accuracy']:.4f} "
            f"val_macro={score:.4f}",
            flush=True,
        )

        improved = (
            score > best_score
        )

        if improved:
            best_score = score

        out = dict(
            checkpoint
        )

        out[
            "model_state_dict"
        ] = model.state_dict()

        out[
            "optimizer_state_dict"
        ] = optimizer.state_dict()

        out[
            "epoch"
        ] = epoch

        out[
            "ordinary_head_finetune"
        ] = {
            "epoch": epoch,
            "best_macro_accuracy": (
                best_score
            ),
            "extra_repeat": (
                args.extra_repeat
            ),
            "base_train": str(
                args.base_train
            ),
            "extra_train": (
                None
                if args.extra_train is None
                else str(
                    args.extra_train
                )
            ),
            "validation": str(
                args.validation
            ),
            "lr": args.lr,
            "backbone_lr": (
                args.backbone_lr
            ),
            "seed": args.seed,
        }

        torch.save(
            out,
            latest_path,
        )

        if improved:
            torch.save(
                out,
                best_path,
            )

        history[
            "epochs"
        ].append(
            {
                "epoch": epoch,
                "train": metrics,
                "validation": validation,
            }
        )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    print(
        "best macro validation "
        f"accuracy: {best_score:.4f}",
        flush=True,
    )

    print(
        f"best checkpoint: {best_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
