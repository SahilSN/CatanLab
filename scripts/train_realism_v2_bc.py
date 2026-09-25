from __future__ import annotations

import argparse
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


TRAINING_PROTOCOL = "realism-v2-bc-v1"


CANONICAL_TRAIN_FAMILY_COUNTS = {
    "discard": 10_800,
    "monopoly_resource": 1_164,
    "ordinary_action": 305_939,
    "road_building": 2_528,
    "robber_tile": 39_428,
    "robber_victim": 38_882,
    "trade_counter": 193_813,
    "trade_proposal": 302_440,
    "trade_response": 228_331,
    "year_of_plenty": 806,
}


def build_family_loss_weights(
    mode: str,
):
    if mode == "none":
        return None

    exponents = {
        "fourth_root_inverse": -0.25,
        "sqrt_inverse": -0.5,
    }

    try:
        exponent = exponents[
            mode
        ]
    except KeyError as exc:
        raise ValueError(
            "Unsupported family weighting mode: "
            f"{mode!r}"
        ) from exc

    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )

    raw = {
        decision_kind: (
            CANONICAL_TRAIN_FAMILY_COUNTS[
                decision_kind.value
            ]
            ** exponent
        )
        for decision_kind in (
            TeacherDecisionKind
        )
    }

    # Normalize so the example-weighted mean training
    # weight is 1.0. This preserves the approximate global
    # gradient scale while increasing rare-family influence.
    weighted_sum = sum(
        CANONICAL_TRAIN_FAMILY_COUNTS[
            decision_kind.value
        ]
        * weight
        for decision_kind, weight
        in raw.items()
    )

    total_examples = sum(
        CANONICAL_TRAIN_FAMILY_COUNTS.values()
    )

    scale = (
        total_examples
        / weighted_sum
    )

    return {
        decision_kind: (
            weight * scale
        )
        for decision_kind, weight
        in raw.items()
    }


def choose_device(
    requested: str,
) -> torch.device:
    if requested != "auto":
        return torch.device(
            requested
        )

    if torch.cuda.is_available():
        return torch.device(
            "cuda"
        )

    if (
        hasattr(
            torch.backends,
            "mps",
        )
        and torch.backends.mps.is_available()
    ):
        return torch.device(
            "mps"
        )

    return torch.device(
        "cpu"
    )


def seed_everything(
    seed: int,
) -> None:
    random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(
            seed
        )


def save_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    epoch: int,
    best_macro_accuracy: float,
    args,
    train_metrics,
    validation_metrics,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name + ".tmp"
    )

    torch.save(
        {
            "training_protocol": (
                TRAINING_PROTOCOL
            ),
            "model_class": (
                "realism_v2"
            ),
            "observation_dim": 1138,
            "action_dim": 202,
            "hidden_dim": (
                args.hidden_dim
            ),
            "epoch": epoch,
            "best_macro_accuracy": (
                best_macro_accuracy
            ),
            "model_state_dict": (
                model.state_dict()
            ),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "training_args": {
                key: (
                    str(value)
                    if isinstance(
                        value,
                        Path,
                    )
                    else value
                )
                for key, value in (
                    vars(args).items()
                )
            },
            "train_metrics": (
                train_metrics
            ),
            "validation_metrics": (
                validation_metrics
            ),
        },
        temporary,
    )

    temporary.replace(
        path
    )


def load_resume_checkpoint(
    path: Path,
    *,
    model,
    optimizer,
    device,
):
    # This training checkpoint is produced locally by
    # this script and includes optimizer state plus metadata.
    # PyTorch 2.6+ defaults torch.load() to weights_only=True,
    # which rejects legacy Path objects in our smoke checkpoint.
    checkpoint = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )

    if (
        checkpoint.get(
            "training_protocol"
        )
        != TRAINING_PROTOCOL
    ):
        raise ValueError(
            "Unsupported or incompatible "
            "realism-v2 training checkpoint."
        )

    if (
        checkpoint.get(
            "model_class"
        )
        != "realism_v2"
    ):
        raise ValueError(
            "Checkpoint is not a realism-v2 model."
        )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    optimizer.load_state_dict(
        checkpoint[
            "optimizer_state_dict"
        ]
    )

    return checkpoint


def print_metrics(
    name: str,
    metrics: dict,
) -> None:
    overall = metrics[
        "overall"
    ]

    print(
        f"{name}: "
        f"loss={overall['loss']:.4f} "
        f"accuracy="
        f"{overall['accuracy']:.4f} "
        f"macro_accuracy="
        f"{metrics['macro_accuracy']:.4f} "
        f"examples="
        f"{overall['examples']} "
        f"batches="
        f"{metrics['batches']} "
        f"kinds="
        f"{metrics['decision_kinds_observed']}/"
        f"{metrics['decision_kinds_total']}",
        flush=True,
    )

    for (
        decision_kind,
        kind_metrics,
    ) in sorted(
        metrics[
            "by_decision_kind"
        ].items()
    ):
        print(
            f"  {decision_kind:20s} "
            f"loss="
            f"{kind_metrics['loss']:.4f} "
            f"acc="
            f"{kind_metrics['accuracy']:.4f} "
            f"n="
            f"{kind_metrics['examples']}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the unified realism-v2 "
            "behavior-cloning policy."
        )
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path(
            "data/teacher_v2/"
            "realism-v2-v1/train/"
            "train.jsonl"
        ),
    )

    parser.add_argument(
        "--validation",
        type=Path,
        default=Path(
            "data/teacher_v2/"
            "realism-v2-v1/validation/"
            "train.jsonl"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "results/realism_v2_bc/v1"
        ),
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
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
        "--family-weighting",
        choices=(
            "none",
            "fourth_root_inverse",
            "sqrt_inverse",
        ),
        default="none",
        help=(
            "Optional decision-family loss weighting. "
            "fourth_root_inverse and sqrt_inverse use "
            "tempered inverse canonical frequency."
        ),
    )

    parser.add_argument(
        "--device",
        default="auto",
        help=(
            "auto, cpu, mps, cuda, etc."
        ),
    )

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--max-train-batches",
        type=int,
        default=None,
        help=(
            "Smoke/debug limit. Omit for a "
            "complete training epoch."
        ),
    )

    parser.add_argument(
        "--max-validation-batches",
        type=int,
        default=None,
        help=(
            "Smoke/debug limit. Omit for full "
            "canonical validation."
        ),
    )

    parser.add_argument(
        "--log-every",
        type=int,
        default=250,
    )

    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "batch-size must be positive."
        )

    if args.hidden_dim <= 0:
        raise ValueError(
            "hidden-dim must be positive."
        )

    if args.lr <= 0:
        raise ValueError(
            "lr must be positive."
        )

    if (
        args.shuffle_buffer_batches
        <= 0
    ):
        raise ValueError(
            "shuffle-buffer-batches "
            "must be positive."
        )

    if not args.train.exists():
        raise FileNotFoundError(
            args.train
        )

    if not args.validation.exists():
        raise FileNotFoundError(
            args.validation
        )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    seed_everything(
        args.seed
    )

    device = choose_device(
        args.device
    )

    print(
        f"device: {device}",
        flush=True,
    )

    family_loss_weights = (
        build_family_loss_weights(
            args.family_weighting
        )
    )

    print(
        "family weighting: "
        f"{args.family_weighting}",
        flush=True,
    )

    if family_loss_weights is not None:
        for (
            decision_kind,
            weight,
        ) in sorted(
            family_loss_weights.items(),
            key=lambda item: (
                item[0].value
            ),
        ):
            print(
                f"  {decision_kind.value:20s} "
                f"{weight:.6f}",
                flush=True,
            )

    model = RealismV2ActorCritic(
        observation_dim=1138,
        action_dim=202,
        hidden_dim=args.hidden_dim,
    ).to(
        device
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    start_epoch = 1
    best_macro_accuracy = float(
        "-inf"
    )

    if args.resume is not None:
        checkpoint = (
            load_resume_checkpoint(
                args.resume,
                model=model,
                optimizer=optimizer,
                device=device,
            )
        )

        checkpoint_args = (
            checkpoint[
                "training_args"
            ]
        )

        for key in (
            "batch_size",
            "hidden_dim",
            "lr",
            "shuffle_buffer_batches",
            "seed",
            "family_weighting",
        ):
            checkpoint_value = (
                checkpoint_args.get(
                    key
                )
            )

            # Checkpoints created before family weighting
            # was introduced have no saved value. Their
            # historical behavior is equivalent to
            # --family-weighting none.
            if (
                key == "family_weighting"
                and checkpoint_value is None
            ):
                checkpoint_value = "none"

            if (
                checkpoint_value
                != getattr(
                    args,
                    key,
                )
            ):
                raise ValueError(
                    "Resume configuration mismatch: "
                    f"{key}"
                )

        start_epoch = (
            int(
                checkpoint[
                    "epoch"
                ]
            )
            + 1
        )

        best_macro_accuracy = float(
            checkpoint[
                "best_macro_accuracy"
            ]
        )

        print(
            f"resuming after epoch "
            f"{start_epoch - 1}",
            flush=True,
        )

    history_path = (
        args.output_dir
        / "history.jsonl"
    )

    latest_path = (
        args.output_dir
        / "latest.pt"
    )

    best_path = (
        args.output_dir
        / "best.pt"
    )

    for epoch in range(
        start_epoch,
        args.epochs + 1,
    ):
        print(
            f"\n=== EPOCH {epoch}/"
            f"{args.epochs} ===",
            flush=True,
        )

        train_batches = (
            iter_shuffled_teacher_v2_batches(
                args.train,
                batch_size=(
                    args.batch_size
                ),
                shuffle_buffer_batches=(
                    args.shuffle_buffer_batches
                ),
                seed=(
                    args.seed
                    + epoch
                ),
            )
        )

        train_metrics = (
            train_teacher_v2_batches(
                model,
                train_batches,
                optimizer,
                device,
                family_loss_weights=(
                    family_loss_weights
                ),
                max_batches=(
                    args.max_train_batches
                ),
                log_every=(
                    args.log_every
                ),
            )
        )

        print_metrics(
            "train",
            train_metrics,
        )

        validation_batches = (
            iter_teacher_v2_batches(
                args.validation,
                batch_size=(
                    args.batch_size
                ),
            )
        )

        validation_metrics = (
            evaluate_teacher_v2_batches(
                model,
                validation_batches,
                device,
                max_batches=(
                    args.max_validation_batches
                ),
            )
        )

        print_metrics(
            "validation",
            validation_metrics,
        )

        macro_accuracy = float(
            validation_metrics[
                "macro_accuracy"
            ]
        )

        improved = (
            validation_metrics[
                "complete_decision_kind_coverage"
            ]
            and (
                macro_accuracy
                > best_macro_accuracy
            )
        )

        if improved:
            best_macro_accuracy = (
                macro_accuracy
            )

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_macro_accuracy=(
                best_macro_accuracy
            ),
            args=args,
            train_metrics=(
                train_metrics
            ),
            validation_metrics=(
                validation_metrics
            ),
        )

        if improved:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_macro_accuracy=(
                    best_macro_accuracy
                ),
                args=args,
                train_metrics=(
                    train_metrics
                ),
                validation_metrics=(
                    validation_metrics
                ),
            )

            print(
                "new best macro validation "
                f"accuracy: "
                f"{best_macro_accuracy:.4f}",
                flush=True,
            )

        with history_path.open(
            "a",
            encoding="utf-8",
        ) as handle:
            handle.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train": (
                            train_metrics
                        ),
                        "validation": (
                            validation_metrics
                        ),
                        "best_macro_accuracy": (
                            best_macro_accuracy
                        ),
                    },
                    sort_keys=True,
                )
            )

            handle.write(
                "\n"
            )

    print()
    print(
        "training complete",
        flush=True,
    )

    if best_path.exists():
        print(
            f"best macro validation accuracy: "
            f"{best_macro_accuracy:.4f}",
            flush=True,
        )

        print(
            f"best checkpoint: {best_path}",
            flush=True,
        )

    else:
        print(
            "best checkpoint: not created "
            "(validation did not cover all "
            "decision families)",
            flush=True,
        )


if __name__ == "__main__":
    main()
