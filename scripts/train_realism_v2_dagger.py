from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch

from catanlab.rl_model import (
    RealismV2ActorCritic,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)
from catanlab.rl_teacher_training import (
    evaluate_teacher_v2_batches,
    iter_shuffled_teacher_v2_batches,
    iter_teacher_v2_batches,
    train_teacher_v2_batches,
)


TRAINING_PROTOCOL = "realism-v2-bc-v1"
DAGGER_PROTOCOL = "realism-v2-dagger-ordinary-v1"


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


def interleave_batch_streams(
    base_batches,
    extra_batch_factories,
    *,
    seed,
):
    """
    Randomly interleave one frozen-corpus stream with a
    collection of extra DAgger streams.

    Every batch from every stream is emitted exactly once.
    """
    rng = random.Random(seed)

    streams = [
        iter(base_batches)
    ]

    streams.extend(
        iter(factory())
        for factory in extra_batch_factories
    )

    active = list(
        range(len(streams))
    )

    while active:
        position = rng.randrange(
            len(active)
        )

        stream_index = active[
            position
        ]

        try:
            yield next(
                streams[stream_index]
            )
        except StopIteration:
            active.pop(position)


def validation_batches(
    path,
    *,
    batch_size,
):
    return iter_teacher_v2_batches(
        path,
        batch_size=batch_size,
        drop_last=False,
    )


def print_metrics(
    prefix,
    metrics,
):
    overall = metrics[
        "overall"
    ]

    print(
        f"{prefix}: "
        f"loss={overall['loss']:.4f} "
        f"accuracy={overall['accuracy']:.4f} "
        f"macro_accuracy="
        f"{metrics['macro_accuracy']:.4f} "
        f"examples={overall['examples']} "
        f"batches={metrics['batches']} "
        f"kinds="
        f"{metrics['decision_kinds_observed']}/"
        f"{metrics['decision_kinds_total']}",
        flush=True,
    )

    # The shared training utilities expose the
    # per-decision-family mapping alongside the overall
    # metrics. Keep this reporting layer tolerant of the
    # historical key name used by the existing trainer.
    by_kind = None

    for key in (
        "by_kind",
        "by_decision_kind",
        "decision_kinds",
        "kind_metrics",
    ):
        candidate = metrics.get(
            key
        )

        if isinstance(
            candidate,
            dict,
        ):
            by_kind = candidate
            break

    if by_kind is None:
        raise KeyError(
            "Could not locate per-family metrics. "
            "Available metric keys: "
            f"{sorted(metrics)}"
        )

    for kind in TeacherDecisionKind:
        stats = by_kind.get(
            kind.value
        )

        if stats is None:
            continue

        print(
            f"  {kind.value:20s} "
            f"loss={stats['loss']:.4f} "
            f"acc={stats['accuracy']:.4f} "
            f"n={stats['examples']}",
            flush=True,
        )


def make_checkpoint(
    *,
    source_checkpoint,
    model,
    optimizer,
    epoch,
    best_macro_accuracy,
    args,
):
    checkpoint = dict(
        source_checkpoint
    )

    # Keep compatibility with the existing realism-v2
    # gameplay checkpoint loader.
    checkpoint[
        "training_protocol"
    ] = TRAINING_PROTOCOL

    checkpoint[
        "model_state_dict"
    ] = model.state_dict()

    checkpoint[
        "optimizer_state_dict"
    ] = optimizer.state_dict()

    checkpoint[
        "epoch"
    ] = epoch

    checkpoint[
        "best_macro_accuracy"
    ] = best_macro_accuracy

    checkpoint[
        "dagger"
    ] = {
        "protocol": DAGGER_PROTOCOL,
        "init_checkpoint": str(
            args.init_checkpoint
        ),
        "base_train": str(
            args.base_train
        ),
        "extra_train": str(
            args.extra_train
        ),
        "extra_repeat": (
            args.extra_repeat
        ),
        "validation": str(
            args.validation
        ),
        "lr": args.lr,
        "batch_size": (
            args.batch_size
        ),
        "shuffle_buffer_batches": (
            args.shuffle_buffer_batches
        ),
        "seed": args.seed,
    }

    return checkpoint


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Fine-tune a realism-v2 BC checkpoint "
            "using ordinary-action DAgger examples "
            "while retaining the frozen teacher corpus."
        )
    )

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
        required=True,
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
        "--extra-repeat",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
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
        default=2e-4,
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

    if args.epochs <= 0:
        raise ValueError(
            "--epochs must be positive."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be positive."
        )

    if args.lr <= 0:
        raise ValueError(
            "--lr must be positive."
        )

    if args.shuffle_buffer_batches <= 0:
        raise ValueError(
            "--shuffle-buffer-batches "
            "must be positive."
        )

    for path in (
        args.init_checkpoint,
        args.base_train,
        args.extra_train,
        args.validation,
    ):
        if not path.exists():
            raise FileNotFoundError(
                path
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    latest_path = (
        args.output_dir
        / "latest.pt"
    )

    best_path = (
        args.output_dir
        / "best.pt"
    )

    history_path = (
        args.output_dir
        / "history.json"
    )

    if (
        latest_path.exists()
        or best_path.exists()
        or history_path.exists()
    ):
        raise FileExistsError(
            "Output directory already contains "
            "DAgger training artifacts. Use a new "
            "directory or remove it explicitly."
        )

    device = choose_device(
        args.device
    )

    print(
        f"device: {device}",
        flush=True,
    )

    source_checkpoint = torch.load(
        args.init_checkpoint,
        map_location="cpu",
        weights_only=False,
    )

    protocol = source_checkpoint.get(
        "training_protocol"
    )

    if protocol != TRAINING_PROTOCOL:
        raise ValueError(
            "Initialization checkpoint has "
            "unexpected training protocol: "
            f"{protocol!r}"
        )

    observation_dim = int(
        source_checkpoint[
            "observation_dim"
        ]
    )

    action_dim = int(
        source_checkpoint[
            "action_dim"
        ]
    )

    hidden_dim = int(
        source_checkpoint[
            "hidden_dim"
        ]
    )

    if hidden_dim != args.hidden_dim:
        raise ValueError(
            "Hidden dimension mismatch: "
            f"checkpoint={hidden_dim}, "
            f"requested={args.hidden_dim}"
        )

    model = RealismV2ActorCritic(
        observation_dim=(
            observation_dim
        ),
        action_dim=action_dim,
        hidden_dim=hidden_dim,
    )

    model.load_state_dict(
        source_checkpoint[
            "model_state_dict"
        ]
    )

    model.to(device)

    # Deliberately start a NEW optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    print(
        "initialization checkpoint: "
        f"{args.init_checkpoint}",
        flush=True,
    )

    print(
        f"base train: {args.base_train}",
        flush=True,
    )

    print(
        f"DAgger train: {args.extra_train}",
        flush=True,
    )

    print(
        f"DAgger repeat: "
        f"{args.extra_repeat}",
        flush=True,
    )

    print(
        f"validation: {args.validation}",
        flush=True,
    )

    # Evaluate the initialization before any DAgger update.
    initial_validation = (
        evaluate_teacher_v2_batches(
            model,
            validation_batches(
                args.validation,
                batch_size=(
                    args.batch_size
                ),
            ),
            device,
        )
    )

    print()
    print_metrics(
        "initial validation",
        initial_validation,
    )

    best_macro_accuracy = (
        initial_validation[
            "macro_accuracy"
        ]
    )

    history = {
        "training_protocol": (
            TRAINING_PROTOCOL
        ),
        "dagger_protocol": (
            DAGGER_PROTOCOL
        ),
        "initial_validation": (
            initial_validation
        ),
        "epochs": [],
    }

    # The unchanged initialization is a valid candidate.
    initial_checkpoint = (
        make_checkpoint(
            source_checkpoint=(
                source_checkpoint
            ),
            model=model,
            optimizer=optimizer,
            epoch=0,
            best_macro_accuracy=(
                best_macro_accuracy
            ),
            args=args,
        )
    )

    torch.save(
        initial_checkpoint,
        best_path,
    )

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        print()
        print(
            f"=== DAGGER EPOCH "
            f"{epoch}/{args.epochs} ===",
            flush=True,
        )

        base_batches = (
            iter_shuffled_teacher_v2_batches(
                args.base_train,
                batch_size=(
                    args.batch_size
                ),
                shuffle_buffer_batches=(
                    args
                    .shuffle_buffer_batches
                ),
                seed=(
                    args.seed
                    + epoch * 10_000
                ),
                drop_last=False,
            )
        )

        extra_factories = []

        for repeat_index in range(
            args.extra_repeat
        ):
            repeat_seed = (
                args.seed
                + epoch * 100_000
                + repeat_index
            )

            def factory(
                repeat_seed=repeat_seed,
            ):
                return (
                    iter_shuffled_teacher_v2_batches(
                        args.extra_train,
                        batch_size=(
                            args.batch_size
                        ),
                        shuffle_buffer_batches=(
                            args
                            .shuffle_buffer_batches
                        ),
                        seed=repeat_seed,
                        drop_last=False,
                    )
                )

            extra_factories.append(
                factory
            )

        train_batches = (
            interleave_batch_streams(
                base_batches,
                extra_factories,
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
                family_loss_weights=None,
                log_every=(
                    args.log_every
                ),
            )
        )

        print_metrics(
            "train",
            train_metrics,
        )

        validation_metrics = (
            evaluate_teacher_v2_batches(
                model,
                validation_batches(
                    args.validation,
                    batch_size=(
                        args.batch_size
                    ),
                ),
                device,
            )
        )

        print_metrics(
            "validation",
            validation_metrics,
        )

        macro_accuracy = (
            validation_metrics[
                "macro_accuracy"
            ]
        )

        epoch_record = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": (
                validation_metrics
            ),
        }

        history[
            "epochs"
        ].append(
            epoch_record
        )

        improved = (
            validation_metrics[
                "complete_decision_kind_coverage"
            ]
            and macro_accuracy
            > best_macro_accuracy
        )

        if improved:
            best_macro_accuracy = (
                macro_accuracy
            )

        checkpoint = make_checkpoint(
            source_checkpoint=(
                source_checkpoint
            ),
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_macro_accuracy=(
                best_macro_accuracy
            ),
            args=args,
        )

        torch.save(
            checkpoint,
            latest_path,
        )

        if improved:
            torch.save(
                checkpoint,
                best_path,
            )

            print(
                "new best macro validation "
                f"accuracy: "
                f"{best_macro_accuracy:.4f}",
                flush=True,
            )

        history_path.write_text(
            json.dumps(
                history,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    print()
    print(
        "DAgger fine-tuning complete",
        flush=True,
    )

    print(
        "best macro validation accuracy: "
        f"{best_macro_accuracy:.4f}",
        flush=True,
    )

    print(
        f"best checkpoint: {best_path}",
        flush=True,
    )

    print(
        f"latest checkpoint: {latest_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
