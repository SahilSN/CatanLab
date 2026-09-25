from __future__ import annotations

from dataclasses import dataclass

import torch

from catanlab.rl_teacher import (
    TeacherDecisionKind,
    TeacherV2Example,
)
from catanlab.rl_teacher_dataset import (
    iter_teacher_v2_jsonl,
)


@dataclass(frozen=True)
class TeacherV2Batch:
    """
    One homogeneous realism-v2 supervision batch.

    Every example in the batch has the same
    TeacherDecisionKind.

    Dynamic categorical spaces are right-padded along
    the candidate/action axis. Padding entries are
    illegal in legal_masks and therefore cannot
    participate in masked cross-entropy.
    """

    decision_kind: TeacherDecisionKind

    observations: torch.Tensor
    legal_masks: torch.Tensor
    labels: torch.Tensor

    candidate_features: (
        torch.Tensor | None
    )

    @property
    def batch_size(self) -> int:
        return int(
            self.labels.shape[0]
        )

    @property
    def action_dim(self) -> int:
        return int(
            self.legal_masks.shape[1]
        )


def collate_teacher_v2_examples(
    examples,
) -> TeacherV2Batch:
    examples = list(
        examples
    )

    if not examples:
        raise ValueError(
            "Cannot collate an empty teacher-v2 batch."
        )

    decision_kind = (
        examples[0].decision_kind
    )

    for example in examples:
        if (
            example.decision_kind
            != decision_kind
        ):
            raise ValueError(
                "Teacher-v2 batch must contain only "
                "one decision kind."
            )

    observation_dim = len(
        examples[0].observation
    )

    for example in examples:
        if (
            len(example.observation)
            != observation_dim
        ):
            raise ValueError(
                "Teacher-v2 batch observations must "
                "have matching dimensions."
            )

    observations = torch.tensor(
        [
            example.observation
            for example in examples
        ],
        dtype=torch.float32,
    )

    labels = torch.tensor(
        [
            example.label
            for example in examples
        ],
        dtype=torch.long,
    )

    max_action_dim = max(
        len(example.legal_mask)
        for example in examples
    )

    legal_masks = torch.zeros(
        len(examples),
        max_action_dim,
        dtype=torch.bool,
    )

    for row, example in enumerate(
        examples
    ):
        action_dim = len(
            example.legal_mask
        )

        legal_masks[
            row,
            :action_dim,
        ] = torch.tensor(
            example.legal_mask,
            dtype=torch.bool,
        )

    has_candidates = [
        example.candidate_features
        is not None
        for example in examples
    ]

    if any(
        has_candidates
    ) and not all(
        has_candidates
    ):
        raise ValueError(
            "Teacher-v2 batch cannot mix examples "
            "with and without candidate features."
        )

    candidate_tensor = None

    if all(
        has_candidates
    ):
        first_candidates = (
            examples[0]
            .candidate_features
        )

        if not first_candidates:
            raise ValueError(
                "Dynamic teacher-v2 example cannot "
                "have an empty candidate vocabulary."
            )

        feature_dim = len(
            first_candidates[0]
        )

        candidate_tensor = torch.zeros(
            len(examples),
            max_action_dim,
            feature_dim,
            dtype=torch.float32,
        )

        for row, example in enumerate(
            examples
        ):
            candidates = (
                example.candidate_features
            )

            if (
                len(candidates)
                != len(
                    example.legal_mask
                )
            ):
                raise ValueError(
                    "Candidate count must match "
                    "legal-mask dimension."
                )

            for candidate in candidates:
                if (
                    len(candidate)
                    != feature_dim
                ):
                    raise ValueError(
                        "Candidate feature dimensions "
                        "must match within a batch."
                    )

            candidate_tensor[
                row,
                :len(candidates),
                :,
            ] = torch.tensor(
                candidates,
                dtype=torch.float32,
            )

    return TeacherV2Batch(
        decision_kind=decision_kind,
        observations=observations,
        legal_masks=legal_masks,
        labels=labels,
        candidate_features=(
            candidate_tensor
        ),
    )


def iter_teacher_v2_batches(
    path,
    *,
    batch_size: int,
    drop_last: bool = False,
):
    """
    Stream a JSONL corpus into decision-kind-homogeneous
    batches with bounded memory.

    At most roughly one incomplete batch per decision
    kind is retained at any time.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be positive."
        )

    buffers = {
        decision_kind: []
        for decision_kind in (
            TeacherDecisionKind
        )
    }

    for example in (
        iter_teacher_v2_jsonl(
            path
        )
    ):
        buffer = buffers[
            example.decision_kind
        ]

        buffer.append(
            example
        )

        if len(buffer) == batch_size:
            yield (
                collate_teacher_v2_examples(
                    buffer
                )
            )

            buffers[
                example.decision_kind
            ] = []

    if drop_last:
        return

    # Deterministic end-of-file flush.
    for decision_kind in (
        TeacherDecisionKind
    ):
        buffer = buffers[
            decision_kind
        ]

        if buffer:
            yield (
                collate_teacher_v2_examples(
                    buffer
                )
            )


FIXED_TEACHER_DECISION_KINDS = frozenset(
    {
        TeacherDecisionKind.ORDINARY_ACTION,
        TeacherDecisionKind.MONOPOLY_RESOURCE,
        TeacherDecisionKind.YEAR_OF_PLENTY,
        TeacherDecisionKind.TRADE_RESPONSE,
    }
)


DYNAMIC_TEACHER_DECISION_KINDS = frozenset(
    {
        TeacherDecisionKind.ROBBER_TILE,
        TeacherDecisionKind.ROBBER_VICTIM,
        TeacherDecisionKind.DISCARD,
        TeacherDecisionKind.ROAD_BUILDING,
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_COUNTER,
    }
)


def teacher_v2_batch_logits(
    model,
    batch: TeacherV2Batch,
) -> torch.Tensor:
    """
    Route one homogeneous teacher-v2 batch through the
    appropriate RealismV2ActorCritic decision head.
    """
    decision_kind = (
        batch.decision_kind
    )

    if (
        decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    ):
        logits, _ = model(
            batch.observations
        )

        return logits

    if (
        decision_kind
        in FIXED_TEACHER_DECISION_KINDS
    ):
        return (
            model.fixed_decision_logits(
                batch.observations,
                decision_kind,
            )
        )

    if (
        decision_kind
        in DYNAMIC_TEACHER_DECISION_KINDS
    ):
        if (
            batch.candidate_features
            is None
        ):
            raise ValueError(
                "Dynamic teacher-v2 batch requires "
                "candidate features."
            )

        return (
            model.dynamic_decision_logits(
                batch.observations,
                decision_kind,
                batch.candidate_features,
            )
        )

    raise ValueError(
        "Unsupported teacher-v2 decision kind: "
        f"{decision_kind.value}"
    )


def teacher_v2_batch_loss(
    model,
    batch: TeacherV2Batch,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute masked categorical imitation loss.

    Returns:
        loss:
            scalar cross-entropy loss

        masked_logits:
            logits after illegal actions have been
            suppressed
    """
    from catanlab.rl_model import (
        mask_policy_logits,
    )

    logits = teacher_v2_batch_logits(
        model,
        batch,
    )

    if (
        logits.shape
        != batch.legal_masks.shape
    ):
        raise ValueError(
            "Teacher-v2 model logits do not match "
            "the batch categorical action space: "
            f"logits={tuple(logits.shape)}, "
            f"mask={tuple(batch.legal_masks.shape)}"
        )

    masked_logits = mask_policy_logits(
        logits,
        batch.legal_masks,
    )

    loss = (
        torch.nn.functional
        .cross_entropy(
            masked_logits,
            batch.labels,
        )
    )

    return (
        loss,
        masked_logits,
    )


def teacher_v2_batch_metrics(
    masked_logits: torch.Tensor,
    labels: torch.Tensor,
) -> dict:
    """
    Return basic categorical metrics for one batch.
    """
    predictions = (
        masked_logits.argmax(
            dim=-1
        )
    )

    correct = int(
        (
            predictions
            == labels
        ).sum().item()
    )

    total = int(
        labels.shape[0]
    )

    return {
        "correct": correct,
        "total": total,
        "accuracy": (
            correct / total
            if total
            else 0.0
        ),
    }


def move_teacher_v2_batch(
    batch: TeacherV2Batch,
    device,
) -> TeacherV2Batch:
    """
    Move all tensors in one teacher-v2 batch to a
    training device.
    """
    return TeacherV2Batch(
        decision_kind=(
            batch.decision_kind
        ),
        observations=(
            batch.observations.to(
                device
            )
        ),
        legal_masks=(
            batch.legal_masks.to(
                device
            )
        ),
        labels=(
            batch.labels.to(
                device
            )
        ),
        candidate_features=(
            None
            if (
                batch.candidate_features
                is None
            )
            else (
                batch.candidate_features
                .to(device)
            )
        ),
    )


@dataclass
class TeacherV2MetricAccumulator:
    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0

    def update(
        self,
        *,
        loss: float,
        correct: int,
        total: int,
    ) -> None:
        self.loss_sum += (
            loss * total
        )

        self.correct += correct
        self.total += total

    def metrics(self) -> dict:
        if self.total == 0:
            return {
                "loss": 0.0,
                "accuracy": 0.0,
                "examples": 0,
            }

        return {
            "loss": (
                self.loss_sum
                / self.total
            ),
            "accuracy": (
                self.correct
                / self.total
            ),
            "examples": (
                self.total
            ),
        }


@torch.no_grad()
def evaluate_teacher_v2_batches(
    model,
    batches,
    device,
    *,
    max_batches: int | None = None,
) -> dict:
    """
    Evaluate a realism-v2 model and report metrics
    separately for every observed decision family.

    Checkpoint-selection macro accuracy is the
    unweighted mean across decision families, not
    example-weighted global accuracy.
    """
    model.eval()

    overall = (
        TeacherV2MetricAccumulator()
    )

    by_kind = {
        decision_kind: (
            TeacherV2MetricAccumulator()
        )
        for decision_kind in (
            TeacherDecisionKind
        )
    }

    batches_completed = 0

    for batch in batches:
        if (
            max_batches is not None
            and batches_completed
            >= max_batches
        ):
            break

        batch = move_teacher_v2_batch(
            batch,
            device,
        )

        (
            loss,
            masked_logits,
        ) = teacher_v2_batch_loss(
            model,
            batch,
        )

        metrics = (
            teacher_v2_batch_metrics(
                masked_logits,
                batch.labels,
            )
        )

        batch_total = (
            metrics["total"]
        )

        overall.update(
            loss=float(
                loss.item()
            ),
            correct=(
                metrics["correct"]
            ),
            total=batch_total,
        )

        by_kind[
            batch.decision_kind
        ].update(
            loss=float(
                loss.item()
            ),
            correct=(
                metrics["correct"]
            ),
            total=batch_total,
        )

        batches_completed += 1

    kind_metrics = {}

    observed_accuracies = []

    for (
        decision_kind,
        accumulator,
    ) in by_kind.items():
        if accumulator.total == 0:
            continue

        metrics = (
            accumulator.metrics()
        )

        kind_metrics[
            decision_kind.value
        ] = metrics

        observed_accuracies.append(
            metrics["accuracy"]
        )

    macro_accuracy = (
        sum(observed_accuracies)
        / len(observed_accuracies)
        if observed_accuracies
        else 0.0
    )

    return {
        "overall": (
            overall.metrics()
        ),
        "by_decision_kind": (
            kind_metrics
        ),
        "macro_accuracy": (
            macro_accuracy
        ),
        "decision_kinds_observed": (
            len(kind_metrics)
        ),
        "decision_kinds_total": (
            len(TeacherDecisionKind)
        ),
        "complete_decision_kind_coverage": (
            len(kind_metrics)
            == len(TeacherDecisionKind)
        ),
        "batches": (
            batches_completed
        ),
    }


def iter_shuffled_teacher_v2_batches(
    path,
    *,
    batch_size: int,
    shuffle_buffer_batches: int,
    seed: int,
    drop_last: bool = False,
):
    """
    Stream kind-homogeneous teacher-v2 batches through a
    bounded shuffle buffer.

    This preserves every source example exactly once per
    epoch while avoiding deterministic corpus-order
    training and keeping memory bounded.

    Shuffling happens at the batch level because each batch
    must remain homogeneous in decision kind.
    """
    import random

    if shuffle_buffer_batches <= 0:
        raise ValueError(
            "shuffle_buffer_batches must be positive."
        )

    rng = random.Random(
        seed
    )

    source = iter_teacher_v2_batches(
        path,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    buffer = []

    for batch in source:
        if (
            len(buffer)
            < shuffle_buffer_batches
        ):
            buffer.append(
                batch
            )
            continue

        index = rng.randrange(
            len(buffer)
        )

        yield buffer[index]

        buffer[index] = batch

    rng.shuffle(
        buffer
    )

    yield from buffer


def iter_shuffled_teacher_v2_batches(
    path,
    *,
    batch_size: int,
    shuffle_buffer_batches: int,
    seed: int,
    drop_last: bool = False,
):
    """
    Stream kind-homogeneous teacher-v2 batches through a
    bounded shuffle buffer.

    This preserves every source example exactly once per
    epoch while avoiding deterministic corpus-order
    training and keeping memory bounded.

    Shuffling happens at the batch level because each batch
    must remain homogeneous in decision kind.
    """
    import random

    if shuffle_buffer_batches <= 0:
        raise ValueError(
            "shuffle_buffer_batches must be positive."
        )

    rng = random.Random(
        seed
    )

    source = iter_teacher_v2_batches(
        path,
        batch_size=batch_size,
        drop_last=drop_last,
    )

    buffer = []

    for batch in source:
        if (
            len(buffer)
            < shuffle_buffer_batches
        ):
            buffer.append(
                batch
            )
            continue

        index = rng.randrange(
            len(buffer)
        )

        yield buffer[index]

        buffer[index] = batch

    rng.shuffle(
        buffer
    )

    yield from buffer



def train_teacher_v2_batches(
    model,
    batches,
    optimizer,
    device,
    *,
    family_loss_weights=None,
    max_batches: int | None = None,
    log_every: int | None = None,
) -> dict:
    """
    Train for one stream of realism-v2 batches.

    Metrics are accumulated globally and separately by
    decision family using the natural example distribution.
    """
    model.train()

    overall = (
        TeacherV2MetricAccumulator()
    )

    by_kind = {
        decision_kind: (
            TeacherV2MetricAccumulator()
        )
        for decision_kind in (
            TeacherDecisionKind
        )
    }

    batches_completed = 0

    for batch in batches:
        if (
            max_batches is not None
            and batches_completed
            >= max_batches
        ):
            break

        batch = move_teacher_v2_batch(
            batch,
            device,
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        (
            loss,
            masked_logits,
        ) = teacher_v2_batch_loss(
            model,
            batch,
        )

        optimization_loss = loss

        if family_loss_weights is not None:
            optimization_loss = (
                loss
                * float(
                    family_loss_weights.get(
                        batch.decision_kind,
                        1.0,
                    )
                )
            )

        optimization_loss.backward()
        optimizer.step()

        metrics = (
            teacher_v2_batch_metrics(
                masked_logits,
                batch.labels,
            )
        )

        batch_total = (
            metrics["total"]
        )

        loss_value = float(
            loss.item()
        )

        overall.update(
            loss=loss_value,
            correct=(
                metrics["correct"]
            ),
            total=batch_total,
        )

        by_kind[
            batch.decision_kind
        ].update(
            loss=loss_value,
            correct=(
                metrics["correct"]
            ),
            total=batch_total,
        )

        batches_completed += 1

        if (
            log_every is not None
            and log_every > 0
            and (
                batches_completed
                % log_every
                == 0
            )
        ):
            current = overall.metrics()

            print(
                f"  train batch "
                f"{batches_completed} "
                f"examples="
                f"{current['examples']} "
                f"loss="
                f"{current['loss']:.4f} "
                f"acc="
                f"{current['accuracy']:.4f}",
                flush=True,
            )

    kind_metrics = {}

    observed_accuracies = []

    for (
        decision_kind,
        accumulator,
    ) in by_kind.items():
        if accumulator.total == 0:
            continue

        metrics = (
            accumulator.metrics()
        )

        kind_metrics[
            decision_kind.value
        ] = metrics

        observed_accuracies.append(
            metrics["accuracy"]
        )

    macro_accuracy = (
        sum(observed_accuracies)
        / len(observed_accuracies)
        if observed_accuracies
        else 0.0
    )

    return {
        "overall": (
            overall.metrics()
        ),
        "by_decision_kind": (
            kind_metrics
        ),
        "macro_accuracy": (
            macro_accuracy
        ),
        "decision_kinds_observed": (
            len(kind_metrics)
        ),
        "decision_kinds_total": (
            len(TeacherDecisionKind)
        ),
        "complete_decision_kind_coverage": (
            len(kind_metrics)
            == len(TeacherDecisionKind)
        ),
        "batches": (
            batches_completed
        ),
    }
