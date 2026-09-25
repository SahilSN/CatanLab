import torch

from catanlab.rl_model import (
    RealismV2ActorCritic,
)

from catanlab.rl_teacher_training import train_teacher_v2_batches

from catanlab.rl_teacher import (
    TeacherDecisionKind,
    TeacherV2Example,
)
from catanlab.rl_teacher_dataset import (
    iter_teacher_v2_jsonl,
    save_teacher_v2_jsonl,
)
from catanlab.rl_teacher_training import (
    collate_teacher_v2_examples,
    iter_teacher_v2_batches,
)


def fixed_example(
    *,
    kind,
    label,
    action_dim,
    observation_value=0.0,
):
    return TeacherV2Example(
        decision_kind=kind,
        observation=tuple(
            observation_value
            for _ in range(1138)
        ),
        player_id=0,
        label=label,
        legal_mask=tuple(
            True
            for _ in range(
                action_dim
            )
        ),
        candidate_features=None,
    )


def dynamic_example(
    *,
    kind,
    label,
    candidate_count,
    feature_dim,
    observation_value=0.0,
):
    candidates = tuple(
        tuple(
            float(
                candidate_index
                + feature_index
            )
            for feature_index in range(
                feature_dim
            )
        )
        for candidate_index in range(
            candidate_count
        )
    )

    return TeacherV2Example(
        decision_kind=kind,
        observation=tuple(
            observation_value
            for _ in range(1138)
        ),
        player_id=0,
        label=label,
        legal_mask=tuple(
            True
            for _ in range(
                candidate_count
            )
        ),
        candidate_features=candidates,
    )


def test_iter_teacher_v2_jsonl_streams_examples(
    tmp_path,
):
    path = tmp_path / "teacher.jsonl"

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .ORDINARY_ACTION
            ),
            label=0,
            action_dim=202,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=1,
            action_dim=2,
        ),
    ]

    save_teacher_v2_jsonl(
        path,
        examples,
    )

    iterator = (
        iter_teacher_v2_jsonl(
            path
        )
    )

    assert iter(
        iterator
    ) is iterator

    assert list(
        iterator
    ) == examples


def test_collate_fixed_teacher_batch():
    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=0,
            action_dim=2,
            observation_value=1.0,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=1,
            action_dim=2,
            observation_value=2.0,
        ),
    ]

    batch = (
        collate_teacher_v2_examples(
            examples
        )
    )

    assert (
        batch.decision_kind
        == TeacherDecisionKind
        .TRADE_RESPONSE
    )

    assert batch.observations.shape == (
        2,
        1138,
    )

    assert batch.legal_masks.shape == (
        2,
        2,
    )

    assert batch.labels.tolist() == [
        0,
        1,
    ]

    assert (
        batch.candidate_features
        is None
    )


def test_collate_dynamic_batch_pads_candidates():
    examples = [
        dynamic_example(
            kind=(
                TeacherDecisionKind
                .DISCARD
            ),
            label=1,
            candidate_count=2,
            feature_dim=5,
        ),
        dynamic_example(
            kind=(
                TeacherDecisionKind
                .DISCARD
            ),
            label=2,
            candidate_count=4,
            feature_dim=5,
        ),
    ]

    batch = (
        collate_teacher_v2_examples(
            examples
        )
    )

    assert batch.legal_masks.shape == (
        2,
        4,
    )

    assert (
        batch.candidate_features
        .shape
        == (
            2,
            4,
            5,
        )
    )

    assert batch.labels.tolist() == [
        1,
        2,
    ]

    assert torch.equal(
        batch.legal_masks[0],
        torch.tensor(
            [
                True,
                True,
                False,
                False,
            ]
        ),
    )

    assert torch.all(
        batch.candidate_features[
            0,
            2:,
        ]
        == 0
    )


def test_collate_rejects_mixed_decision_kinds():
    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .ORDINARY_ACTION
            ),
            label=0,
            action_dim=202,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=0,
            action_dim=2,
        ),
    ]

    import pytest

    with pytest.raises(
        ValueError,
        match="one decision kind",
    ):
        collate_teacher_v2_examples(
            examples
        )


def test_streaming_batches_group_by_decision_kind(
    tmp_path,
):
    path = tmp_path / "teacher.jsonl"

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .ORDINARY_ACTION
            ),
            label=0,
            action_dim=202,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=0,
            action_dim=2,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .ORDINARY_ACTION
            ),
            label=1,
            action_dim=202,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=1,
            action_dim=2,
        ),
    ]

    save_teacher_v2_jsonl(
        path,
        examples,
    )

    batches = list(
        iter_teacher_v2_batches(
            path,
            batch_size=2,
        )
    )

    assert len(batches) == 2

    assert (
        batches[0].decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )

    assert (
        batches[1].decision_kind
        == TeacherDecisionKind
        .TRADE_RESPONSE
    )

    assert batches[0].batch_size == 2
    assert batches[1].batch_size == 2


def test_streaming_batches_flush_partial_batches(
    tmp_path,
):
    path = tmp_path / "teacher.jsonl"

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=0,
            action_dim=2,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=1,
            action_dim=2,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=0,
            action_dim=2,
        ),
    ]

    save_teacher_v2_jsonl(
        path,
        examples,
    )

    batches = list(
        iter_teacher_v2_batches(
            path,
            batch_size=2,
        )
    )

    assert [
        batch.batch_size
        for batch in batches
    ] == [
        2,
        1,
    ]


def test_ordinary_batch_routes_through_202_head():
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_teacher_training import (
        teacher_v2_batch_logits,
    )

    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    batch = (
        collate_teacher_v2_examples(
            [
                fixed_example(
                    kind=(
                        TeacherDecisionKind
                        .ORDINARY_ACTION
                    ),
                    label=0,
                    action_dim=202,
                ),
                fixed_example(
                    kind=(
                        TeacherDecisionKind
                        .ORDINARY_ACTION
                    ),
                    label=1,
                    action_dim=202,
                ),
            ]
        )
    )

    logits = (
        teacher_v2_batch_logits(
            model,
            batch,
        )
    )

    assert logits.shape == (
        2,
        202,
    )


def test_fixed_teacher_batch_routes_to_fixed_head():
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_teacher_training import (
        teacher_v2_batch_logits,
    )

    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    batch = (
        collate_teacher_v2_examples(
            [
                fixed_example(
                    kind=(
                        TeacherDecisionKind
                        .YEAR_OF_PLENTY
                    ),
                    label=0,
                    action_dim=15,
                ),
                fixed_example(
                    kind=(
                        TeacherDecisionKind
                        .YEAR_OF_PLENTY
                    ),
                    label=1,
                    action_dim=15,
                ),
            ]
        )
    )

    logits = (
        teacher_v2_batch_logits(
            model,
            batch,
        )
    )

    assert logits.shape == (
        2,
        15,
    )


def test_dynamic_teacher_batch_routes_to_candidate_head():
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_teacher_training import (
        teacher_v2_batch_logits,
    )

    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    batch = (
        collate_teacher_v2_examples(
            [
                dynamic_example(
                    kind=(
                        TeacherDecisionKind
                        .DISCARD
                    ),
                    label=0,
                    candidate_count=3,
                    feature_dim=5,
                ),
                dynamic_example(
                    kind=(
                        TeacherDecisionKind
                        .DISCARD
                    ),
                    label=1,
                    candidate_count=5,
                    feature_dim=5,
                ),
            ]
        )
    )

    logits = (
        teacher_v2_batch_logits(
            model,
            batch,
        )
    )

    assert logits.shape == (
        2,
        5,
    )


def test_teacher_v2_loss_masks_dynamic_padding():
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_teacher_training import (
        teacher_v2_batch_loss,
    )

    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    batch = (
        collate_teacher_v2_examples(
            [
                dynamic_example(
                    kind=(
                        TeacherDecisionKind
                        .DISCARD
                    ),
                    label=1,
                    candidate_count=2,
                    feature_dim=5,
                ),
                dynamic_example(
                    kind=(
                        TeacherDecisionKind
                        .DISCARD
                    ),
                    label=2,
                    candidate_count=4,
                    feature_dim=5,
                ),
            ]
        )
    )

    loss, masked_logits = (
        teacher_v2_batch_loss(
            model,
            batch,
        )
    )

    assert torch.isfinite(
        loss
    )

    assert torch.isneginf(
        masked_logits[
            0,
            2,
        ]
    )

    assert torch.isneginf(
        masked_logits[
            0,
            3,
        ]
    )


def test_teacher_v2_loss_backprop_reaches_shared_backbone():
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_teacher_training import (
        teacher_v2_batch_loss,
    )

    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    batch = (
        collate_teacher_v2_examples(
            [
                fixed_example(
                    kind=(
                        TeacherDecisionKind
                        .TRADE_RESPONSE
                    ),
                    label=0,
                    action_dim=2,
                    observation_value=1.0,
                ),
                fixed_example(
                    kind=(
                        TeacherDecisionKind
                        .TRADE_RESPONSE
                    ),
                    label=1,
                    action_dim=2,
                    observation_value=2.0,
                ),
            ]
        )
    )

    loss, _ = (
        teacher_v2_batch_loss(
            model,
            batch,
        )
    )

    loss.backward()

    first_linear = (
        model.backbone[0]
    )

    assert (
        first_linear.weight.grad
        is not None
    )

    assert torch.any(
        first_linear.weight.grad
        != 0
    )


def test_teacher_v2_batch_metrics():
    from catanlab.rl_teacher_training import (
        teacher_v2_batch_metrics,
    )

    logits = torch.tensor(
        [
            [
                3.0,
                1.0,
            ],
            [
                0.0,
                5.0,
            ],
            [
                4.0,
                1.0,
            ],
        ]
    )

    labels = torch.tensor(
        [
            0,
            1,
            1,
        ]
    )

    metrics = (
        teacher_v2_batch_metrics(
            logits,
            labels,
        )
    )

    assert metrics["correct"] == 2
    assert metrics["total"] == 3

    assert metrics["accuracy"] == (
        2 / 3
    )


def test_metric_accumulator_is_example_weighted():
    from catanlab.rl_teacher_training import (
        TeacherV2MetricAccumulator,
    )

    accumulator = (
        TeacherV2MetricAccumulator()
    )

    accumulator.update(
        loss=2.0,
        correct=1,
        total=2,
    )

    accumulator.update(
        loss=1.0,
        correct=3,
        total=3,
    )

    metrics = (
        accumulator.metrics()
    )

    assert metrics["examples"] == 5

    assert metrics["accuracy"] == (
        4 / 5
    )

    assert metrics["loss"] == (
        7 / 5
    )


def test_shuffled_batches_preserve_all_examples(
    tmp_path,
):
    from catanlab.rl_teacher_training import (
        iter_shuffled_teacher_v2_batches,
    )

    path = tmp_path / "teacher.jsonl"

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=index % 2,
            action_dim=2,
            observation_value=float(
                index
            ),
        )
        for index in range(12)
    ]

    save_teacher_v2_jsonl(
        path,
        examples,
    )

    batches = list(
        iter_shuffled_teacher_v2_batches(
            path,
            batch_size=2,
            shuffle_buffer_batches=3,
            seed=7,
        )
    )

    observed = []

    for batch in batches:
        observed.extend(
            int(value)
            for value in (
                batch.observations[
                    :,
                    0,
                ].tolist()
            )
        )

    assert sorted(
        observed
    ) == list(
        range(12)
    )


def test_shuffled_batches_are_deterministic_for_seed(
    tmp_path,
):
    from catanlab.rl_teacher_training import (
        iter_shuffled_teacher_v2_batches,
    )

    path = tmp_path / "teacher.jsonl"

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=index % 2,
            action_dim=2,
            observation_value=float(
                index
            ),
        )
        for index in range(20)
    ]

    save_teacher_v2_jsonl(
        path,
        examples,
    )

    def order():
        return [
            int(
                batch.observations[
                    0,
                    0,
                ].item()
            )
            for batch in (
                iter_shuffled_teacher_v2_batches(
                    path,
                    batch_size=2,
                    shuffle_buffer_batches=4,
                    seed=123,
                )
            )
        ]

    assert order() == order()


def test_shuffled_batches_change_with_seed(
    tmp_path,
):
    from catanlab.rl_teacher_training import (
        iter_shuffled_teacher_v2_batches,
    )

    path = tmp_path / "teacher.jsonl"

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=index % 2,
            action_dim=2,
            observation_value=float(
                index
            ),
        )
        for index in range(40)
    ]

    save_teacher_v2_jsonl(
        path,
        examples,
    )

    def order(seed):
        return [
            int(
                batch.observations[
                    0,
                    0,
                ].item()
            )
            for batch in (
                iter_shuffled_teacher_v2_batches(
                    path,
                    batch_size=2,
                    shuffle_buffer_batches=5,
                    seed=seed,
                )
            )
        ]

    assert order(1) != order(2)


def test_family_loss_weight_scales_backprop_not_metrics():
    model = RealismV2ActorCritic(
        hidden_dim=32,
    )

    examples = [
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=0,
            action_dim=2,
            observation_value=1.0,
        ),
        fixed_example(
            kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            label=1,
            action_dim=2,
            observation_value=2.0,
        ),
    ]

    batch = (
        collate_teacher_v2_examples(
            examples
        )
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=0.01,
    )

    before = (
        model.trade_response_head
        .weight.detach().clone()
    )

    metrics = train_teacher_v2_batches(
        model,
        [batch],
        optimizer,
        torch.device("cpu"),
        family_loss_weights={
            TeacherDecisionKind
            .TRADE_RESPONSE: 2.0,
        },
    )

    after = (
        model.trade_response_head
        .weight.detach().clone()
    )

    assert not torch.equal(
        before,
        after,
    )

    assert (
        metrics["overall"]["examples"]
        == 2
    )

    assert (
        metrics["by_decision_kind"][
            "trade_response"
        ]["examples"]
        == 2
    )
