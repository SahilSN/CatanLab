import json

import pytest

from catanlab.rl_teacher import (
    TeacherDecisionKind,
    TeacherV2Example,
)
from catanlab.rl_teacher_dataset import (
    TEACHER_V2_DATASET_VERSION,
    load_teacher_v2_jsonl,
    save_teacher_v2_jsonl,
    teacher_v2_example_from_record,
    teacher_v2_example_to_record,
    validate_teacher_v2_example,
)


def make_fixed_example():
    return TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=(
            0.0,
            1.0,
            2.0,
        ),
        player_id=0,
        label=4,
        legal_mask=(
            True,
            True,
            True,
            True,
            True,
        ),
    )


def make_dynamic_example():
    return TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=(
            3.0,
            4.0,
            5.0,
        ),
        player_id=1,
        label=1,
        legal_mask=(
            True,
            True,
        ),
        candidate_features=(
            (
                2.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
            (
                1.0,
                1.0,
                0.0,
                0.0,
                0.0,
            ),
        ),
    )


def test_teacher_dataset_fixed_record_round_trip():
    example = make_fixed_example()

    record = teacher_v2_example_to_record(
        example
    )

    restored = teacher_v2_example_from_record(
        record
    )

    assert restored == example

    assert (
        record["version"]
        == TEACHER_V2_DATASET_VERSION
    )

    assert (
        record["candidate_features"]
        is None
    )


def test_teacher_dataset_dynamic_record_round_trip():
    example = make_dynamic_example()

    record = teacher_v2_example_to_record(
        example
    )

    restored = teacher_v2_example_from_record(
        record
    )

    assert restored == example

    assert len(
        restored.candidate_features
    ) == len(
        restored.legal_mask
    )


def test_teacher_dataset_jsonl_round_trip(
    tmp_path,
):
    examples = [
        make_fixed_example(),
        make_dynamic_example(),
    ]

    path = (
        tmp_path
        / "teacher_v2.jsonl"
    )

    written = save_teacher_v2_jsonl(
        path,
        examples,
    )

    assert written == 2

    loaded = load_teacher_v2_jsonl(
        path
    )

    assert loaded == examples


def test_teacher_dataset_is_plain_jsonl(
    tmp_path,
):
    path = (
        tmp_path
        / "teacher_v2.jsonl"
    )

    save_teacher_v2_jsonl(
        path,
        [
            make_dynamic_example(),
        ],
    )

    line = path.read_text().strip()

    record = json.loads(
        line
    )

    assert (
        record["decision_kind"]
        == "discard"
    )

    assert record["label"] == 1
    assert isinstance(
        record["observation"],
        list,
    )

    assert isinstance(
        record["candidate_features"],
        list,
    )


def test_teacher_dataset_rejects_illegal_label():
    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind
            .TRADE_RESPONSE
        ),
        observation=(0.0,),
        player_id=0,
        label=1,
        legal_mask=(
            True,
            False,
        ),
    )

    with pytest.raises(
        ValueError,
        match="marked illegal",
    ):
        validate_teacher_v2_example(
            example
        )


def test_teacher_dataset_rejects_dynamic_missing_candidates():
    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=(0.0,),
        player_id=0,
        label=0,
        legal_mask=(True,),
    )

    with pytest.raises(
        ValueError,
        match="requires candidate features",
    ):
        validate_teacher_v2_example(
            example
        )


def test_teacher_dataset_rejects_candidate_count_mismatch():
    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=(0.0,),
        player_id=0,
        label=0,
        legal_mask=(
            True,
            True,
        ),
        candidate_features=(
            (
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="candidate count",
    ):
        validate_teacher_v2_example(
            example
        )


def test_teacher_dataset_rejects_wrong_candidate_width():
    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=(0.0,),
        player_id=0,
        label=0,
        legal_mask=(True,),
        candidate_features=(
            (
                1.0,
                0.0,
            ),
        ),
    )

    with pytest.raises(
        ValueError,
        match="feature width",
    ):
        validate_teacher_v2_example(
            example
        )


def test_teacher_dataset_rejects_candidates_on_fixed_kind():
    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=(0.0,),
        player_id=0,
        label=0,
        legal_mask=(
            True,
            True,
            True,
            True,
            True,
        ),
        candidate_features=(
            (0.0,),
        ),
    )

    with pytest.raises(
        ValueError,
        match="must not store",
    ):
        validate_teacher_v2_example(
            example
        )


def test_teacher_dataset_rejects_unknown_version():
    record = (
        teacher_v2_example_to_record(
            make_fixed_example()
        )
    )

    record["version"] = 999

    with pytest.raises(
        ValueError,
        match="Unsupported",
    ):
        teacher_v2_example_from_record(
            record
        )


def test_teacher_dataset_reports_jsonl_line_number(
    tmp_path,
):
    path = (
        tmp_path
        / "bad.jsonl"
    )

    path.write_text(
        '{"version":1}\n'
        'not-json\n'
    )

    with pytest.raises(
        ValueError,
        match="line",
    ):
        load_teacher_v2_jsonl(
            path
        )
