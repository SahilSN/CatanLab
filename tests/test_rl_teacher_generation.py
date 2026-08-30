from types import SimpleNamespace

import pytest

from catanlab.rl_teacher import (
    TeacherDecisionKind,
    TeacherV2Example,
)
from catanlab.rl_teacher_dataset import (
    load_teacher_v2_jsonl,
)
from catanlab.rl_teacher_generation import (
    TeacherV2GenerationConfig,
    generate_teacher_v2_dataset,
    generate_teacher_v2_examples,
)


def fake_game_runner(
    *,
    strategies,
    seed,
    max_turns,
    validate_conservation,
    turn_agents,
):
    assert len(strategies) == 4
    assert len(turn_agents) == 4
    assert max_turns > 0

    # Deterministic synthetic example attached directly to
    # the recording agent so the generator/serialization
    # path can be tested independently of game complexity.
    turn_agents[0].v2_examples.append(
        TeacherV2Example(
            decision_kind=(
                TeacherDecisionKind
                .MONOPOLY_RESOURCE
            ),
            observation=tuple(
                float(seed)
                for _ in range(3)
            ),
            player_id=0,
            label=seed % 5,
            legal_mask=(
                True,
                True,
                True,
                True,
                True,
            ),
        )
    )

    turn_agents[1].v2_examples.append(
        TeacherV2Example(
            decision_kind=(
                TeacherDecisionKind.DISCARD
            ),
            observation=tuple(
                float(seed + 1)
                for _ in range(3)
            ),
            player_id=1,
            label=0,
            legal_mask=(True,),
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
    )

    return SimpleNamespace(
        winner_id=seed % 4,
        turns_played=20 + seed,
    )


def test_teacher_generation_uses_exact_seed_range():
    config = TeacherV2GenerationConfig(
        seed_start=10,
        seed_count=3,
        max_turns=50,
    )

    examples, metadata = (
        generate_teacher_v2_examples(
            config,
            game_runner=(
                fake_game_runner
            ),
        )
    )

    assert config.seeds == (
        10,
        11,
        12,
    )

    assert len(examples) == 6

    assert (
        metadata["games_attempted"]
        == 3
    )

    assert (
        metadata["games_completed"]
        == 3
    )

    assert [
        game["seed"]
        for game in metadata["games"]
    ] == [
        10,
        11,
        12,
    ]


def test_teacher_generation_preserves_seed_order():
    config = TeacherV2GenerationConfig(
        seed_start=7,
        seed_count=2,
    )

    examples, _ = (
        generate_teacher_v2_examples(
            config,
            game_runner=(
                fake_game_runner
            ),
        )
    )

    assert examples[0].observation == (
        7.0,
        7.0,
        7.0,
    )

    assert examples[2].observation == (
        8.0,
        8.0,
        8.0,
    )


def test_teacher_dataset_generation_writes_expected_files(
    tmp_path,
):
    config = TeacherV2GenerationConfig(
        seed_start=3,
        seed_count=2,
    )

    result = generate_teacher_v2_dataset(
        tmp_path,
        config,
        game_runner=fake_game_runner,
    )

    assert (
        result["dataset_path"].name
        == "train.jsonl"
    )

    assert (
        result["metadata_path"].name
        == "metadata.json"
    )

    assert result[
        "dataset_path"
    ].is_file()

    assert result[
        "metadata_path"
    ].is_file()

    loaded = load_teacher_v2_jsonl(
        result["dataset_path"]
    )

    assert loaded == result[
        "examples"
    ]


def test_teacher_generation_metadata_counts_kinds(
    tmp_path,
):
    config = TeacherV2GenerationConfig(
        seed_start=0,
        seed_count=4,
    )

    result = generate_teacher_v2_dataset(
        tmp_path,
        config,
        game_runner=fake_game_runner,
    )

    metadata = result[
        "metadata"
    ]

    assert (
        metadata["examples_total"]
        == 8
    )

    assert (
        metadata[
            "counts_by_decision_kind"
        ][
            "monopoly_resource"
        ]
        == 4
    )

    assert (
        metadata[
            "counts_by_decision_kind"
        ][
            "discard"
        ]
        == 4
    )


def test_teacher_generation_is_byte_deterministic(
    tmp_path,
):
    config = TeacherV2GenerationConfig(
        seed_start=100,
        seed_count=3,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"

    generate_teacher_v2_dataset(
        first,
        config,
        game_runner=fake_game_runner,
    )

    generate_teacher_v2_dataset(
        second,
        config,
        game_runner=fake_game_runner,
    )

    assert (
        (first / "train.jsonl")
        .read_bytes()
        ==
        (second / "train.jsonl")
        .read_bytes()
    )

    assert (
        (first / "metadata.json")
        .read_bytes()
        ==
        (second / "metadata.json")
        .read_bytes()
    )


def test_teacher_generation_rejects_invalid_config():
    with pytest.raises(
        ValueError,
        match="seed_count",
    ):
        TeacherV2GenerationConfig(
            seed_start=0,
            seed_count=0,
        )
