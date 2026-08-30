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


def test_ordinary_teacher_example_converts_to_v2():
    from catanlab.rl_teacher import (
        TeacherExample,
    )
    from catanlab.rl_teacher_generation import (
        ordinary_teacher_example_to_v2,
    )

    legal_mask = tuple(
        index in {
            0,
            17,
            201,
        }
        for index in range(202)
    )

    legacy = TeacherExample(
        observation=tuple(
            float(index)
            for index in range(1138)
        ),
        legal_mask=legal_mask,
        action_id=17,
        player_id=2,
    )

    example = (
        ordinary_teacher_example_to_v2(
            legacy
        )
    )

    assert (
        example.decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )

    assert (
        example.observation
        == legacy.observation
    )

    assert (
        example.player_id
        == legacy.player_id
    )

    assert example.label == 17

    assert (
        example.legal_mask
        == legal_mask
    )

    assert (
        example.candidate_features
        is None
    )


def test_collect_agent_examples_includes_ordinary_and_structured():
    from catanlab.rl_teacher import (
        TeacherExample,
    )
    from catanlab.rl_teacher_generation import (
        collect_agent_v2_examples,
    )
    from catanlab.strategies import (
        StrategyType,
    )
    from catanlab.rl_teacher import (
        RecordingSearchAgent,
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    agent.examples.append(
        TeacherExample(
            observation=(1.0, 2.0),
            legal_mask=(
                True,
                False,
                True,
            ),
            action_id=2,
            player_id=0,
        )
    )

    agent.v2_examples.append(
        TeacherV2Example(
            decision_kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            observation=(3.0, 4.0),
            player_id=0,
            label=0,
            legal_mask=(
                True,
                False,
            ),
        )
    )

    examples = (
        collect_agent_v2_examples(
            [agent]
        )
    )

    assert len(examples) == 2

    assert (
        examples[0].decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )

    assert examples[0].label == 2

    assert (
        examples[1].decision_kind
        == TeacherDecisionKind
        .TRADE_RESPONSE
    )


def test_serialized_ordinary_example_round_trips(
    tmp_path,
):
    from catanlab.rl_teacher import (
        TeacherExample,
    )
    from catanlab.rl_teacher_generation import (
        ordinary_teacher_example_to_v2,
    )
    from catanlab.rl_teacher_dataset import (
        load_teacher_v2_jsonl,
        save_teacher_v2_jsonl,
    )

    legacy = TeacherExample(
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        legal_mask=tuple(
            index == 0
            for index in range(202)
        ),
        action_id=0,
        player_id=1,
    )

    unified = (
        ordinary_teacher_example_to_v2(
            legacy
        )
    )

    path = tmp_path / "ordinary.jsonl"

    save_teacher_v2_jsonl(
        path,
        [unified],
    )

    loaded = load_teacher_v2_jsonl(
        path
    )

    assert loaded == [
        unified
    ]

    assert (
        loaded[0].decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )


def test_ordinary_teacher_example_converts_to_v2():
    from catanlab.rl_teacher import (
        TeacherExample,
    )
    from catanlab.rl_teacher_generation import (
        ordinary_teacher_example_to_v2,
    )

    legal_mask = tuple(
        index in {
            0,
            17,
            201,
        }
        for index in range(202)
    )

    legacy = TeacherExample(
        observation=tuple(
            float(index)
            for index in range(1138)
        ),
        legal_mask=legal_mask,
        action_id=17,
        player_id=2,
    )

    example = (
        ordinary_teacher_example_to_v2(
            legacy
        )
    )

    assert (
        example.decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )

    assert (
        example.observation
        == legacy.observation
    )

    assert (
        example.player_id
        == legacy.player_id
    )

    assert example.label == 17

    assert (
        example.legal_mask
        == legal_mask
    )

    assert (
        example.candidate_features
        is None
    )


def test_collect_agent_examples_includes_ordinary_and_structured():
    from catanlab.rl_teacher import (
        TeacherExample,
    )
    from catanlab.rl_teacher_generation import (
        collect_agent_v2_examples,
    )
    from catanlab.strategies import (
        StrategyType,
    )
    from catanlab.rl_teacher import (
        RecordingSearchAgent,
    )

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    agent.examples.append(
        TeacherExample(
            observation=(1.0, 2.0),
            legal_mask=(
                True,
                False,
                True,
            ),
            action_id=2,
            player_id=0,
        )
    )

    agent.v2_examples.append(
        TeacherV2Example(
            decision_kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            observation=(3.0, 4.0),
            player_id=0,
            label=0,
            legal_mask=(
                True,
                False,
            ),
        )
    )

    examples = (
        collect_agent_v2_examples(
            [agent]
        )
    )

    assert len(examples) == 2

    assert (
        examples[0].decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )

    assert examples[0].label == 2

    assert (
        examples[1].decision_kind
        == TeacherDecisionKind
        .TRADE_RESPONSE
    )


def test_serialized_ordinary_example_round_trips(
    tmp_path,
):
    from catanlab.rl_teacher import (
        TeacherExample,
    )
    from catanlab.rl_teacher_generation import (
        ordinary_teacher_example_to_v2,
    )
    from catanlab.rl_teacher_dataset import (
        load_teacher_v2_jsonl,
        save_teacher_v2_jsonl,
    )

    legacy = TeacherExample(
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        legal_mask=tuple(
            index == 0
            for index in range(202)
        ),
        action_id=0,
        player_id=1,
    )

    unified = (
        ordinary_teacher_example_to_v2(
            legacy
        )
    )

    path = tmp_path / "ordinary.jsonl"

    save_teacher_v2_jsonl(
        path,
        [unified],
    )

    loaded = load_teacher_v2_jsonl(
        path
    )

    assert loaded == [
        unified
    ]

    assert (
        loaded[0].decision_kind
        == TeacherDecisionKind
        .ORDINARY_ACTION
    )


def test_canonical_teacher_v2_train_config():
    from catanlab.rl_teacher_generation import (
        CANONICAL_TEACHER_V2_PROTOCOL,
        canonical_teacher_v2_config,
    )

    config = canonical_teacher_v2_config(
        "train"
    )

    assert (
        config.generation_protocol
        == CANONICAL_TEACHER_V2_PROTOCOL
    )
    assert config.split == "train"

    assert config.seed_start == 1_000_000
    assert config.seed_count == 2_000

    assert config.seeds[0] == 1_000_000
    assert config.seeds[-1] == 1_001_999

    assert config.max_turns == 2000
    assert config.search_depth == 2

    assert (
        config.use_transposition_cache
        is False
    )

    assert config.search_maritime_trades
    assert config.search_year_of_plenty
    assert config.search_road_building
    assert config.search_monopoly
    assert config.search_robber_decisions
    assert config.search_discard_decisions
    assert config.search_domestic_trades
    assert config.validate_conservation


def test_canonical_teacher_v2_validation_config():
    from catanlab.rl_teacher_generation import (
        CANONICAL_TEACHER_V2_PROTOCOL,
        canonical_teacher_v2_config,
    )

    config = canonical_teacher_v2_config(
        "validation"
    )

    assert (
        config.generation_protocol
        == CANONICAL_TEACHER_V2_PROTOCOL
    )
    assert config.split == "validation"

    assert config.seed_start == 1_100_000
    assert config.seed_count == 250

    assert config.seeds[0] == 1_100_000
    assert config.seeds[-1] == 1_100_249

    assert config.max_turns == 2000
    assert config.search_depth == 2


def test_canonical_teacher_v2_splits_are_disjoint():
    from catanlab.rl_teacher_generation import (
        CANONICAL_TEACHER_V2_EVAL_SEED_START,
        canonical_teacher_v2_config,
    )

    train = canonical_teacher_v2_config(
        "train"
    )
    validation = (
        canonical_teacher_v2_config(
            "validation"
        )
    )

    assert set(
        train.seeds
    ).isdisjoint(
        validation.seeds
    )

    assert (
        max(train.seeds)
        < min(validation.seeds)
    )

    assert (
        max(validation.seeds)
        < CANONICAL_TEACHER_V2_EVAL_SEED_START
    )


def test_canonical_teacher_v2_rejects_unknown_split():
    import pytest

    from catanlab.rl_teacher_generation import (
        canonical_teacher_v2_config,
    )

    with pytest.raises(
        ValueError,
        match="train.*validation",
    ):
        canonical_teacher_v2_config(
            "test"
        )
