from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from catanlab.game import run_game
from catanlab.rl_teacher import (
    RecordingSearchAgent,
    TeacherV2Example,
)
from catanlab.rl_teacher_dataset import (
    TEACHER_V2_DATASET_VERSION,
    save_teacher_v2_jsonl,
)
from catanlab.strategies import StrategyType


DEFAULT_TEACHER_STRATEGIES = (
    StrategyType.FIVE_RESOURCE,
    StrategyType.HYBRID_OWS,
    StrategyType.FULL_OWS,
    StrategyType.PORT,
)


@dataclass(frozen=True)
class TeacherV2GenerationConfig:
    seed_start: int
    seed_count: int

    max_turns: int = 2000

    search_depth: int = 2
    use_transposition_cache: bool = False

    search_maritime_trades: bool = True
    search_year_of_plenty: bool = True
    search_road_building: bool = True
    search_monopoly: bool = True
    search_robber_decisions: bool = True
    search_discard_decisions: bool = True
    search_domestic_trades: bool = True

    validate_conservation: bool = True

    strategies: tuple[
        StrategyType,
        StrategyType,
        StrategyType,
        StrategyType,
    ] = DEFAULT_TEACHER_STRATEGIES

    def __post_init__(self):
        if self.seed_start < 0:
            raise ValueError(
                "seed_start cannot be negative."
            )

        if self.seed_count <= 0:
            raise ValueError(
                "seed_count must be positive."
            )

        if self.max_turns <= 0:
            raise ValueError(
                "max_turns must be positive."
            )

        if self.search_depth <= 0:
            raise ValueError(
                "search_depth must be positive."
            )

        if len(self.strategies) != 4:
            raise ValueError(
                "Teacher generation requires exactly "
                "four strategies."
            )

    @property
    def seeds(self) -> tuple[int, ...]:
        return tuple(
            range(
                self.seed_start,
                self.seed_start
                + self.seed_count,
            )
        )


def _build_recording_agent(
    strategy: StrategyType,
    config: TeacherV2GenerationConfig,
) -> RecordingSearchAgent:
    return RecordingSearchAgent(
        strategy,
        search_depth=config.search_depth,
        use_transposition_cache=(
            config.use_transposition_cache
        ),
        search_maritime_trades=(
            config.search_maritime_trades
        ),
        search_year_of_plenty=(
            config.search_year_of_plenty
        ),
        search_road_building=(
            config.search_road_building
        ),
        search_monopoly=(
            config.search_monopoly
        ),
        search_robber_decisions=(
            config.search_robber_decisions
        ),
        search_discard_decisions=(
            config.search_discard_decisions
        ),
        search_domestic_trades=(
            config.search_domestic_trades
        ),
    )


def build_recording_agents(
    config: TeacherV2GenerationConfig,
) -> list[RecordingSearchAgent]:
    return [
        _build_recording_agent(
            strategy,
            config,
        )
        for strategy in config.strategies
    ]


def collect_agent_v2_examples(
    agents,
) -> list[TeacherV2Example]:
    examples = []

    for agent in agents:
        examples.extend(
            agent.v2_examples
        )

    return examples


def generate_teacher_v2_examples(
    config: TeacherV2GenerationConfig,
    *,
    game_runner=run_game,
):
    """
    Run the configured deterministic seed range and collect
    all structured realism-v2 Search teacher decisions.

    Examples are emitted in deterministic order:
        seed order
        then agent/seat order
        then decision-recording order within that agent.
    """
    examples: list[TeacherV2Example] = []

    games_completed = 0
    games_with_winner = 0

    game_summaries = []

    for seed in config.seeds:
        agents = build_recording_agents(
            config
        )

        result = game_runner(
            strategies=list(
                config.strategies
            ),
            seed=seed,
            max_turns=config.max_turns,
            validate_conservation=(
                config.validate_conservation
            ),
            turn_agents=agents,
        )

        games_completed += 1

        if result.winner_id is not None:
            games_with_winner += 1

        game_examples = (
            collect_agent_v2_examples(
                agents
            )
        )

        examples.extend(
            game_examples
        )

        game_counts = Counter(
            example.decision_kind.value
            for example in game_examples
        )

        game_summaries.append(
            {
                "seed": seed,
                "winner_id": (
                    result.winner_id
                ),
                "turns_played": (
                    result.turns_played
                ),
                "examples": len(
                    game_examples
                ),
                "counts_by_decision_kind": {
                    key: game_counts[key]
                    for key in sorted(
                        game_counts
                    )
                },
            }
        )

    return (
        examples,
        {
            "games_attempted": (
                config.seed_count
            ),
            "games_completed": (
                games_completed
            ),
            "games_with_winner": (
                games_with_winner
            ),
            "games": game_summaries,
        },
    )


def teacher_v2_metadata(
    config: TeacherV2GenerationConfig,
    examples,
    run_metadata,
) -> dict:
    counts = Counter(
        example.decision_kind.value
        for example in examples
    )

    observation_dims = sorted(
        {
            len(example.observation)
            for example in examples
        }
    )

    return {
        "dataset_version": (
            TEACHER_V2_DATASET_VERSION
        ),
        "seed_start": config.seed_start,
        "seed_count": config.seed_count,
        "seeds": list(
            config.seeds
        ),
        "max_turns": config.max_turns,
        "games_attempted": (
            run_metadata[
                "games_attempted"
            ]
        ),
        "games_completed": (
            run_metadata[
                "games_completed"
            ]
        ),
        "games_with_winner": (
            run_metadata[
                "games_with_winner"
            ]
        ),
        "examples_total": len(
            examples
        ),
        "counts_by_decision_kind": {
            key: counts[key]
            for key in sorted(
                counts
            )
        },
        "observation_dims": (
            observation_dims
        ),
        "strategies": [
            strategy.value
            for strategy in (
                config.strategies
            )
        ],
        "search": {
            "depth": (
                config.search_depth
            ),
            "use_transposition_cache": (
                config.use_transposition_cache
            ),
            "maritime_trades": (
                config.search_maritime_trades
            ),
            "year_of_plenty": (
                config.search_year_of_plenty
            ),
            "road_building": (
                config.search_road_building
            ),
            "monopoly": (
                config.search_monopoly
            ),
            "robber_decisions": (
                config.search_robber_decisions
            ),
            "discard_decisions": (
                config.search_discard_decisions
            ),
            "domestic_trades": (
                config.search_domestic_trades
            ),
        },
        "validate_conservation": (
            config.validate_conservation
        ),
        "games": run_metadata[
            "games"
        ],
    }


def generate_teacher_v2_dataset(
    output_dir,
    config: TeacherV2GenerationConfig,
    *,
    game_runner=run_game,
):
    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    examples, run_metadata = (
        generate_teacher_v2_examples(
            config,
            game_runner=game_runner,
        )
    )

    dataset_path = (
        output_dir
        / "train.jsonl"
    )

    metadata_path = (
        output_dir
        / "metadata.json"
    )

    save_teacher_v2_jsonl(
        dataset_path,
        examples,
    )

    metadata = teacher_v2_metadata(
        config,
        examples,
        run_metadata,
    )

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "dataset_path": dataset_path,
        "metadata_path": metadata_path,
        "examples": examples,
        "metadata": metadata,
    }
