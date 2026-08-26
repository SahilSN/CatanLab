# CatanLab

CatanLab is a simulation and strategy-analysis toolkit for Settlers of Catan.

The project provides a four-player Catan simulator, deterministic simulation infrastructure, heuristic strategy agents, and benchmarking tools for studying differences between Catan strategies.

## Current Status

CatanLab currently supports:

- a standard 19-hex Catan board
- randomized resource layouts and standard number-token placement
- four-player snake-order setup
- roads, settlements, and cities with standard piece limits
- settlement distance and road-connectivity rules
- production and finite-bank resource handling
- the robber, discarding on 7, and resource stealing
- domestic trading
- maritime trading with 4:1, 3:1, and resource-specific 2:1 ports
- the standard development-card deck
- Knight, Monopoly, Year of Plenty, Road Building, and Victory Point cards
- Largest Army
- Longest Road
- Longest Road interruption and tie handling
- 10-point victory conditions
- hidden-information-safe strategy observations
- deterministic seeded simulation
- resource-conservation validation
- multi-action turns
- per-turn action histories and game diagnostics

The core rules implementation has been extensively tested and is currently treated as stable.

Run the tests with:

    pytest -q

At the current strategy-freeze point, the full test suite contains 293 passing tests.

## Heuristic Strategy Baseline

CatanLab currently contains six rule-based strategy archetypes:

- `full_ows`
- `hybrid_ows`
- `road_building`
- `roads_and_cities`
- `five_resource`
- `port`

These agents are intentionally behaviorally different. They are not tuned to have equal win rates.

A large benchmark of the current baseline used all 15 possible four-strategy lineups, four cyclic seat rotations per lineup, and 50 repetitions per rotation:

- 3,000 games
- 12,000 player-game observations
- 2,000 observations per strategy

The resulting aggregate performance was:

| Strategy | Win Rate | Avg. Final VP |
| --- | ---: | ---: |
| `hybrid_ows` | 38.35% | 8.057 |
| `full_ows` | 30.80% | 7.785 |
| `roads_and_cities` | 24.80% | 7.124 |
| `five_resource` | 24.00% | 6.967 |
| `port` | 18.25% | 6.479 |
| `road_building` | 13.80% | 6.069 |

The benchmark also identified:

- a measurable first-seat advantage
- substantial opponent-lineup effects for some strategies
- strong behavioral differences between the six archetypes
- a particularly strong Largest Army tendency for the OWS strategies
- a particularly strong Longest Road tendency for `road_building`
- a clearly differentiated maritime-trading identity for `port`

See [Strategy Benchmark Findings](docs/strategy_benchmark_findings.md) for the full evaluation.

## Running the Strategy Benchmark

For a quick benchmark:

    python scripts/strategy_benchmark.py \
        --repetitions 5 \
        --validate-conservation \
        --output-dir results/strategy_benchmark

For the larger 3,000-game evaluation:

    python scripts/strategy_benchmark.py \
        --repetitions 50 \
        --validate-conservation \
        --output-dir results/strategy_benchmark_final

The benchmark produces per-player game records and aggregate strategy summaries.

## Analyzing Benchmark Results

Run:

    python scripts/analyze_strategy_benchmark.py \
        results/strategy_benchmark_final/player_games.csv \
        --output-dir results/strategy_benchmark_final/analysis

The analysis currently supports:

- strategy win-rate confidence intervals
- final-VP confidence intervals
- seat-position summaries
- strategy-by-seat performance
- lineup sensitivity
- pairwise strategy comparisons
- VP correlations

The per-game dataset also contains `game_id`, lineup, seat, repetition, and seed information, allowing shared-game and paired analyses.

## Current Strategy Policy

The six heuristic agents are now treated as a frozen baseline.

Several tuning experiments were evaluated, including:

- increased ore valuation for `road_building`
- stronger post-Longest-Road conversion behavior
- increased city priority for `port`
- phase-dependent city priority for `port`

These experiments either failed to improve performance consistently or produced only weak paired evidence. They were therefore reverted rather than retained through benchmark-specific overfitting.

The baseline agents should be interpreted as distinct heuristic archetypes, not six equally strong optimal policies.

## Research Direction

The current heuristic strategies provide a reproducible baseline for future decision-making systems.

Possible future agents include:

- stronger search-based agents
- Monte Carlo Tree Search
- reinforcement-learning agents
- learned policy or value models
- adaptive agents
- opponent-aware agents

The benchmark infrastructure provides a common evaluation framework for comparing these future agents with the fixed heuristic baseline.
