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

At the current core-v1 freeze point, the full test suite contains 417 passing tests.

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

## Agent Research Stack

CatanLab now includes four major levels of decision-making agents:

- heuristic `AdaptiveStrategyAgent` policies
- depth-2 expectimax search with `OneStepLookaheadAgent`
- behavior-cloned policies refined with DAgger
- KL-regularized PPO policies initialized from the DAgger policy

The heuristic and search policies are treated as frozen baselines.

The strongest validated learned policy is:

    results/rl_baselines/ppo_bckl_v1.pt

It was initialized from:

    results/rl_baselines/bc_dagger_v1.pt

and fine-tuned using:

    reward:              terminal win
    BC KL coefficient:   1.0
    learning rate:       1e-4
    PPO epochs:          2
    games/update:        32
    gamma:               0.99
    GAE lambda:          0.95
    selected checkpoint: update 15

In a fresh 4,000-game paired validation against BC + DAgger,
KL-PPO improved:

- win rate from 22.55% to 23.95%
- mean VP from 7.1327 to 7.2107

The paired win-rate improvement was +1.40 percentage points
with a 95% confidence interval of [+0.10, +2.75] percentage
points.

The paired mean-VP improvement was +0.0780 with a 95%
confidence interval of [+0.0215, +0.1340].

See [`docs/rl_baselines.md`](docs/rl_baselines.md) for the
learned-agent development and validation record.

<!-- SEARCH-BASELINE:START -->
## Search-agent baseline

CatanLab includes a depth-n same-turn expectimax agent,
`OneStepLookaheadAgent`, layered on top of the adaptive strategy policy.

The current validated configuration is:

- depth 2
- maritime-trade search enabled
- Road Building search enabled
- Year of Plenty search disabled
- Monopoly search disabled
- transposition cache disabled
- specialized fast search-state cloning enabled

In a 400-pair benchmark against `AdaptiveStrategyAgent`, with
`FIVE_RESOURCE` as the target strategy and the target rotated through all four
seats against `HYBRID_OWS`, `FULL_OWS`, and `PORT`:

| Metric | Adaptive | Search | Paired delta |
| --- | ---: | ---: | ---: |
| Win rate | 21.50% | 45.75% | +24.25 pp |
| Mean VP | 6.8625 | 8.3650 | +1.5025 |
| Roads | 11.980 | 9.175 | -2.805 |
| Settlements | 2.600 | 3.285 | +0.685 |
| Cities | 1.3925 | 1.7725 | +0.380 |
| Dev cards | 3.2000 | 3.8025 | +0.6025 |
| Runtime | 1.6677 s | 2.0634 s | +0.3957 s |

The search-agent win-rate and VP improvements were positive in every target
seat.

See [`docs/search_baseline.md`](docs/search_baseline.md) for search design,
information-safety constraints, optimization results, corrected maritime
results, development-card ablations, confidence intervals, and seat-level
analysis.
<!-- SEARCH-BASELINE:END -->

## Canonical Agent Benchmark

The major agent families were evaluated under a single
standardized 400-game protocol. Each agent received the same
game seeds, target-seat rotation, `FIVE_RESOURCE` target
strategy, and opponent lineup of `HYBRID_OWS`, `FULL_OWS`,
and `PORT`.

| Agent | Win Rate | Mean VP | Mean Turns | Seconds/Game |
| --- | ---: | ---: | ---: | ---: |
| Adaptive | 26.00% | 7.0050 | 118.58 | 1.7199 |
| Depth-2 search | 46.25% | 8.3375 | 105.30 | 2.0565 |
| BC + DAgger | 21.00% | 7.1600 | 113.56 | 1.6307 |
| KL-PPO | 22.25% | 7.1225 | 113.27 | 1.6323 |

Depth-2 search was decisively strongest.

Against KL-PPO, search achieved:

    win-rate difference: +24.00 percentage points
    95% CI:              [+18.50, +29.50]

    mean-VP difference:  +1.2150
    95% CI:              [+0.9775, +1.4525]

The 400-game canonical benchmark is intended to compare all
agent families under one common protocol. The smaller
KL-PPO-versus-BC improvement is supported by the separate
4,000-game paired validation described above.

Run the canonical benchmark with:

    python scripts/canonical_agent_benchmark.py         --games 400         --seed-offset 3600000         --output-csv results/rl_benchmarks/canonical_agents_400.csv

## Core v1 Scope

CatanLab Core v1 is treated as a stable research baseline.

It currently includes:

- a tested four-player Catan rules engine
- deterministic seeded simulation
- hidden-information-safe observations
- heuristic strategy agents
- depth-2 expectimax search
- a fixed neural observation encoder and legal-action space
- behavior cloning and DAgger
- PPO with GAE and BC-policy KL regularization
- paired bootstrap evaluation
- seat-controlled canonical benchmarking

Future work is intentionally separated from the Core v1
baseline. Planned directions include richer learned handling
of trading and special decisions, improved simulator realism,
and external validation against independent Catan
environments or bot populations.

