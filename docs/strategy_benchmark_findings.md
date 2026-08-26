# Strategy Benchmark Findings

## 1. Overview

CatanLab currently contains six heuristic strategy archetypes:

1. `full_ows`
2. `hybrid_ows`
3. `road_building`
4. `roads_and_cities`
5. `five_resource`
6. `port`

The goal of the benchmark was not to tune all six strategies toward identical win rates.

Instead, the evaluation asked:

1. Do the strategies produce recognizable and distinct behavior?
2. Are their relative strengths and weaknesses stable across many games?
3. Are apparent weaknesses caused by implementation problems, or are they consequences of the strategy archetypes themselves?
4. How sensitive are strategies to seat position and opponent composition?

The results indicate that all six strategies have distinct behavioral signatures and that the major performance differences are stable.

## 2. Benchmark Design

There are 15 possible four-strategy subsets of the six available strategies.

Each lineup was evaluated using:

- 4 cyclic seat rotations
- 50 repetitions per rotation
- deterministic random seeds
- resource-conservation validation

This produces:

    15 lineups × 4 rotations × 50 repetitions = 3,000 games

Each game contains four players:

    3,000 games × 4 players = 12,000 player-game observations

Each strategy occurs in 10 of the 15 lineups, giving:

    10 lineups × 4 rotations × 50 repetitions = 2,000 observations per strategy

Every strategy therefore receives equal aggregate exposure and equal exposure to each seat.

## 3. Aggregate Results

| Strategy | Games | Win Rate | Avg. VP | Roads | Settlements | Cities | Dev Cards | LR Rate | LA Rate | Maritime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `full_ows` | 2000 | 30.80% | 7.79 | 7.39 | 0.81 | 1.73 | 9.84 | 5.15% | 72.00% | 10.33 |
| `hybrid_ows` | 2000 | 38.35% | 8.06 | 9.33 | 1.00 | 1.82 | 9.34 | 11.85% | 64.75% | 9.51 |
| `road_building` | 2000 | 13.80% | 6.07 | 12.96 | 3.24 | 0.66 | 2.94 | 45.40% | 0.00% | 9.09 |
| `roads_and_cities` | 2000 | 24.80% | 7.12 | 12.08 | 1.45 | 2.18 | 3.70 | 27.75% | 0.00% | 10.80 |
| `five_resource` | 2000 | 24.00% | 6.97 | 11.70 | 2.29 | 1.55 | 5.33 | 25.55% | 0.00% | 10.66 |
| `port` | 2000 | 18.25% | 6.48 | 12.76 | 2.99 | 0.99 | 3.97 | 34.25% | 0.00% | 12.52 |

## 4. Strategy Interpretations

### HYBRID OWS

`hybrid_ows` is the strongest heuristic strategy in the current baseline.

It combines the ore-wheat-sheep economic core with more expansion flexibility than `full_ows`.

Important characteristics include:

- 38.35% overall win rate
- 8.06 average final VP
- 64.75% Largest Army rate
- more road activity than `full_ows`
- substantial development-card purchasing

Its combination of economic development, development cards, and moderate expansion appears particularly effective.

### FULL OWS

`full_ows` is the second-strongest strategy.

Its clearest behavioral signature is development-card and Largest Army pursuit:

- 30.80% win rate
- 9.84 development-card purchases per game
- 72.00% Largest Army rate
- relatively low road and settlement expansion

The strategy strongly expresses the traditional ore-wheat-sheep archetype.

### ROADS AND CITIES

`roads_and_cities` forms part of the middle tier.

Its most distinctive statistic is city construction:

- 24.80% win rate
- 2.18 cities per game, highest among the six strategies
- 12.08 roads per game
- 27.75% Longest Road rate

It combines road-network growth with stronger conversion into cities than the road-focused alternatives.

### FIVE RESOURCE

`five_resource` is broadly comparable to `roads_and_cities`.

Its behavior is more balanced:

- 24.00% win rate
- 2.29 settlements per game
- 1.55 cities per game
- 5.33 development cards per game
- 25.55% Longest Road rate

The aggregate win-rate difference between `five_resource` and `roads_and_cities` is small enough that the two should not be strongly ranked relative to one another.

### PORT

`port` has a clear maritime-trading identity:

- 18.25% win rate
- 12.52 maritime trades per game, highest among all strategies
- 12.76 roads per game
- 2.99 settlements per game
- 34.25% Longest Road rate

The strategy is weaker than the middle-tier economic strategies but successfully executes its intended port and expansion behavior.

### ROAD BUILDING

`road_building` is the weakest strategy by overall win rate, but it successfully executes its defining objective:

- 13.80% win rate
- 12.96 roads per game
- 3.24 settlements per game
- 45.40% Longest Road rate, highest among all strategies

Its weakness arises primarily from poor conversion of road-network strength into other victory points:

- only 0.66 cities per game
- only 2.94 development-card purchases per game
- 6.07 average final VP

This indicates a strategically weak archetype rather than a failure to pursue Longest Road.

## 5. Win-Rate Confidence Intervals

The 95% Wilson intervals for aggregate win rates were:

| Strategy | Win Rate | 95% CI |
| --- | ---: | ---: |
| `five_resource` | 24.00% | 22.18% - 25.92% |
| `full_ows` | 30.80% | 28.82% - 32.86% |
| `hybrid_ows` | 38.35% | 36.24% - 40.50% |
| `port` | 18.25% | 16.62% - 20.00% |
| `road_building` | 13.80% | 12.36% - 15.38% |
| `roads_and_cities` | 24.80% | 22.96% - 26.74% |

The large differences between the strongest and weakest strategies are therefore not plausibly explained by ordinary simulation noise.

The confidence intervals for `five_resource` and `roads_and_cities` overlap substantially.

## 6. Seat Effects

The benchmark revealed a measurable first-seat advantage:

| Seat | Games | Win Rate | Avg. Final VP |
| --- | ---: | ---: | ---: |
| 0 | 3000 | 29.33% | 7.372 |
| 1 | 3000 | 23.23% | 6.975 |
| 2 | 3000 | 23.73% | 6.990 |
| 3 | 3000 | 23.70% | 6.983 |

Seat 0 therefore wins substantially more often than the other three seats.

However, every strategy receives equal exposure to every seat because cyclic seat rotations are used. The aggregate strategy comparison is therefore seat-balanced.

The strategy-by-seat win rates were:

| Strategy | Seat 0 | Seat 1 | Seat 2 | Seat 3 |
| --- | ---: | ---: | ---: | ---: |
| `five_resource` | 28.0% | 23.8% | 23.2% | 21.0% |
| `full_ows` | 37.4% | 29.4% | 27.2% | 29.2% |
| `hybrid_ows` | 43.0% | 34.4% | 37.2% | 38.8% |
| `port` | 22.0% | 16.4% | 18.8% | 15.8% |
| `road_building` | 16.6% | 12.4% | 11.2% | 15.0% |
| `roads_and_cities` | 29.0% | 23.0% | 24.8% | 22.4% |

The major strategy ordering remains visible across seat positions.

## 7. Pairwise Aggregate Comparisons

The aggregate pairwise analysis showed several large and statistically clear differences.

Examples include:

- `hybrid_ows` over `full_ows`: +7.55 percentage points
- `hybrid_ows` over `road_building`: +24.55 points
- `full_ows` over `road_building`: +17.00 points
- `full_ows` over `port`: +12.55 points
- `roads_and_cities` over `port`: +6.55 points
- `port` over `road_building`: +4.45 points

The comparison between `five_resource` and `roads_and_cities` was much smaller:

- win-rate difference: 0.80 percentage points in favor of `roads_and_cities`
- confidence interval for the win-rate difference included zero

The two middle-tier strategies are therefore broadly comparable by win rate.

## 8. Opponent-Lineup Sensitivity

Strategy performance varies depending on the three opposing archetypes.

| Strategy | Min Win Rate | Max Win Rate | Range |
| --- | ---: | ---: | ---: |
| `five_resource` | 13.5% | 33.5% | 20.0 pp |
| `full_ows` | 19.5% | 46.0% | 26.5 pp |
| `hybrid_ows` | 25.0% | 51.5% | 26.5 pp |
| `port` | 12.0% | 27.0% | 15.0 pp |
| `road_building` | 10.0% | 19.0% | 9.0 pp |
| `roads_and_cities` | 19.0% | 29.5% | 10.5 pp |

The two OWS strategies are particularly matchup-sensitive.

`road_building`, by contrast, is consistently weak rather than being harmed by only a few specific lineups.

## 9. Shared-Game Head-to-Head Results

A second comparison considered only games in which both members of a strategy pair were present.

Each pair shared 1,200 games.

Selected results:

| Strategy A | Strategy B | A Win Rate | B Win Rate | Difference |
| --- | --- | ---: | ---: | ---: |
| `hybrid_ows` | `full_ows` | 31.50% | 24.17% | +7.33 pp |
| `hybrid_ows` | `roads_and_cities` | 40.42% | 24.50% | +15.92 pp |
| `hybrid_ows` | `road_building` | 41.33% | 13.08% | +28.25 pp |
| `full_ows` | `port` | 32.83% | 18.83% | +14.00 pp |
| `full_ows` | `road_building` | 34.33% | 14.17% | +20.17 pp |
| `five_resource` | `roads_and_cities` | 22.75% | 24.25% | -1.50 pp |
| `port` | `road_building` | 17.08% | 14.25% | +2.83 pp |

The shared-game comparison preserves the main aggregate hierarchy.

## 10. Tuning Experiments

Several strategy-tuning experiments were performed after the initial benchmark.

The goal was to determine whether weaker strategies contained obvious heuristic defects.

### ROAD_BUILDING Post-Longest-Road Pivot

A stronger action-utility pivot was tested after `road_building` obtained Longest Road.

The modified policy reduced road utility and increased city utility more aggressively.

The experiment did not improve performance:

- win rate decreased
- average VP decreased
- settlement construction decreased
- city construction changed only slightly

The modification was reverted.

### ROAD_BUILDING Ore Weight

The `road_building` ore resource weight was experimentally increased from 0.3 to 0.7.

The aggregate benchmark initially appeared modestly favorable.

A paired comparison using 400 matched ROAD_BUILDING observations showed:

- win-rate change: +1.5 percentage points
- average final VP change: +0.095
- city change: +0.07
- Longest Road change: +2.25 percentage points
- settlement change: -0.12

Win transitions were:

- 26 baseline losses became wins
- 20 baseline wins became losses

For final VP:

- 106 paired observations improved
- 205 were unchanged
- 89 worsened

The approximate 95% interval for the paired mean VP change included zero.

The evidence was therefore considered too weak to justify permanently changing the heuristic, and the ore weight was restored to 0.3.

### PORT Static City Priority

PORT's base city utility was increased experimentally.

Increasing city utility from 3.0 to 5.0 substantially increased city construction but harmed expansion and overall performance.

A smaller increase from 3.0 to 4.0 also performed worse than the original strategy.

Both changes were reverted.

### PORT Phase-Dependent City Priority

A phase rule was then tested in which PORT retained its original expansion priorities before receiving an additional city bonus after establishing a minimum number of building sites.

A three-site trigger improved city construction but reduced win rate and VP.

A four-site trigger came much closer to the baseline:

- 19.00% win rate versus 19.25% baseline
- 6.45 VP versus 6.50 baseline

However, it did not meaningfully outperform the simpler original policy.

The phase rule was therefore also reverted.

## 11. Strategy Freeze Decision

The tuning experiments produced an important methodological conclusion.

A weaker strategy should not automatically be modified until its win rate approaches the stronger strategies.

The six agents are intended to represent different heuristic archetypes, and different archetypes can legitimately differ in strength.

Further tuning against the same benchmark would risk overfitting the heuristics to the benchmark rather than improving their conceptual fidelity.

The strategy policy layer is therefore frozen at the original heuristic configuration.

The benchmark results should be interpreted as characterization of the baseline, not as a target that future manual tuning should optimize directly.

## 12. Current Baseline Hierarchy

The evidence supports the following broad interpretation:

| Strategy | Interpretation |
| --- | --- |
| `hybrid_ows` | strongest overall; highly matchup-sensitive |
| `full_ows` | strong development-card and Largest Army strategy |
| `roads_and_cities` | stable middle-tier economic-expansion strategy |
| `five_resource` | balanced middle-tier strategy; comparable to roads-and-cities |
| `port` | coherent maritime/expansion strategy, but weaker overall |
| `road_building` | strong Longest Road execution but consistently weakest overall |

This hierarchy describes empirical performance, not an intended balance target.

## 13. Future Use

The six frozen heuristic agents now provide a reproducible baseline for more sophisticated Catan agents.

Future systems can be compared against them using the same benchmark infrastructure.

Potential directions include:

- Monte Carlo Tree Search
- reinforcement learning
- learned policy networks
- learned value functions
- adaptive strategy selection
- opponent modeling
- search-based planning

Because the heuristic layer is now fixed, improvements from future agents can be measured without simultaneously changing the comparison baseline.
