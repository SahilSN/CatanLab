# Search-Agent Baseline

This document records the validated search-agent milestones for CatanLab.

## Production configuration

The current validated search configuration is:

- Search depth: 2
- Fast ordinary-search clone: enabled
- Transposition cache: disabled
- Maritime-trade search: enabled

The search agent retains heuristic handling for turn components that are not yet represented explicitly in the search tree.

## Depth selection

Depth 2 was selected over depth 3 after a paired benchmark with 400 games per depth.

| Metric | Depth 2 | Depth 3 |
| --- | ---: | ---: |
| Win rate | 31.00% | 30.50% |
| Average VP | 7.4600 | 7.4525 |
| Average runtime | 1.8279 s | 2.1368 s |

Paired changes for depth 3 minus depth 2:

- Win rate: -0.0050, 95% CI [-0.0234, +0.0134]
- Final VP: -0.0075, 95% CI [-0.0641, +0.0491]
- Runtime: +0.3089 s, 95% CI [+0.2542, +0.3635]

Depth 3 therefore produced no measurable strength improvement while increasing runtime by about 16.9%.

## Search-state cloning optimization

Profiling identified general-purpose `deepcopy` as the dominant cost in depth-2 search.

Before optimization:

- 100 search evaluations: 0.095725 s
- Clone calls: 100
- Clone time: 0.083860 s
- Average clone time: 838.60 us
- Clone share of search time: 87.61%

A specialized clone was introduced for ordinary search transitions. It shares the `Board`, which current ordinary search actions only read, while independently copying player state, inventories, development-deck storage, and the resource bank.

General `SearchState.clone()` retains full `deepcopy` semantics so that mutable board state, including the robber position, remains isolated for general callers.

After optimization:

- 100 search evaluations: 0.011002 s
- Clone calls: 100
- Clone time: 0.000956 s
- Average clone time: 9.56 us
- Clone share of search time: 8.69%

This reduced average clone cost by approximately 87.7x.

A subsequent real-game benchmark reduced depth-2 game runtime from roughly 2.96 seconds per game to roughly 1.90 seconds per game.

## Transposition cache

The depth-2 transposition cache was tested over 20 paired games after the cloning optimization.

| Variant | Average runtime |
| --- | ---: |
| Cached | 1.9021 s |
| Uncached | 1.8935 s |

Paired cached-minus-uncached runtime:

- +0.0086 s
- 95% CI [-0.0408, +0.0581]
- Mean paired speedup: 0.998x

Behavioral equivalence passed for all 20 paired games.

The cache therefore provides no measurable performance benefit at depth 2 and is disabled in the production configuration.

## Maritime-trade search

The initial search action space contained:

- Build city
- Build settlement
- Build road
- Buy development card
- Pass

Maritime trades were subsequently added as deterministic search actions. This allows depth-2 search to reason explicitly about same-turn sequences such as:

    maritime trade -> city
    maritime trade -> settlement
    maritime trade -> development card

A focused unit test verifies that depth-2 search selects a maritime trade when that trade enables a city on the following search ply.

### Paired benchmark

Maritime search was evaluated with 400 paired observations using identical seeds and seats.

| Metric | Maritime OFF | Maritime ON | Change |
| --- | ---: | ---: | ---: |
| Win rate | 20.00% | 36.50% | +16.50 pp |
| Average VP | 6.5000 | 7.7700 | +1.2700 |
| Roads | 7.2975 | 7.4475 | +0.1500 |
| Settlements | 2.5250 | 2.5625 | +0.0375 |
| Cities | 1.3975 | 1.8775 | +0.4800 |
| Development cards | 3.2175 | 4.4000 | +1.1825 |
| Longest Road rate | 21.00% | 17.25% | -3.75 pp |
| Average runtime | 1.7370 s | 1.9036 s | +0.1666 s |

Key paired confidence intervals:

- Win rate: +0.1650, 95% CI [+0.1138, +0.2162]
- Final VP: +1.2700, 95% CI [+1.0474, +1.4926]
- Cities: +0.4800, 95% CI [+0.3608, +0.5992]
- Development cards: +1.1825, 95% CI [+0.9587, +1.4063]
- Runtime: +0.1666 s, 95% CI [+0.1165, +0.2166]

The runtime increase is approximately 9.6%, while both win rate and average victory points improve substantially.

Maritime-trade search is therefore part of the validated production search configuration.

## Next search extension

The next planned action-space extension is development-card play search.

The intended order is:

1. Year of Plenty
2. Road Building
3. Monopoly
4. Knight

Year of Plenty is the first target because it is primarily a deterministic economic sequencing action and closely resembles the planning problem solved by maritime-trade search.
