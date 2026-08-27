# Search Agent Baseline

This document records the validated search-agent configuration for CatanLab,
the experiments used to select that configuration, and the final comparison
against the non-search adaptive heuristic agent.

## Final configuration

The current practical search baseline is:

- search depth: 2
- maritime-trade search: enabled
- Road Building search: enabled
- Year of Plenty search: disabled
- Monopoly search: disabled
- transposition cache: disabled
- specialized fast search-state cloning: enabled

In code, the intended configuration is equivalent to:

    OneStepLookaheadAgent(
        strategy,
        search_depth=2,
        use_transposition_cache=False,
        search_maritime_trades=True,
        search_year_of_plenty=False,
        search_road_building=True,
        search_monopoly=False,
    )

`OneStepLookaheadAgent` retains its historical class name, but it now performs
depth-n same-turn deterministic / expectimax search rather than only literal
one-step lookahead.

## Search model

Ordinary same-turn search currently considers:

- build city
- build settlement
- build road
- buy development card
- maritime trade
- pass

Domestic trades are not included in the search tree.

Development-card purchases are modeled as chance nodes. Search consumes one
unknown physical card from the hypothetical deck without reading its hidden
identity, then evaluates possible card identities according to an
information-safe development-card belief.

The search evaluator rewards:

- victory points
- productive settlement and city locations
- progress toward useful build costs
- Longest Road development
- future legal settlement access
- development-card option value
- current Longest Road / Largest Army ownership

Raw resource-card count is intentionally not rewarded independently of its
strategic usefulness.

## Search-state cloning

The initial implementation used full `deepcopy` cloning for every hypothetical
transition.

Profiling showed cloning was the dominant search cost.

A specialized fast clone was therefore introduced for ordinary search actions.
It shares the static `Board` object while cloning mutable:

- player state
- inventories
- development-card deck
- bank state

This is safe for the current ordinary search action set because those actions
do not mutate the board itself.

The general-purpose full clone remains available for operations that may mutate
board state.

### Clone profiling

Before specialized cloning:

- depth-2 microbenchmark total: 0.095725 s
- clone calls: 100
- clone time: 0.083860 s
- average clone: 838.60 us
- clone share of search time: 87.61%

After specialized cloning:

- depth-2 microbenchmark total: 0.011002 s
- clone time: 0.000956 s
- average clone: 9.56 us
- clone share of search time: 8.69%

This corresponds to approximately:

- 87.7x faster cloning
- 8.7x faster microbenchmark search

Real-game search runtime also fell substantially after this optimization.

## Search-depth selection

### Depth 1 vs depth 2

Paired benchmark:

- 200 paired observations

Depth 1:

- win rate: 0.220
- average VP: 7.015
- average runtime: 1.8625 s

Depth 2:

- win rate: 0.290
- average VP: 7.340
- average runtime: 2.8198 s

Paired depth-2 minus depth-1 effects:

- win rate: +0.070
  - 95% CI: [+0.0135, +0.1265]
- VP: +0.325
  - 95% CI: [+0.084, +0.566]
- runtime: +0.9573 s

Depth 2 therefore produced a meaningful strength improvement over depth 1.

### Depth 2 vs depth 3

After clone optimization, a 100-repetition / 400-observation comparison found:

Depth 2:

- win rate: 0.3100
- average VP: 7.4600
- runtime: 1.8279 s

Depth 3:

- win rate: 0.3050
- average VP: 7.4525
- runtime: 2.1368 s

Paired depth-3 minus depth-2 effects:

- win rate: -0.005
- VP: -0.0075
- runtime: +0.3089 s

Strength differences were effectively neutral while depth 3 was measurably
slower.

Therefore:

**depth 2 is the practical search depth.**

## Transposition-cache ablation

After the fast-clone optimization, caching was benchmarked again.

Average runtime:

- cache enabled: 1.9021 s
- cache disabled: 1.8935 s

Paired runtime difference:

- +0.0086 s
- 95% CI: [-0.0408, +0.0581]

The cache did not provide a useful depth-2 speedup.

Therefore:

**transposition caching is disabled in the practical baseline.**

## Maritime-trade search

An early maritime benchmark contained a seat/strategy alignment bug: the target
search agent rotated seats while the target FIVE_RESOURCE strategy did not.

Those results are invalid and should not be used.

The benchmark was corrected so that the target FIVE_RESOURCE strategy and its
search agent rotate seats together.

### Corrected 400-pair result

Maritime search OFF:

- win rate: 0.2725
- average VP: 7.315
- roads: 9.715
- settlements: 3.4225
- cities: 1.2825
- dev cards: 3.0075
- Longest Road rate: 0.2925
- runtime: 1.8914 s

Maritime search ON:

- win rate: 0.4425
- average VP: 8.270
- roads: 8.955
- settlements: 3.2100
- cities: 1.7600
- dev cards: 3.8975
- Longest Road rate: 0.2825
- runtime: 2.0345 s

Paired ON minus OFF effects:

- win rate: +0.1700
  - 95% CI: [+0.1123, +0.2277]
- VP: +0.9550
  - 95% CI: [+0.7161, +1.1939]
- roads: -0.7600
  - 95% CI: [-1.0928, -0.4272]
- settlements: -0.2125
  - 95% CI: [-0.3668, -0.0582]
- cities: +0.4775
  - 95% CI: [+0.3441, +0.6109]
- dev cards: +0.8900
  - 95% CI: [+0.6699, +1.1101]
- Longest Road rate: -0.0100
- runtime: +0.1431 s
  - 95% CI: [+0.0856, +0.2006]

This is one of the largest validated improvements in the search system.

Therefore:

**maritime-trade search is enabled.**

## Development-card search

Development-card search was introduced conservatively behind independent
feature flags.

### Year of Plenty: resource-pair search

The first experiment retained the heuristic decision about whether to play
Year of Plenty and searched only over the 15 unordered resource pairs.

Full 400-pair result, search ON minus OFF:

- win rate: -0.005
  - 95% CI: [-0.0319, +0.0219]
- VP: -0.0525
  - 95% CI: [-0.1435, +0.0385]
- runtime: +0.0942 s
  - 95% CI: [+0.0537, +0.1346]

There was no strength improvement and there was a measurable runtime cost.

### Year of Plenty: HOLD-vs-PLAY search

The second experiment allowed post-roll search to compare:

- holding Year of Plenty
- playing each legal resource pair

Full 400-pair result, search ON minus OFF:

- win rate: +0.0025
  - 95% CI: [-0.0060, +0.0110]
- VP: +0.0150
  - 95% CI: [-0.0100, +0.0400]
- runtime: +0.0777 s
  - 95% CI: [+0.0421, +0.1133]

Again, strength was effectively neutral while runtime increased.

Therefore:

**Year of Plenty search is disabled.**

The implementation is retained as an experimental ablation.

## Road Building search

Road Building was a stronger candidate because road placement is combinatorial.
The first free road can alter the legal set for the second road, and the final
road topology can affect:

- future settlement access
- Longest Road
- same-turn continuation value

The search compares HOLD against legal one- and two-road sequences. Second-road
legality is recomputed after the first hypothetical road.

### 400-pair result

Road Building search OFF:

- win rate: 0.4425
- average VP: 8.270
- roads: 8.955
- settlements: 3.210
- cities: 1.7600
- dev cards: 3.8975
- Longest Road rate: 0.2825
- runtime: 2.0099 s

Road Building search ON:

- win rate: 0.4575
- average VP: 8.365
- roads: 9.175
- settlements: 3.285
- cities: 1.7725
- dev cards: 3.8025
- Longest Road rate: 0.2900
- runtime: 2.1290 s

Paired ON minus OFF effects:

- win rate: +0.0150
  - 95% CI: [-0.0090, +0.0390]
- VP: +0.0950
  - 95% CI: [+0.0083, +0.1817]
- roads: +0.2200
  - 95% CI: [+0.0829, +0.3571]
- settlements: +0.0750
  - 95% CI: [+0.0019, +0.1481]
- cities: +0.0125
- dev cards: -0.0950
  - 95% CI: [-0.1760, -0.0140]
- Longest Road rate: +0.0075
- runtime: +0.1191 s
  - 95% CI: [+0.0576, +0.1806]

The VP improvement was positive, while win-rate improvement was directional but
uncertain.

### Seat robustness

Paired VP effect by target seat:

- seat 0: +0.21
- seat 1: +0.04
- seat 2: +0.03
- seat 3: +0.10

All four seats had positive average VP changes.

Therefore:

**Road Building search is enabled.**

## Monopoly search

Monopoly requires special care because opponent resource identities are hidden.

The search implementation therefore does not inspect actual opponent resource
composition. It constructs a public-information belief using:

- publicly visible opponent hand size
- visible settlement/city production profile

For each candidate Monopoly resource, search evaluates a distribution over
possible numbers of cards collected and computes an expectimax continuation
value.

Hidden-information invariance tests verify that changing an opponent's private
resource composition while holding public information fixed does not change
the Monopoly search decision.

### 40-pair pilot

Monopoly search OFF:

- win rate: 0.525
- average VP: 8.55
- runtime: 2.1578 s

Monopoly search ON:

- win rate: 0.500
- average VP: 8.50
- runtime: 2.8211 s

Paired ON minus OFF effects:

- win rate: -0.0250
  - 95% CI: [-0.0740, +0.0240]
- VP: -0.0500
  - 95% CI: [-0.2061, +0.1061]
- runtime: +0.6633 s
  - 95% CI: [+0.1226, +1.2039]

The pilot showed no strength signal and a large runtime increase.

A 400-pair follow-up was therefore not run.

Therefore:

**Monopoly search is disabled.**

The information-safe implementation is retained as an experimental ablation.

## Final search agent vs AdaptiveStrategyAgent

The finalized search agent was compared directly with the original
`AdaptiveStrategyAgent`.

The target strategy was FIVE_RESOURCE in both conditions.

Opponent strategies were:

- HYBRID_OWS
- FULL_OWS
- PORT

The target was rotated through all four seats. Each paired observation used the
same seed and target seat.

Total:

- 400 paired observations

### Aggregate result

AdaptiveStrategyAgent:

- win rate: 0.2150
- average VP: 6.8625
- roads: 11.980
- settlements: 2.600
- cities: 1.3925
- dev cards: 3.2000
- Longest Road rate: 0.3175
- runtime: 1.6677 s

Final search agent:

- win rate: 0.4575
- average VP: 8.3650
- roads: 9.175
- settlements: 3.285
- cities: 1.7725
- dev cards: 3.8025
- Longest Road rate: 0.2900
- runtime: 2.0634 s

Paired search minus adaptive effects:

- win rate: +0.2425
  - 95% CI: [+0.1825, +0.3025]
- VP: +1.5025
  - 95% CI: [+1.2358, +1.7692]
- roads: -2.8050
  - 95% CI: [-3.1332, -2.4768]
- settlements: +0.6850
  - 95% CI: [+0.4983, +0.8717]
- cities: +0.3800
  - 95% CI: [+0.2174, +0.5426]
- dev cards: +0.6025
  - 95% CI: [+0.3996, +0.8054]
- Longest Road rate: -0.0275
  - 95% CI: [-0.0814, +0.0264]
- runtime: +0.3957 s
  - 95% CI: [+0.3137, +0.4777]

The final search agent therefore gained:

- +24.25 percentage points in win rate
- +1.5025 average victory points

for approximately:

- +0.396 seconds of runtime per game

in this benchmark matchup.

### Effect by seat

Paired search-minus-adaptive effects:

| Seat | Win-rate delta | VP delta | Roads | Settlements | Cities | Dev cards |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | +0.23 | +1.12 | -3.82 | +0.49 | +0.36 | +0.74 |
| 1 | +0.24 | +1.62 | -2.44 | +0.85 | +0.36 | +0.56 |
| 2 | +0.20 | +1.51 | -2.64 | +0.50 | +0.45 | +0.51 |
| 3 | +0.30 | +1.76 | -2.32 | +0.90 | +0.35 | +0.60 |

Win rate and VP improved in every seat.

The aggregate strength improvement is therefore not explained by a single
favorable seating position.

## Behavioral interpretation

The largest behavioral difference between the final search agent and the
adaptive heuristic is resource allocation.

Search builds substantially fewer roads:

- -2.805 roads per game

while producing more:

- settlements: +0.685
- cities: +0.380
- development cards: +0.6025

Despite the large reduction in roads, Longest Road ownership does not change
meaningfully.

This suggests that the adaptive heuristic frequently invests in additional
roads whose opportunity cost exceeds their strategic value. Same-turn search
is better able to compare those road expenditures against settlement, city,
development-card, and trading continuations.

## Final ablation decisions

| Component | Decision | Reason |
| --- | --- | --- |
| Search depth 2 | KEEP | Clear gain over depth 1 |
| Search depth 3 | REJECT | No strength gain; slower |
| Specialized fast clone | KEEP | Large search-speed improvement |
| Transposition cache | REJECT | No useful depth-2 speedup |
| Maritime-trade search | KEEP | Large win-rate and VP improvement |
| YOP pair selection | REJECT | Neutral/slightly negative; slower |
| YOP HOLD-vs-PLAY | REJECT | Neutral; slower |
| Road Building search | KEEP | Small positive VP improvement |
| Monopoly expectimax | REJECT | Neutral/negative pilot; expensive |

## Current status

The search agent described here is the current primary agent baseline for
CatanLab.

Further search extensions should be treated as experimental and benchmarked
against this configuration rather than against older intermediate search
versions.
