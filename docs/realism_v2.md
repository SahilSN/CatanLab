# Realism-v2 Learned Agent

## Overview

Realism-v2 is CatanLab's latest learned-agent research stage.

The goal was to imitate and approximate the behavior of the depth-2
Search-v2 teacher across the major decision families required for a
complete post-opening Catan policy.

The learned policy covers:

- ordinary turn actions
- robber tile selection
- robber victim selection
- discarding
- Monopoly resource selection
- Year of Plenty resource selection
- Road Building edge selection
- domestic trade proposals
- domestic trade responses
- domestic trade counters

The policy operates after the opening phase. Initial settlement and road
placement remain controlled by the `FIVE_RESOURCE` strategy opening policy.

## Teacher

The primary teacher is the canonical Search-v2 agent:

- search depth: 2
- transposition cache: disabled
- maritime-trade search: enabled
- Year of Plenty search: enabled
- Road Building search: enabled
- Monopoly search: enabled
- robber-decision search: enabled
- discard-decision search: enabled
- domestic-trade search: enabled

The canonical teacher dataset uses an observation dimension of 1138 and
an ordinary-action dimension of 202.

## Frozen Teacher Dataset

Training:

- seeds: 1,000,000 through 1,001,999
- games: 2,000
- examples: 1,124,131

Validation:

- seeds: 1,100,000 through 1,100,249
- games: 250
- examples: 142,277

## Baseline Behavior Cloning

The selected initial realism-v2 BC checkpoint was:

`results/realism_v2_bc/natural_baseline_spatial_v2/best.pt`

Its frozen validation performance was:

| Metric | Accuracy |
| --- | ---: |
| Overall | 0.7096 |
| Macro | 0.5987 |
| Ordinary action | 0.8256 |
| Robber tile | 0.2299 |
| Robber victim | 0.7208 |
| Discard | 0.7117 |
| Monopoly | 0.6525 |
| Year of Plenty | 0.4713 |
| Road Building | 0.2298 |
| Trade proposal | 0.4874 |
| Trade response | 0.8509 |
| Trade counter | 0.8072 |

## Internal Diagnosis

A 100-game intervention study showed that replacing only the learned
ordinary-action decisions with Search-v2 decisions produced the largest
observed improvement.

Relative to the learned model:

- teacher ordinary-action intervention: +1.97 mean VP
- teacher trade intervention: +0.02 mean VP

These intervention experiments should not be interpreted as an additive
causal decomposition. They were used diagnostically to identify ordinary
actions as the main corrective-training target.

## DAgger

The first DAgger corpus was collected on learner-controlled trajectories:

- games: 100
- seeds: 1,420,000 through 1,420,099
- ordinary-action examples: 5,117

Search-v2 labeled the ordinary decisions reached by the learner, while
the learned policy continued to control the actual game trajectory.

Several corrective-training formulations were evaluated.

### Full-model DAgger

A full-model continuation with R1 DAgger data did not improve game
performance over its matched continued-BC control.

### Ordinary-head-only DAgger

Training only the ordinary heads while freezing the shared backbone also
failed to improve fresh learner-state teacher agreement.

### Ordinary-only backbone adaptation

Allowing the backbone to adapt using only ordinary supervision improved
ordinary validation accuracy but damaged several other decision families.

This demonstrated that freezing the non-ordinary heads was insufficient:
changes to the shared representation still altered their behavior.

## Protected-Backbone DAgger

The successful formulation allowed the shared backbone and ordinary heads
to train while freezing the other heads, but replayed the complete
10-family frozen teacher corpus during adaptation.

This protected non-ordinary behavior while allowing the shared
representation to change.

Matched frozen validation:

| Model | Overall | Macro | Ordinary |
| --- | ---: | ---: | ---: |
| Original BC | 0.7096 | 0.5987 | 0.8256 |
| Protected control | 0.7194 | 0.6124 | 0.8304 |
| Protected DAgger R1 | 0.7189 | 0.6117 | 0.8292 |

Fresh learner-state Search-v2 agreement:

| Model | Agreement |
| --- | ---: |
| Original BC | 0.5222 |
| Protected control | 0.4841 |
| Protected DAgger R1 | 0.5949 |

In a fresh paired 100-game engineering evaluation:

| Model | Win rate | Mean VP |
| --- | ---: | ---: |
| Original BC | 0.100 | 6.390 |
| Protected control | 0.100 | 6.400 |
| Protected DAgger R1 | 0.170 | 6.860 |

Protected DAgger R1 versus matched control:

- mean VP delta: +0.460
- 95% paired-bootstrap CI: [+0.080, +0.840]
- win-rate delta: +0.070
- VP W/T/L: 44 / 30 / 26

This was the first corrective-training formulation that improved both
fresh teacher agreement and game-level performance.

## DAgger Round 2

A second DAgger dataset was collected from the successful R1 policy:

- games: 100
- seeds: 1,480,000 through 1,480,099
- ordinary examples: 5,094
- Search-v2 agreement during collection: 0.6064

R2 was compared against an exposure-matched continuation control.

Fresh agreement:

| Model | Agreement |
| --- | ---: |
| R1 | 0.5752 |
| R2 continuation control | 0.6817 |
| R2 DAgger | 0.6403 |

Fresh 100-game engineering evaluation:

| Model | Win rate | Mean VP |
| --- | ---: | ---: |
| R1 | 0.180 | 6.750 |
| R2 continuation control | 0.320 | 7.580 |
| R2 DAgger | 0.320 | 7.650 |

R2 DAgger minus the matched continuation control:

- mean VP delta: +0.070
- win-rate delta: +0.000
- VP W/T/L: 36 / 28 / 36

The new R2 learner-state data therefore did not provide a meaningful
incremental benefit over continued training on the existing corrective
distribution. R3 was not pursued.

## Archived Selected Checkpoint

For this project snapshot, the selected learned checkpoint is:

`results/realism_v2_protected_backbone/dagger_r1_repeat8/best.pt`

SHA-256:

`d08bc2055dde6f740c90adcfd55dba69d490e16bcf4d8c51db0809210274a8f1`

Checkpoint size:

`8,783,909 bytes`

Architecture:

- model: `RealismV2ActorCritic`
- observation dimension: 1138
- ordinary-action dimension: 202
- hidden dimension: 256

Protected R1 training:

- ordinary-head learning rate: 1e-3
- backbone learning rate: 1e-4
- DAgger repeat: 8
- frozen full-family validation macro: 0.6117056622

The checkpoint retains metadata inherited from its original BC checkpoint.
The protected-backbone result is recorded under
`ordinary_head_finetune.best_macro_accuracy`.

## Interpretation

The main internal result is that learner-state corrective supervision can
improve a learned Search-v2 approximation when the shared representation
is allowed to adapt while the full multi-family teacher corpus constrains
that representation.

The results also show that:

- frozen teacher-distribution accuracy is not a reliable proxy for
  learner-trajectory performance;
- changing a shared backbone can alter frozen special-decision heads;
- additional DAgger rounds are not automatically beneficial;
- matched continuation controls are necessary to distinguish the value of
  new learner-state data from continued optimization.
