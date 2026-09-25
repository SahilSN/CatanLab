# Future Work

CatanLab was intentionally shelved after the realism-v2 internal
development and model-selection stage.

The following items are known future work rather than accidental omissions.

## Locked Final Internal Evaluation

A final seed block beginning at 1,200,000 was reserved throughout
development.

It remains unused.

A future continuation could freeze a complete evaluation protocol and run
the selected checkpoint once on that held-out block.

## External Validation

The main planned external-validation target is Colonist.io.

The intended experiment is to evaluate the frozen CatanLab policy against
independently implemented Catan bots.

Required work includes:

1. external game-state parsing;
2. mapping external board/state information into CatanLab observations;
3. legal-action translation;
4. action translation back into the external environment;
5. adapter correctness testing;
6. predefined bot-strength and seat-rotation protocols;
7. external performance comparison with internal results.

The learned model should remain frozen during external validation.

## Opening Policy

Realism-v2 does not learn initial settlement and road placement.

The archived agent uses:

    StrategyOpeningAgent(FIVE_RESOURCE)

for the opening and the neural realism-v2 policy for post-opening play.

A future project could learn the opening phase as part of an end-to-end
policy.

## Further DAgger

Protected-backbone DAgger R1 produced the clearest successful corrective
training result.

A second DAgger round did not meaningfully outperform its matched
continuation control.

R3 was intentionally not pursued.

Any future additional DAgger work should begin with a new hypothesis
rather than simply adding rounds.

## Opponent Diversity

Internal learned-agent evaluation currently uses the fixed opponent set:

- `HYBRID_OWS`
- `FULL_OWS`
- `PORT`

Broader opponent distributions, adaptive opponents, self-play, or
external bots would provide a stronger test of generalization.

## Packaging

The selected model checkpoint is currently kept outside Git because
generated checkpoints and experiment artifacts are ignored.

A future public release could publish the selected checkpoint as a GitHub
Release asset and record its SHA-256 in the release notes.
