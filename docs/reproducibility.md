# Reproducibility

This document records the reproducible state of the CatanLab realism-v2
research branch at archival time.

## Code Snapshot

Development branch:

    realism-v2

Base revision before archival changes:

    9142c54dd1a49772a5a1413dd7c96e9ea78a1525

The final archival commit should be recorded here after it is created.

## Validated Environment

Validated on:

- Python 3.10.0
- NumPy 2.2.6
- PyTorch 2.11.0

See `pyproject.toml` for package requirements.

## Verification

At archival time:

- realism-v2 targeted tests: 126 passed
- complete CatanLab test suite: 608 passed
- `git diff --check`: clean
- realism-v2 modules and scripts pass `python -m py_compile`

Run:

    pytest -q

## Generated Artifacts

Large generated datasets, checkpoints, and experiment outputs are not
tracked directly in Git.

The following paths are intentionally ignored:

- `data/`
- `results/realism_v2_bc/`
- `results/realism_v2_dagger/`
- `results/realism_v2_game_benchmark/`
- `results/realism_v2_interventions/`
- `results/realism_v2_ordinary_backbone/`
- `results/realism_v2_ordinary_heads/`
- `results/realism_v2_protected_backbone/`

The frozen teacher corpus is approximately 8.8 GB and the DAgger datasets
are approximately 173 MB.

## Selected Checkpoint

Selected archived model:

    results/realism_v2_protected_backbone/dagger_r1_repeat8/best.pt

SHA-256:

    d08bc2055dde6f740c90adcfd55dba69d490e16bcf4d8c51db0809210274a8f1

Size:

    8,783,909 bytes

Model configuration:

- model class: realism_v2 / `RealismV2ActorCritic`
- observation dimension: 1138
- ordinary-action dimension: 202
- hidden dimension: 256

## Canonical Teacher Corpus

Training corpus:

    data/teacher_v2/realism-v2-v1/train/train.jsonl

- seed start: 1,000,000
- games: 2,000
- seed range: 1,000,000 through 1,001,999

Validation corpus:

    data/teacher_v2/realism-v2-v1/validation/train.jsonl

- seed start: 1,100,000
- games: 250
- seed range: 1,100,000 through 1,100,249

The canonical teacher uses:

- search depth 2
- transposition cache disabled
- maritime search enabled
- Year of Plenty search enabled
- Road Building search enabled
- Monopoly search enabled
- robber search enabled
- discard search enabled
- domestic-trade search enabled
- conservation validation enabled

## R1 DAgger Corpus

    data/dagger_v2/ordinary_r1_n100/train.jsonl

- games: 100
- seed range: 1,420,000 through 1,420,099
- ordinary examples: 5,117

Generation command:

    python scripts/generate_realism_v2_dagger.py \
        --model <baseline-checkpoint> \
        --output-dir data/dagger_v2/ordinary_r1_n100 \
        --seed-start 1420000 \
        --seed-count 100 \
        --max-turns 2000

## Protected-Backbone R1 Training

The selected protected-backbone DAgger run used:

    python scripts/train_realism_v2_ordinary_heads.py \
        --init-checkpoint \
        results/realism_v2_bc/natural_baseline_spatial_v2/best.pt \
        --base-train \
        data/teacher_v2/realism-v2-v1/train/train.jsonl \
        --extra-train \
        data/dagger_v2/ordinary_r1_n100/train.jsonl \
        --extra-repeat 8 \
        --extra-every 27 \
        --validation \
        data/teacher_v2/realism-v2-v1/validation/train.jsonl \
        --output-dir \
        results/realism_v2_protected_backbone/dagger_r1_repeat8 \
        --epochs 2 \
        --batch-size 128 \
        --lr 1e-3 \
        --backbone-lr 1e-4 \
        --shuffle-buffer-batches 256 \
        --seed 0 \
        --log-every 250

Trainable parameters in this formulation are the shared backbone and the
five ordinary-action heads. Other learned decision heads are frozen.

The entire 10-family teacher corpus is replayed so special-decision losses
still constrain the shared representation.

## Internal Opponent Protocol

Normal learned-agent game benchmarks use one learned target player against:

- `HYBRID_OWS`
- `FULL_OWS`
- `PORT`

The target strategy is `FIVE_RESOURCE`.

The target seat rotates as:

    game_index % 4

The learned agent controls post-opening play. Opening placement is handled
by the strategy opening policy.

## Engineering Seed Blocks

Engineering and diagnostic seed ranges used during development include:

- 1,420,000+: R1 DAgger collection
- 1,430,000+: first three-way learned evaluation
- 1,440,000+: fresh agreement
- 1,450,000+: ordinary-head agreement
- 1,460,000+: protected-backbone agreement
- 1,470,000+: protected R1 game evaluation
- 1,480,000+: R2 DAgger collection
- 1,490,000+: R2 agreement
- 1,500,000+: R2 game evaluation
- 1,510,000+: checkpoint-selection engineering evaluation

## Reserved Final Evaluation Seeds

The `1,200,000+` final-evaluation seed block was deliberately reserved
during development and was not consumed before this archival snapshot.

No result in this repository should be described as having used that
locked final-evaluation block unless a future continuation of the project
explicitly performs that experiment.

## External Validation

Colonist.io validation was planned but not performed before archival.

No external Colonist bot result should be attributed to this snapshot.
