# RL Baselines

This document records the main learned-agent baselines and their validation results.

## Baseline progression

    Adaptive heuristic
           ↓
    Depth-2 search
           ↓
    Search teacher
           ↓
    Behavior cloning
           ↓
    DAgger
           ↓
    BC-anchored PPO

## Frozen learned baselines

### BC + DAgger

Checkpoint:

    results/rl_baselines/bc_dagger_v1.pt

This was the strongest validated learned policy before PPO fine-tuning.

### KL-regularized PPO

Checkpoint:

    results/rl_baselines/ppo_bckl_v1.pt

Training configuration:

    initialization:      bc_dagger_v1.pt
    reward:              win
    BC KL coefficient:   1.0
    learning rate:       1e-4
    PPO epochs:          2
    games/update:        32
    batch size:          512
    gamma:               0.99
    GAE lambda:          0.95
    potential shaping:   off
    selected checkpoint: update 15

## Final paired validation

The selected PPO checkpoint was compared against BC + DAgger over 4000 fresh paired games.

### Win rate

    BC + DAgger:   0.2255
    KL-PPO:        0.2395
    difference:   +0.0140
    95% CI:       [+0.0010, +0.0275]

### Mean victory points

    BC + DAgger:   7.1327
    KL-PPO:        7.2107
    difference:   +0.0780
    95% CI:       [+0.0215, +0.1340]

### Paired wins

    BC + DAgger only: 340
    KL-PPO only:      396

Both the win-rate and mean-VP confidence intervals are entirely above zero, so the PPO checkpoint is promoted as the strongest validated learned policy.

## Current status

    best validated learned policy:
        results/rl_baselines/ppo_bckl_v1.pt

    best frozen search policy:
        OneStepLookaheadAgent, depth 2

The next evaluation milestone is a direct comparison between the frozen KL-PPO learned policy and the frozen depth-2 search policy.

## Learned policy vs depth-2 search

The frozen KL-PPO policy was compared directly against the
frozen depth-2 search agent over 400 fresh paired games.

Search configuration:

    search depth:          2
    transposition cache:   off
    maritime trades:       on
    Year of Plenty:        off
    Road Building:         on
    Monopoly:              off

Win rate:

    depth-2 search:  0.4725
    KL-PPO:          0.2125
    PPO - search:   -0.2600
    95% CI:         [-0.3200, -0.2000]

Mean victory points:

    depth-2 search:  8.3750
    KL-PPO:          7.0850
    PPO - search:   -1.2900
    95% CI:         [-1.5300, -1.0574]

Paired exclusive wins:

    search only: 137
    KL-PPO only: 33

The depth-2 search policy therefore remains substantially
stronger than the best validated learned policy.

Full-game wall-clock timing was:

    depth-2 search: 2.1322 seconds/game
    KL-PPO:         1.7126 seconds/game
    ratio:          1.25x

These timings measure complete game execution and should not
be interpreted as isolated agent decision latency.

