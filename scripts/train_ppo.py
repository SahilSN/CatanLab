from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from catanlab.game import run_game
from catanlab.ppo import (
    compute_gae,
    ppo_update,
    public_vp_margin_potential,
    terminal_reward,
)
from catanlab.rl_agent import (
    PPORolloutAgent,
)
from catanlab.rl_model import (
    CatanActorCritic,
)
from catanlab.strategies import (
    StrategyType,
)
from catanlab.turns import (
    AdaptiveStrategyAgent,
)


def load_bc_model(
    checkpoint_path: Path,
) -> tuple[
    CatanActorCritic,
    dict,
]:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )

    model = CatanActorCritic(
        observation_dim=checkpoint[
            "observation_dim"
        ],
        action_dim=checkpoint[
            "action_dim"
        ],
        hidden_dim=checkpoint[
            "hidden_dim"
        ],
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    return model, checkpoint


def zero_value_head(
    model: CatanActorCritic,
) -> None:
    """
    Preserve the pretrained policy while discarding
    the behavior-cloning model's unsupervised critic.
    """
    with torch.no_grad():
        model.value_head.weight.zero_()
        model.value_head.bias.zero_()


def collect_episode(
    model: CatanActorCritic,
    game_index: int,
    seed_offset: int,
    reward_mode: str,
    potential_alpha: float,
    gamma: float,
    gae_lambda: float,
):
    target_seat = (
        game_index % 4
    )

    target_strategy = (
        StrategyType.FIVE_RESOURCE
    )

    opponent_strategies = [
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

    strategies = []
    agents = []

    ppo_agent = None
    opponent_index = 0

    for seat in range(4):
        if seat == target_seat:
            strategies.append(
                target_strategy
            )

            ppo_agent = PPORolloutAgent(
                target_strategy,
                model=model,
                deterministic=False,
                seed=(
                    seed_offset
                    + game_index
                ),
            )

            agents.append(
                ppo_agent
            )

        else:
            strategy = (
                opponent_strategies[
                    opponent_index
                ]
            )

            opponent_index += 1

            strategies.append(
                strategy
            )

            agents.append(
                AdaptiveStrategyAgent(
                    strategy
                )
            )

    game_seed = (
        seed_offset
        + game_index
    )

    result = run_game(
        strategies=strategies,
        seed=game_seed,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    if ppo_agent is None:
        raise RuntimeError(
            "PPO agent was not created."
        )

    won = (
        result.winner_id
        == target_seat
    )

    target_vp = float(
        result.players[
            target_seat
        ].victory_points
    )

    opponent_vps = [
        float(
            result.players[
                seat
            ].victory_points
        )
        for seat in range(
            len(result.players)
        )
        if seat != target_seat
    ]

    if not opponent_vps:
        raise RuntimeError(
            "PPO training requires at least "
            "one opponent."
        )

    best_opponent_vp = max(
        opponent_vps
    )

    episode_reward = terminal_reward(
        won=won,
        final_vp=target_vp,
        reward_mode=reward_mode,
        best_opponent_vp=(
            best_opponent_vp
        ),
    )

    terminal_potential = (
        public_vp_margin_potential(
            result.players,
            target_seat,
        )
    )

    if potential_alpha > 0.0:
        ppo_agent.rollout.finish_episode_with_potential(
            terminal_reward=episode_reward,

            # The environment transitions into an
            # absorbing terminal state whose potential
            # is defined as zero. This keeps the shaping
            # term policy-invariant.
            terminal_potential=0.0,

            shaping_alpha=potential_alpha,
            gamma=gamma,
        )
    else:
        ppo_agent.rollout.finish_episode(
            episode_reward
        )

    tensors = (
        ppo_agent.rollout
        .to_tensors()
    )

    total_transition_reward = float(
        tensors["rewards"].sum().item()
    )

    shaping_return = (
        total_transition_reward
        - episode_reward
    )

    mean_abs_reward = float(
        tensors["rewards"]
        .abs()
        .mean()
        .item()
    )

    max_abs_reward = float(
        tensors["rewards"]
        .abs()
        .max()
        .item()
    )

    potentials = tensors[
        "potentials"
    ]

    next_potentials = torch.empty_like(
        potentials
    )

    if len(potentials) > 1:
        next_potentials[:-1] = (
            potentials[1:]
        )

    # Absorbing terminal state has Phi = 0.
    next_potentials[-1] = 0.0

    shaping_rewards = (
        potential_alpha
        * (
            gamma
            * next_potentials
            - potentials
        )
    )

    mean_abs_shaping = float(
        shaping_rewards
        .abs()
        .mean()
        .item()
    )

    max_abs_shaping = float(
        shaping_rewards
        .abs()
        .max()
        .item()
    )

    nonzero_shaping_fraction = float(
        (
            shaping_rewards.abs()
            > 1e-8
        )
        .float()
        .mean()
        .item()
    )

    initial_potential = float(
        tensors["potentials"][0].item()
    )

    advantages, returns = (
        compute_gae(
            tensors["rewards"],
            tensors["dones"],
            tensors["old_values"],
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
    )

    return {
        "observations": (
            tensors["observations"]
        ),
        "legal_masks": (
            tensors["legal_masks"]
        ),
        "action_ids": (
            tensors["action_ids"]
        ),
        "old_log_probs": (
            tensors["old_log_probs"]
        ),
        "advantages": advantages,
        "returns": returns,
        "won": int(won),
        "vp": target_vp,
        "reward": episode_reward,
        "shaping_return": shaping_return,
        "mean_abs_reward": mean_abs_reward,
        "max_abs_reward": max_abs_reward,
        "mean_abs_shaping": (
            mean_abs_shaping
        ),
        "max_abs_shaping": (
            max_abs_shaping
        ),
        "nonzero_shaping_fraction": (
            nonzero_shaping_fraction
        ),
        "initial_potential": initial_potential,
        "final_game_potential": (
            terminal_potential
        ),
        "turns": (
            result.turns_played
        ),
        "seat": target_seat,
    }


def concatenate_episodes(
    episodes,
):
    tensor_keys = (
        "observations",
        "legal_masks",
        "action_ids",
        "old_log_probs",
        "advantages",
        "returns",
    )

    return {
        key: torch.cat(
            [
                episode[key]
                for episode in episodes
            ],
            dim=0,
        )
        for key in tensor_keys
    }


def save_checkpoint(
    path: Path,
    model: CatanActorCritic,
    *,
    observation_dim: int,
    action_dim: int,
    hidden_dim: int,
    update_index: int,
    total_games: int,
    total_wins: int,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": (
                model.state_dict()
            ),
            "model_class": "flat",
            "observation_dim": (
                observation_dim
            ),
            "action_dim": (
                action_dim
            ),
            "hidden_dim": (
                hidden_dim
            ),
            "ppo_update": (
                update_index
            ),
            "ppo_total_games": (
                total_games
            ),
            "ppo_total_wins": (
                total_wins
            ),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--updates",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--games-per-update",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--ppo-epochs",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--clip-epsilon",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--value-coefficient",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--entropy-coefficient",
        type=float,
        default=0.01,
    )

    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--reward-mode",
        choices=(
            "win",
            "vp",
            "vp_win",
            "win_margin",
        ),
        default="win",
    )

    parser.add_argument(
        "--potential-alpha",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--bc-kl-coefficient",
        type=float,
        default=0.0,
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--seed-offset",
        type=int,
        default=200000,
    )

    parser.add_argument(
        "--optimizer-seed",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    if args.updates <= 0:
        raise ValueError(
            "--updates must be positive."
        )

    if args.games_per_update <= 0:
        raise ValueError(
            "--games-per-update must be positive."
        )

    if args.potential_alpha < 0.0:
        raise ValueError(
            "--potential-alpha must be "
            "nonnegative."
        )

    if args.bc_kl_coefficient < 0.0:
        raise ValueError(
            "--bc-kl-coefficient must be "
            "nonnegative."
        )

    if not (
        0.0 < args.gamma <= 1.0
    ):
        raise ValueError(
            "--gamma must be in (0, 1]."
        )

    if not (
        0.0 <= args.gae_lambda <= 1.0
    ):
        raise ValueError(
            "--gae-lambda must be in [0, 1]."
        )

    model, source_checkpoint = (
        load_bc_model(
            args.model
        )
    )

    reference_model, _ = (
        load_bc_model(
            args.model
        )
    )

    reference_model.eval()

    for parameter in (
        reference_model.parameters()
    ):
        parameter.requires_grad_(False)

    zero_value_head(
        model
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
    )

    shuffle_generator = (
        torch.Generator()
    )

    shuffle_generator.manual_seed(
        args.optimizer_seed
    )

    observation_dim = int(
        source_checkpoint[
            "observation_dim"
        ]
    )

    action_dim = int(
        source_checkpoint[
            "action_dim"
        ]
    )

    hidden_dim = int(
        source_checkpoint[
            "hidden_dim"
        ]
    )

    total_games = 0
    total_wins = 0

    print(
        "=== PPO TRAINING ==="
    )
    print(
        f"source: {args.model}"
    )
    print(
        f"updates: {args.updates}"
    )
    print(
        "games/update: "
        f"{args.games_per_update}"
    )
    print(
        f"learning rate: {args.lr}"
    )
    print(
        "value head reset: yes"
    )
    print(
        f"reward mode: "
        f"{args.reward_mode}"
    )
    print(
        f"potential alpha: "
        f"{args.potential_alpha}"
    )
    print(
        f"BC KL coefficient: "
        f"{args.bc_kl_coefficient}"
    )
    print(
        f"gamma: {args.gamma}"
    )
    print(
        f"GAE lambda: "
        f"{args.gae_lambda}"
    )
    print()

    for update_index in range(
        1,
        args.updates + 1,
    ):
        episodes = []

        update_wins = 0
        update_vps = []
        update_rewards = []
        update_shaping_returns = []
        update_abs_rewards = []
        update_max_rewards = []
        update_abs_shaping = []
        update_max_shaping = []
        update_nonzero_shaping = []
        update_turns = []
        update_transitions = 0

        for local_game_index in range(
            args.games_per_update
        ):
            global_game_index = (
                total_games
                + local_game_index
            )

            episode = collect_episode(
                model,
                global_game_index,
                args.seed_offset,
                args.reward_mode,
                args.potential_alpha,
                args.gamma,
                args.gae_lambda,
            )

            episodes.append(
                episode
            )

            update_wins += (
                episode["won"]
            )

            update_vps.append(
                episode["vp"]
            )

            update_rewards.append(
                episode["reward"]
            )

            update_shaping_returns.append(
                episode[
                    "shaping_return"
                ]
            )

            update_abs_rewards.append(
                episode[
                    "mean_abs_reward"
                ]
            )

            update_max_rewards.append(
                episode[
                    "max_abs_reward"
                ]
            )

            update_abs_shaping.append(
                episode[
                    "mean_abs_shaping"
                ]
            )

            update_max_shaping.append(
                episode[
                    "max_abs_shaping"
                ]
            )

            update_nonzero_shaping.append(
                episode[
                    "nonzero_shaping_fraction"
                ]
            )

            update_turns.append(
                episode["turns"]
            )

            update_transitions += len(
                episode["action_ids"]
            )

        batch = concatenate_episodes(
            episodes
        )

        metrics = ppo_update(
            model,
            optimizer,
            batch["observations"],
            batch["legal_masks"],
            batch["action_ids"],
            batch["old_log_probs"],
            batch["advantages"],
            batch["returns"],
            epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            clip_epsilon=(
                args.clip_epsilon
            ),
            value_coefficient=(
                args.value_coefficient
            ),
            entropy_coefficient=(
                args.entropy_coefficient
            ),
            max_grad_norm=(
                args.max_grad_norm
            ),
            generator=(
                shuffle_generator
            ),
            reference_model=(
                reference_model
            ),
            bc_kl_coefficient=(
                args.bc_kl_coefficient
            ),
        )

        total_games += (
            args.games_per_update
        )

        total_wins += (
            update_wins
        )

        update_win_rate = (
            update_wins
            / args.games_per_update
        )

        cumulative_win_rate = (
            total_wins
            / total_games
        )

        print(
            f"update "
            f"{update_index:3d}/"
            f"{args.updates} | "
            f"games={args.games_per_update:3d} "
            f"transitions="
            f"{update_transitions:5d} | "
            f"win={update_win_rate:.3f} "
            f"cum_win="
            f"{cumulative_win_rate:.3f} | "
            f"vp="
            f"{np.mean(update_vps):.3f} "
            f"reward="
            f"{np.mean(update_rewards):.3f} "
            f"shape="
            f"{np.mean(update_shaping_returns):+.4f} "
            f"|r|="
            f"{np.mean(update_abs_rewards):.4f} "
            f"max|r|="
            f"{np.max(update_max_rewards):.3f} "
            f"|shape|="
            f"{np.mean(update_abs_shaping):.5f} "
            f"max|shape|="
            f"{np.max(update_max_shaping):.4f} "
            f"shape_nz="
            f"{np.mean(update_nonzero_shaping):.3f} "
            f"turns="
            f"{np.mean(update_turns):.1f} | "
            f"policy="
            f"{metrics.policy_loss:+.4f} "
            f"value="
            f"{metrics.value_loss:.4f} "
            f"entropy="
            f"{metrics.entropy:.4f} "
            f"kl="
            f"{metrics.approx_kl:.5f} "
            f"bc_kl="
            f"{metrics.bc_kl:.5f} "
            f"clip="
            f"{metrics.clip_fraction:.3f}",
            flush=True,
        )

        snapshot = (
            args.output.parent
            / (
                f"{args.output.stem}"
                f"_update"
                f"{update_index:03d}"
                f"{args.output.suffix}"
            )
        )

        save_checkpoint(
            snapshot,
            model,
            observation_dim=(
                observation_dim
            ),
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            update_index=(
                update_index
            ),
            total_games=total_games,
            total_wins=total_wins,
        )

    save_checkpoint(
        args.output,
        model,
        observation_dim=(
            observation_dim
        ),
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        update_index=args.updates,
        total_games=total_games,
        total_wins=total_wins,
    )

    print()
    print(
        "=== PPO TRAINING COMPLETE ==="
    )
    print(
        f"games: {total_games}"
    )
    print(
        f"wins: {total_wins}"
    )
    print(
        "rollout win rate: "
        f"{total_wins / total_games:.3f}"
    )
    print(
        f"saved: {args.output}"
    )


if __name__ == "__main__":
    main()
