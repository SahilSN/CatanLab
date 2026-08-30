from __future__ import annotations

from pathlib import Path

import torch

from catanlab.game import run_game
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


MODEL_PATH = Path(
    "results/rl_baselines/"
    "bc_dagger_v1.pt"
)


def load_model():
    checkpoint = torch.load(
        MODEL_PATH,
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

    return model


def main():
    model = load_model()

    target_seat = 0

    target_strategy = (
        StrategyType.FIVE_RESOURCE
    )

    ppo_agent = PPORolloutAgent(
        target_strategy,
        model=model,
        deterministic=False,
        seed=123,
    )

    agents = [
        ppo_agent,
        AdaptiveStrategyAgent(
            StrategyType.HYBRID_OWS
        ),
        AdaptiveStrategyAgent(
            StrategyType.FULL_OWS
        ),
        AdaptiveStrategyAgent(
            StrategyType.PORT
        ),
    ]

    strategies = [
        StrategyType.FIVE_RESOURCE,
        StrategyType.HYBRID_OWS,
        StrategyType.FULL_OWS,
        StrategyType.PORT,
    ]

    result = run_game(
        strategies=strategies,
        seed=123,
        max_turns=2000,
        validate_conservation=True,
        turn_agents=agents,
    )

    won = (
        result.winner_id
        == target_seat
    )

    terminal_reward = (
        1.0
        if won
        else 0.0
    )

    ppo_agent.rollout.finish_episode(
        terminal_reward
    )

    tensors = (
        ppo_agent.rollout
        .to_tensors()
    )

    print(
        "=== PPO ROLLOUT SMOKE ==="
    )
    print(
        "winner:",
        result.winner_id,
    )
    print(
        "target VP:",
        result.players[
            target_seat
        ].victory_points,
    )
    print(
        "transitions:",
        len(
            ppo_agent.rollout
        ),
    )
    print(
        "observation shape:",
        tuple(
            tensors[
                "observations"
            ].shape
        ),
    )
    print(
        "mask shape:",
        tuple(
            tensors[
                "legal_masks"
            ].shape
        ),
    )
    print(
        "terminal reward:",
        float(
            tensors[
                "rewards"
            ][-1].item()
        ),
    )
    print(
        "done count:",
        int(
            tensors[
                "dones"
            ].sum().item()
        ),
    )


if __name__ == "__main__":
    main()
