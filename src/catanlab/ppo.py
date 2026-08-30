from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PPOTransition:
    observation: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    action_id: int
    log_prob: float
    value: float

    potential: float = 0.0
    reward: float = 0.0
    done: bool = False


class PPORolloutBuffer:
    def __init__(self):
        self.transitions: list[
            PPOTransition
        ] = []

    def __len__(self) -> int:
        return len(
            self.transitions
        )

    def append(
        self,
        transition: PPOTransition,
    ) -> None:
        self.transitions.append(
            transition
        )

    def clear(self) -> None:
        self.transitions.clear()

    def finish_episode(
        self,
        terminal_reward: float,
    ) -> None:
        if not self.transitions:
            return

        last_index = (
            len(self.transitions) - 1
        )

        updated = []

        for index, transition in enumerate(
            self.transitions
        ):
            reward = (
                terminal_reward
                if index == last_index
                else 0.0
            )

            done = (
                index == last_index
            )

            updated.append(
                PPOTransition(
                    observation=(
                        transition.observation
                    ),
                    legal_mask=(
                        transition.legal_mask
                    ),
                    action_id=(
                        transition.action_id
                    ),
                    log_prob=(
                        transition.log_prob
                    ),
                    value=(
                        transition.value
                    ),
                    potential=(
                        transition.potential
                    ),
                    reward=reward,
                    done=done,
                )
            )

        self.transitions = updated

    def finish_episode_with_potential(
        self,
        terminal_reward: float,
        terminal_potential: float,
        shaping_alpha: float,
        gamma: float = 1.0,
    ) -> None:
        """
        Finalize one complete trajectory using

            r' = r + alpha * (
                gamma * Phi(s_next) - Phi(s)
            )

        The terminal environment reward is applied
        only to the final transition.
        """
        if not self.transitions:
            return

        if shaping_alpha < 0.0:
            raise ValueError(
                "shaping_alpha must be nonnegative."
            )

        updated = []

        for index, transition in enumerate(
            self.transitions
        ):
            is_last = (
                index
                == len(self.transitions) - 1
            )

            if is_last:
                next_potential = (
                    terminal_potential
                )
            else:
                next_potential = (
                    self.transitions[
                        index + 1
                    ].potential
                )

            shaping_reward = (
                shaping_alpha
                * (
                    gamma
                    * next_potential
                    - transition.potential
                )
            )

            environment_reward = (
                terminal_reward
                if is_last
                else 0.0
            )

            updated.append(
                PPOTransition(
                    observation=(
                        transition.observation
                    ),
                    legal_mask=(
                        transition.legal_mask
                    ),
                    action_id=(
                        transition.action_id
                    ),
                    log_prob=(
                        transition.log_prob
                    ),
                    value=(
                        transition.value
                    ),
                    potential=(
                        transition.potential
                    ),
                    reward=(
                        environment_reward
                        + shaping_reward
                    ),
                    done=is_last,
                )
            )

        self.transitions = updated

    def to_tensors(
        self,
        device: torch.device | str = "cpu",
    ):
        if not self.transitions:
            raise ValueError(
                "Cannot tensorize an empty rollout."
            )

        observations = torch.tensor(
            [
                transition.observation
                for transition
                in self.transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        legal_masks = torch.tensor(
            [
                transition.legal_mask
                for transition
                in self.transitions
            ],
            dtype=torch.bool,
            device=device,
        )

        action_ids = torch.tensor(
            [
                transition.action_id
                for transition
                in self.transitions
            ],
            dtype=torch.long,
            device=device,
        )

        old_log_probs = torch.tensor(
            [
                transition.log_prob
                for transition
                in self.transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        old_values = torch.tensor(
            [
                transition.value
                for transition
                in self.transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        potentials = torch.tensor(
            [
                transition.potential
                for transition
                in self.transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        rewards = torch.tensor(
            [
                transition.reward
                for transition
                in self.transitions
            ],
            dtype=torch.float32,
            device=device,
        )

        dones = torch.tensor(
            [
                transition.done
                for transition
                in self.transitions
            ],
            dtype=torch.bool,
            device=device,
        )

        return {
            "observations": observations,
            "legal_masks": legal_masks,
            "action_ids": action_ids,
            "old_log_probs": old_log_probs,
            "old_values": old_values,
            "potentials": potentials,
            "rewards": rewards,
            "dones": dones,
        }


@dataclass(frozen=True)
class PPOUpdateMetrics:
    total_loss: float
    policy_loss: float
    value_loss: float
    entropy: float
    approx_kl: float
    bc_kl: float
    clip_fraction: float


def compute_gae(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    values: torch.Tensor,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    """
    Compute generalized advantage estimates and
    value-function targets for one completed episode.
    """
    if rewards.ndim != 1:
        raise ValueError(
            "rewards must be one-dimensional."
        )

    if (
        rewards.shape != dones.shape
        or rewards.shape != values.shape
    ):
        raise ValueError(
            "rewards, dones, and values must "
            "have identical shapes."
        )

    advantages = torch.zeros_like(
        rewards
    )

    next_value = torch.tensor(
        0.0,
        dtype=values.dtype,
        device=values.device,
    )

    next_advantage = torch.tensor(
        0.0,
        dtype=values.dtype,
        device=values.device,
    )

    for index in reversed(
        range(len(rewards))
    ):
        nonterminal = (
            1.0
            - dones[index].float()
        )

        delta = (
            rewards[index]
            + gamma
            * next_value
            * nonterminal
            - values[index]
        )

        advantages[index] = (
            delta
            + gamma
            * gae_lambda
            * nonterminal
            * next_advantage
        )

        next_value = values[index]
        next_advantage = (
            advantages[index]
        )

    returns = (
        advantages + values
    )

    return (
        advantages,
        returns,
    )


def ppo_update(
    model,
    optimizer,
    observations: torch.Tensor,
    legal_masks: torch.Tensor,
    action_ids: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    *,
    epochs: int = 4,
    batch_size: int = 128,
    clip_epsilon: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    max_grad_norm: float = 0.5,
    generator: torch.Generator | None = None,
    reference_model=None,
    bc_kl_coefficient: float = 0.0,
) -> PPOUpdateMetrics:
    """
    Run clipped PPO optimization over one collected
    on-policy batch.
    """
    from catanlab.rl_model import (
        mask_policy_logits,
    )

    count = observations.shape[0]

    if bc_kl_coefficient < 0.0:
        raise ValueError(
            "bc_kl_coefficient must be nonnegative."
        )

    if (
        bc_kl_coefficient > 0.0
        and reference_model is None
    ):
        raise ValueError(
            "Positive BC KL coefficient requires "
            "reference_model."
        )

    if reference_model is not None:
        reference_model.eval()

        for parameter in (
            reference_model.parameters()
        ):
            parameter.requires_grad_(False)

    if count == 0:
        raise ValueError(
            "Cannot update PPO on an empty batch."
        )

    if not (
        legal_masks.shape[0]
        == count
        == action_ids.shape[0]
        == old_log_probs.shape[0]
        == advantages.shape[0]
        == returns.shape[0]
    ):
        raise ValueError(
            "PPO batch tensors disagree "
            "on row count."
        )

    advantages = (
        advantages
        - advantages.mean()
    ) / (
        advantages.std(
            unbiased=False
        )
        + 1e-8
    )

    metric_total = 0.0
    metric_policy = 0.0
    metric_value = 0.0
    metric_entropy = 0.0
    metric_kl = 0.0
    metric_bc_kl = 0.0
    metric_clip = 0.0
    metric_batches = 0

    model.train()

    for _ in range(epochs):
        indices = torch.randperm(
            count,
            generator=generator,
            device="cpu",
        )

        if observations.device.type != "cpu":
            indices = indices.to(
                observations.device
            )

        for start in range(
            0,
            count,
            batch_size,
        ):
            batch_indices = indices[
                start:
                start + batch_size
            ]

            batch_observations = (
                observations[
                    batch_indices
                ]
            )

            batch_masks = legal_masks[
                batch_indices
            ]

            batch_actions = action_ids[
                batch_indices
            ]

            batch_old_log_probs = (
                old_log_probs[
                    batch_indices
                ]
            )

            batch_advantages = (
                advantages[
                    batch_indices
                ]
            )

            batch_returns = returns[
                batch_indices
            ]

            logits, values = model(
                batch_observations
            )

            masked_logits = (
                mask_policy_logits(
                    logits,
                    batch_masks,
                )
            )

            distribution = (
                torch.distributions.Categorical(
                    logits=masked_logits
                )
            )

            if reference_model is not None:
                with torch.no_grad():
                    (
                        reference_logits,
                        _,
                    ) = reference_model(
                        batch_observations
                    )

                    reference_masked_logits = (
                        mask_policy_logits(
                            reference_logits,
                            batch_masks,
                        )
                    )

                    reference_distribution = (
                        torch.distributions.Categorical(
                            logits=(
                                reference_masked_logits
                            )
                        )
                    )

                # Compute KL(current || frozen BC) safely.
                #
                # Illegal actions have probability zero under
                # both policies. Their masked logits are -inf,
                # so torch.distributions.kl_divergence can form
                # undefined expressions involving -inf - -inf.
                #
                # Clamping only affects zero-probability entries;
                # multiplying by the current probability makes
                # illegal actions contribute exactly zero.
                current_probs = (
                    distribution.probs
                )

                reference_probs = (
                    reference_distribution.probs
                )

                probability_floor = (
                    torch.finfo(
                        current_probs.dtype
                    ).tiny
                )

                current_log_probs_all = (
                    torch.log(
                        current_probs.clamp_min(
                            probability_floor
                        )
                    )
                )

                reference_log_probs_all = (
                    torch.log(
                        reference_probs.clamp_min(
                            probability_floor
                        )
                    )
                )

                per_action_bc_kl = (
                    current_probs
                    * (
                        current_log_probs_all
                        - reference_log_probs_all
                    )
                )

                # Be explicit that only legal actions are
                # part of the policy support.
                per_action_bc_kl = (
                    per_action_bc_kl.masked_fill(
                        ~batch_masks,
                        0.0,
                    )
                )

                bc_kl = (
                    per_action_bc_kl
                    .sum(dim=-1)
                    .mean()
                )
            else:
                bc_kl = torch.zeros(
                    (),
                    dtype=masked_logits.dtype,
                    device=masked_logits.device,
                )

            new_log_probs = (
                distribution.log_prob(
                    batch_actions
                )
            )

            entropy = (
                distribution.entropy().mean()
            )

            ratio = torch.exp(
                new_log_probs
                - batch_old_log_probs
            )

            unclipped = (
                ratio
                * batch_advantages
            )

            clipped = (
                torch.clamp(
                    ratio,
                    1.0 - clip_epsilon,
                    1.0 + clip_epsilon,
                )
                * batch_advantages
            )

            policy_loss = -torch.minimum(
                unclipped,
                clipped,
            ).mean()

            value_loss = (
                0.5
                * (
                    values.squeeze(-1)
                    - batch_returns
                )
                .pow(2)
                .mean()
            )

            total_loss = (
                policy_loss
                + value_coefficient
                * value_loss
                - entropy_coefficient
                * entropy
                + bc_kl_coefficient
                * bc_kl
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_grad_norm,
            )

            optimizer.step()

            with torch.no_grad():
                log_ratio = (
                    new_log_probs
                    - batch_old_log_probs
                )

                approx_kl = (
                    (
                        torch.exp(log_ratio)
                        - 1.0
                        - log_ratio
                    )
                    .mean()
                )

                clip_fraction = (
                    (
                        torch.abs(
                            ratio - 1.0
                        )
                        > clip_epsilon
                    )
                    .float()
                    .mean()
                )

            metric_total += float(
                total_loss.item()
            )

            metric_policy += float(
                policy_loss.item()
            )

            metric_value += float(
                value_loss.item()
            )

            metric_entropy += float(
                entropy.item()
            )

            metric_kl += float(
                approx_kl.item()
            )

            metric_bc_kl += float(
                bc_kl.item()
            )

            metric_clip += float(
                clip_fraction.item()
            )

            metric_batches += 1

    if metric_batches == 0:
        raise RuntimeError(
            "PPO update produced no minibatches."
        )

    denominator = float(
        metric_batches
    )

    return PPOUpdateMetrics(
        total_loss=(
            metric_total / denominator
        ),
        policy_loss=(
            metric_policy / denominator
        ),
        value_loss=(
            metric_value / denominator
        ),
        entropy=(
            metric_entropy / denominator
        ),
        approx_kl=(
            metric_kl / denominator
        ),
        bc_kl=(
            metric_bc_kl / denominator
        ),
        clip_fraction=(
            metric_clip / denominator
        ),
    )



def terminal_reward(
    *,
    won: bool,
    final_vp: float,
    reward_mode: str,
    best_opponent_vp: float | None = None,
) -> float:
    if reward_mode == "win":
        return (
            1.0
            if won
            else 0.0
        )

    if reward_mode == "vp":
        return (
            final_vp / 10.0
        )

    if reward_mode == "vp_win":
        return (
            final_vp / 10.0
            + (
                1.0
                if won
                else 0.0
            )
        )

    if reward_mode == "win_margin":
        if best_opponent_vp is None:
            raise ValueError(
                "win_margin reward requires "
                "best_opponent_vp."
            )

        margin = (
            final_vp
            - best_opponent_vp
        ) / 10.0

        shaping = (
            0.1 * margin
        )

        return (
            shaping
            + (
                1.0
                if won
                else 0.0
            )
        )

    raise ValueError(
        "Unknown reward mode: "
        f"{reward_mode!r}"
    )


def public_vp_margin_potential(
    players,
    player_id: int,
) -> float:
    """
    Information-safe progress potential.

    Uses only publicly visible victory points, so
    hidden victory-point development cards cannot
    leak into the shaping signal.
    """
    if (
        player_id < 0
        or player_id >= len(players)
    ):
        raise ValueError(
            f"Invalid player_id: {player_id}"
        )

    if len(players) < 2:
        raise ValueError(
            "VP-margin potential requires "
            "at least two players."
        )

    learner_vp = float(
        players[
            player_id
        ].public_victory_points
    )

    opponent_vps = [
        float(
            player.public_victory_points
        )
        for index, player
        in enumerate(players)
        if index != player_id
    ]

    return (
        learner_vp
        - max(opponent_vps)
    ) / 10.0
