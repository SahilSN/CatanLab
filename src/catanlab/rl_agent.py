from __future__ import annotations

import random

from catanlab.action_space import (
    build_action_vocabulary,
    legal_action_mask,
    to_turn_action,
)
from catanlab.devcards import DevCardDeck
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.search import SearchState
from catanlab.turns import (
    AdaptiveStrategyAgent,
    TurnAction,
)


class RandomMaskedAgent(
    AdaptiveStrategyAgent
):
    """
    RL-interface smoke-test agent.

    Ordinary turn actions are selected uniformly from
    the fixed RL action vocabulary after applying the
    authoritative legal-action mask.

    All other decisions currently retain the inherited
    AdaptiveStrategyAgent behavior.
    """

    def __init__(
        self,
        strategy,
        seed: int | None = None,
    ):
        super().__init__(strategy)

        self.rng = random.Random(seed)

        self.actions_selected = 0

        self.last_action_id: int | None = None
        self.last_legal_count: int | None = None

    @staticmethod
    def _empty_inventory():
        return PlayerInventory()

    def _make_mask_state(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck,
        bank,
    ) -> SearchState:
        """
        Construct the minimal information-safe state
        required by the ordinary-action legal mask.

        Opponent resource identities are deliberately
        unavailable.
        """
        inventories = [
            self._empty_inventory()
            for _ in players
        ]

        inventories[
            player.player_id
        ] = inventory

        return SearchState(
            board=board,
            players=players,
            inventories=inventories,
            dev_deck=dev_deck,
            bank=bank,
        )

    def choose_action(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck: DevCardDeck | None = None,
        bank: ResourceBank | None = None,
    ) -> TurnAction:
        if dev_deck is None:
            raise ValueError(
                "RandomMaskedAgent requires dev_deck."
            )

        if bank is None:
            raise ValueError(
                "RandomMaskedAgent requires bank."
            )

        vocabulary = build_action_vocabulary(
            len(board.vertices),
            board.edges,
        )

        state = self._make_mask_state(
            board,
            players,
            player,
            inventory,
            dev_deck,
            bank,
        )

        mask = legal_action_mask(
            state,
            player.player_id,
            vocabulary,
        )

        legal_ids = [
            action_id
            for action_id, is_legal
            in enumerate(mask)
            if is_legal
        ]

        if not legal_ids:
            raise RuntimeError(
                "Legal-action mask contains no "
                "legal actions."
            )

        action_id = self.rng.choice(
            legal_ids
        )

        # Defensive assertion: the policy must never
        # return an action masked as illegal.
        if not mask[action_id]:
            raise RuntimeError(
                "RandomMaskedAgent selected an "
                "illegal masked action."
            )

        self.actions_selected += 1
        self.last_action_id = action_id
        self.last_legal_count = len(
            legal_ids
        )

        return to_turn_action(
            vocabulary[action_id]
        )


class NeuralPolicyAgent(
    AdaptiveStrategyAgent
):
    """
    Ordinary-action agent driven by a PyTorch policy
    network.

    Special turn decisions continue to use inherited
    AdaptiveStrategyAgent behavior for now.
    """

    def __init__(
        self,
        strategy,
        model,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        super().__init__(strategy)

        import torch

        self.model = model
        self.deterministic = deterministic

        self.generator = torch.Generator()

        if seed is not None:
            self.generator.manual_seed(
                seed
            )

        self.actions_selected = 0
        self.last_action_id = None
        self.last_value = None
        self.last_legal_count = None

    def choose_action(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck=None,
        bank=None,
        inventories=None,
    ):
        import torch

        from catanlab.action_space import (
            to_turn_action,
        )
        from catanlab.rl_interface import (
            build_rl_decision_context,
        )
        from catanlab.rl_model import (
            mask_policy_logits,
        )

        if dev_deck is None:
            raise ValueError(
                "NeuralPolicyAgent requires dev_deck."
            )

        if bank is None:
            raise ValueError(
                "NeuralPolicyAgent requires bank."
            )

        if inventories is None:
            raise ValueError(
                "NeuralPolicyAgent requires inventories."
            )

        context = build_rl_decision_context(
            board,
            players,
            inventories,
            player.player_id,
            bank,
            dev_deck,
        )

        policy_input = (
            context.policy_input
        )

        observation = torch.tensor(
            policy_input.observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        legal_mask = torch.tensor(
            policy_input.legal_mask,
            dtype=torch.bool,
        ).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            logits, value = self.model(
                observation
            )

            masked_logits = (
                mask_policy_logits(
                    logits,
                    legal_mask,
                )
            )

            if self.deterministic:
                action_id = int(
                    torch.argmax(
                        masked_logits,
                        dim=-1,
                    ).item()
                )
            else:
                probabilities = torch.softmax(
                    masked_logits,
                    dim=-1,
                )

                action_id = int(
                    torch.multinomial(
                        probabilities,
                        num_samples=1,
                        generator=self.generator,
                    ).item()
                )

        if not policy_input.legal_mask[
            action_id
        ]:
            raise RuntimeError(
                "Neural policy selected a masked "
                "illegal action."
            )

        self.actions_selected += 1
        self.last_action_id = action_id
        self.last_value = float(
            value.item()
        )
        self.last_legal_count = len(
            policy_input.legal_action_ids
        )

        return to_turn_action(
            context.vocabulary[
                action_id
            ]
        )




class PPORolloutAgent(
    NeuralPolicyAgent
):
    """
    Neural policy that records on-policy ordinary-action
    transitions for later PPO optimization.

    Special turn decisions still use inherited
    AdaptiveStrategyAgent behavior.
    """

    def __init__(
        self,
        strategy,
        model,
        deterministic: bool = False,
        seed: int | None = None,
    ):
        super().__init__(
            strategy,
            model=model,
            deterministic=deterministic,
            seed=seed,
        )

        from catanlab.ppo import (
            PPORolloutBuffer,
        )

        self.rollout = (
            PPORolloutBuffer()
        )

    def choose_action(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck=None,
        bank=None,
        inventories=None,
    ):
        import torch

        from catanlab.action_space import (
            to_turn_action,
        )
        from catanlab.ppo import (
            PPOTransition,
            public_vp_margin_potential,
        )
        from catanlab.rl_interface import (
            build_rl_decision_context,
        )
        from catanlab.rl_model import (
            mask_policy_logits,
        )

        if dev_deck is None:
            raise ValueError(
                "PPORolloutAgent requires dev_deck."
            )

        if bank is None:
            raise ValueError(
                "PPORolloutAgent requires bank."
            )

        if inventories is None:
            raise ValueError(
                "PPORolloutAgent requires inventories."
            )

        potential = (
            public_vp_margin_potential(
                players,
                player.player_id,
            )
        )

        context = build_rl_decision_context(
            board,
            players,
            inventories,
            player.player_id,
            bank,
            dev_deck,
        )

        policy_input = (
            context.policy_input
        )

        observation = torch.tensor(
            policy_input.observation,
            dtype=torch.float32,
        ).unsqueeze(0)

        legal_mask = torch.tensor(
            policy_input.legal_mask,
            dtype=torch.bool,
        ).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            logits, value = self.model(
                observation
            )

            masked_logits = (
                mask_policy_logits(
                    logits,
                    legal_mask,
                )
            )

            distribution = (
                torch.distributions.Categorical(
                    logits=masked_logits
                )
            )

            if self.deterministic:
                action_tensor = torch.argmax(
                    masked_logits,
                    dim=-1,
                )
            else:
                probabilities = torch.softmax(
                    masked_logits,
                    dim=-1,
                )

                action_tensor = torch.multinomial(
                    probabilities,
                    num_samples=1,
                    generator=self.generator,
                ).squeeze(-1)

            log_prob = (
                distribution.log_prob(
                    action_tensor
                )
            )

        action_id = int(
            action_tensor.item()
        )

        if not policy_input.legal_mask[
            action_id
        ]:
            raise RuntimeError(
                "PPO policy selected a masked "
                "illegal action."
            )

        self.rollout.append(
            PPOTransition(
                observation=(
                    policy_input.observation
                ),
                legal_mask=(
                    policy_input.legal_mask
                ),
                action_id=action_id,
                log_prob=float(
                    log_prob.item()
                ),
                value=float(
                    value.item()
                ),
                potential=potential,
            )
        )

        self.actions_selected += 1
        self.last_action_id = action_id
        self.last_value = float(
            value.item()
        )
        self.last_legal_count = len(
            policy_input.legal_action_ids
        )

        return to_turn_action(
            context.vocabulary[
                action_id
            ]
        )
