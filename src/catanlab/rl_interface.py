from __future__ import annotations

from dataclasses import dataclass

from catanlab.action_space import (
    RLAction,
    build_action_vocabulary,
    legal_action_mask,
)
from catanlab.observation import (
    game_observation,
)
from catanlab.observation_encoder import (
    encode_game_observation,
)
from catanlab.search import SearchState


@dataclass(frozen=True)
class RLPolicyInput:
    """
    Complete information-safe input for an ordinary
    learned-policy decision.
    """

    observation: tuple[float, ...]
    legal_mask: tuple[bool, ...]

    @property
    def observation_dim(self) -> int:
        return len(self.observation)

    @property
    def action_dim(self) -> int:
        return len(self.legal_mask)

    @property
    def legal_action_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            action_id
            for action_id, is_legal
            in enumerate(self.legal_mask)
            if is_legal
        )


@dataclass(frozen=True)
class RLDecisionContext:
    """
    Policy input together with the fixed action
    vocabulary needed to decode a selected action ID.
    """

    policy_input: RLPolicyInput

    vocabulary: tuple[
        RLAction,
        ...
    ]


def build_rl_decision_context(
    board,
    players,
    inventories,
    player_id: int,
    bank,
    dev_deck,
) -> RLDecisionContext:
    """
    Construct the sanitized numeric observation and
    authoritative ordinary-action legal mask for one
    player.
    """
    observation = game_observation(
        board,
        players,
        inventories,
        player_id,
        bank,
        dev_deck,
    )

    encoded = encode_game_observation(
        observation
    )

    vocabulary = build_action_vocabulary(
        len(board.vertices),
        board.edges,
    )

    # The legal mask only needs the active player's
    # private inventory. Opponent resource identities
    # remain unavailable.
    from catanlab.economy import (
        PlayerInventory,
    )

    masked_inventories = [
        PlayerInventory()
        for _ in players
    ]

    masked_inventories[
        player_id
    ] = inventories[
        player_id
    ]

    mask_state = SearchState(
        board=board,
        players=players,
        inventories=masked_inventories,
        dev_deck=dev_deck,
        bank=bank,
    )

    mask = legal_action_mask(
        mask_state,
        player_id,
        vocabulary,
    )

    return RLDecisionContext(
        policy_input=RLPolicyInput(
            observation=encoded,
            legal_mask=mask,
        ),
        vocabulary=vocabulary,
    )
