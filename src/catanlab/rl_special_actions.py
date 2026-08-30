from __future__ import annotations

from dataclasses import dataclass

from catanlab.resources import Resource


MONOPOLY_RESOURCE_VOCABULARY = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


@dataclass(frozen=True)
class CategoricalDecisionInput:
    """
    Fixed-order categorical decision presented to a
    phase-specific learned-policy head.

    vocabulary[i] is the simulator-facing value represented
    by categorical action ID i.

    legal_mask[i] states whether that choice is currently
    legal.
    """

    vocabulary: tuple[object, ...]
    legal_mask: tuple[bool, ...]

    def __post_init__(self):
        if len(self.vocabulary) != len(
            self.legal_mask
        ):
            raise ValueError(
                "Categorical vocabulary and legal mask "
                "must have matching lengths."
            )

    @property
    def action_dim(self) -> int:
        return len(self.vocabulary)

    @property
    def legal_action_ids(
        self,
    ) -> tuple[int, ...]:
        return tuple(
            action_id
            for action_id, legal
            in enumerate(self.legal_mask)
            if legal
        )

    def encode(self, value) -> int:
        """
        Return the categorical ID for one vocabulary value.
        """
        try:
            return self.vocabulary.index(
                value
            )
        except ValueError as exc:
            raise ValueError(
                "Value is not represented in categorical "
                f"vocabulary: {value!r}"
            ) from exc

    def decode(self, action_id: int):
        """
        Return the simulator-facing value represented by ID.
        """
        if (
            action_id < 0
            or action_id >= len(self.vocabulary)
        ):
            raise ValueError(
                f"Invalid categorical action ID: "
                f"{action_id}"
            )

        return self.vocabulary[action_id]

    def is_legal_value(self, value) -> bool:
        action_id = self.encode(value)
        return self.legal_mask[action_id]


def robber_tile_decision_input(
    board,
) -> CategoricalDecisionInput:
    """
    Build the categorical robber-destination decision.

    Every actual board tile receives one category. The
    robber's current tile is the only masked destination.
    """
    vocabulary = tuple(
        tile.id
        for tile in board.tiles
    )

    legal_mask = tuple(
        tile_id != board.robber_tile_id
        for tile_id in vocabulary
    )

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=legal_mask,
    )


def robber_victim_decision_input(
    board,
    players,
    inventories,
    player,
) -> CategoricalDecisionInput:
    """
    Build the categorical robber-victim decision.

    Player IDs form the stable vocabulary. Only adjacent
    opponents with at least one public resource card are
    legal victims.
    """
    from catanlab.devcards import (
        players_adjacent_to_tile,
    )

    vocabulary = tuple(
        candidate.player_id
        for candidate in players
    )

    if board.robber_tile_id is None:
        eligible = set()
    else:
        eligible = set(
            players_adjacent_to_tile(
                board,
                players,
                board.robber_tile_id,
                exclude_player_id=(
                    player.player_id
                ),
            )
        )

        eligible = {
            victim_id
            for victim_id in eligible
            if inventories[
                victim_id
            ].total() > 0
        }

    legal_mask = tuple(
        player_id in eligible
        for player_id in vocabulary
    )

    return CategoricalDecisionInput(
        vocabulary=vocabulary,
        legal_mask=legal_mask,
    )


def monopoly_resource_decision_input(
) -> CategoricalDecisionInput:
    """
    Build the fixed five-way Monopoly resource decision.

    All producing resources are legal Monopoly targets.
    """
    return CategoricalDecisionInput(
        vocabulary=(
            MONOPOLY_RESOURCE_VOCABULARY
        ),
        legal_mask=tuple(
            True
            for _ in (
                MONOPOLY_RESOURCE_VOCABULARY
            )
        ),
    )


CATEGORICAL_TEACHER_DECISION_KINDS = (
    "robber_tile",
    "robber_victim",
    "monopoly_resource",
)
