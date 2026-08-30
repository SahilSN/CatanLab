from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from catanlab.resources import Resource


PRODUCING_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


class RLActionType(str, Enum):
    PASS = "pass"
    BUILD_SETTLEMENT = "build_settlement"
    BUILD_CITY = "build_city"
    BUILD_ROAD = "build_road"
    BUY_DEV_CARD = "buy_dev_card"
    MARITIME_TRADE = "maritime_trade"


@dataclass(frozen=True)
class RLAction:
    action_type: RLActionType

    vertex_id: int | None = None
    edge: tuple[int, int] | None = None

    give_resource: Resource | None = None
    receive_resource: Resource | None = None


def build_action_vocabulary(
    num_vertices: int,
    edges,
) -> tuple[RLAction, ...]:
    """
    Build a deterministic fixed action vocabulary.

    The vocabulary shape depends only on board topology,
    not on the current player's legal actions.
    """
    actions = [
        RLAction(
            action_type=RLActionType.PASS
        )
    ]

    for vertex_id in range(num_vertices):
        actions.append(
            RLAction(
                action_type=(
                    RLActionType.BUILD_SETTLEMENT
                ),
                vertex_id=vertex_id,
            )
        )

    for vertex_id in range(num_vertices):
        actions.append(
            RLAction(
                action_type=RLActionType.BUILD_CITY,
                vertex_id=vertex_id,
            )
        )

    for edge in edges:
        canonical = tuple(
            sorted(
                (
                    edge.vertex_a,
                    edge.vertex_b,
                )
            )
        )

        actions.append(
            RLAction(
                action_type=RLActionType.BUILD_ROAD,
                edge=canonical,
            )
        )

    actions.append(
        RLAction(
            action_type=RLActionType.BUY_DEV_CARD
        )
    )

    for give_resource in PRODUCING_RESOURCES:
        for receive_resource in PRODUCING_RESOURCES:
            if give_resource == receive_resource:
                continue

            actions.append(
                RLAction(
                    action_type=(
                        RLActionType.MARITIME_TRADE
                    ),
                    give_resource=give_resource,
                    receive_resource=receive_resource,
                )
            )

    return tuple(actions)


def to_turn_action(
    action: RLAction,
):
    """
    Convert one RL-facing action into the simulator's
    existing TurnAction representation.
    """
    from catanlab.turns import (
        ActionType,
        TurnAction,
    )

    if action.action_type == RLActionType.PASS:
        return TurnAction(
            action_type=ActionType.PASS
        )

    if (
        action.action_type
        == RLActionType.BUILD_SETTLEMENT
    ):
        return TurnAction(
            action_type=(
                ActionType.BUILD_SETTLEMENT
            ),
            vertex_id=action.vertex_id,
        )

    if (
        action.action_type
        == RLActionType.BUILD_CITY
    ):
        return TurnAction(
            action_type=ActionType.BUILD_CITY,
            vertex_id=action.vertex_id,
        )

    if (
        action.action_type
        == RLActionType.BUILD_ROAD
    ):
        return TurnAction(
            action_type=ActionType.BUILD_ROAD,
            edge=action.edge,
        )

    if (
        action.action_type
        == RLActionType.BUY_DEV_CARD
    ):
        return TurnAction(
            action_type=ActionType.BUY_DEV_CARD
        )

    if (
        action.action_type
        == RLActionType.MARITIME_TRADE
    ):
        return TurnAction(
            action_type=ActionType.MARITIME_TRADE,
            give_resource=action.give_resource,
            receive_resource=(
                action.receive_resource
            ),
        )

    raise ValueError(
        f"Unsupported RL action: {action}"
    )


def legal_action_mask(
    state,
    player_id: int,
    vocabulary: tuple[RLAction, ...],
) -> tuple[bool, ...]:
    """
    Return one boolean per fixed RL action.

    Legality is derived from the simulator/search
    action enumerator rather than reimplementing Catan
    rules here.
    """
    from catanlab.search import (
        enumerate_search_actions,
    )

    legal_turn_actions = (
        enumerate_search_actions(
            state,
            player_id,
            include_maritime_trades=True,
        )
    )

    legal = set(
        repr(action)
        for action in legal_turn_actions
    )

    mask = []

    for rl_action in vocabulary:
        turn_action = to_turn_action(
            rl_action
        )

        mask.append(
            repr(turn_action) in legal
        )

    return tuple(mask)


def turn_action_id(
    turn_action,
    vocabulary: tuple[RLAction, ...],
) -> int:
    """
    Return the fixed RL action ID corresponding to an
    ordinary simulator TurnAction.

    Raises ValueError if the action is outside the
    current ordinary RL action vocabulary.
    """
    for action_id, rl_action in enumerate(
        vocabulary
    ):
        if (
            to_turn_action(rl_action)
            == turn_action
        ):
            return action_id

    raise ValueError(
        "TurnAction is not represented in the "
        f"ordinary RL vocabulary: {turn_action}"
    )
