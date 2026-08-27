from __future__ import annotations

from dataclasses import dataclass

from catanlab.search import (
    SearchState,
    apply_search_action,
    apply_search_dev_card_outcome,
    build_dev_card_belief,
    enumerate_search_actions,
    evaluate_search_state,
)
from catanlab.turns import (
    ActionType,
    AdaptiveStrategyAgent,
    TurnAction,
)


@dataclass(frozen=True)
class SearchActionEvaluation:
    """
    Evaluation of one candidate root action.
    """

    action: TurnAction
    value: float
    continuation: tuple[
        TurnAction,
        ...
    ] = ()

    @property
    def line(
        self,
    ) -> tuple[TurnAction, ...]:
        return (
            self.action,
            *self.continuation,
        )


@dataclass(frozen=True)
class SearchDecision:
    """
    Complete deterministic / expectimax search result.
    """

    action: TurnAction
    value: float
    candidates: tuple[
        SearchActionEvaluation,
        ...
    ]
    principal_variation: tuple[
        TurnAction,
        ...
    ] = ()


class OneStepLookaheadAgent(
    AdaptiveStrategyAgent
):
    """
    Depth-n same-turn expectimax search agent.

    Ordinary actions are deterministic.

    Development-card purchases are chance nodes whose
    probabilities come from legally observable card
    information rather than the hidden deck order.

    The historical class name remains for backwards
    compatibility.
    """

    def __init__(
        self,
        strategy,
        search_depth: int = 1,
    ):
        super().__init__(strategy)

        if search_depth < 1:
            raise ValueError(
                "search_depth must be at least 1"
            )

        self.search_depth = search_depth

    @staticmethod
    def _empty_inventory():
        from catanlab.economy import (
            PlayerInventory,
        )

        return PlayerInventory()

    def _make_search_state(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck,
        bank,
    ) -> SearchState:
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

    def _evaluate_action(
        self,
        state: SearchState,
        player_id: int,
        action: TurnAction,
        depth: int,
    ) -> tuple[
        float,
        tuple[TurnAction, ...],
    ]:
        """
        Evaluate one action.

        BUY_DEV_CARD is an expectimax chance node.
        Every other ordinary action is deterministic.
        """
        if (
            action.action_type
            == ActionType.BUY_DEV_CARD
        ):
            belief = build_dev_card_belief(
                state.players,
                player_id,
            )

            if belief.total <= 0:
                raise ValueError(
                    "No development-card identities "
                    "remain in the belief state."
                )

            expected_value = 0.0

            for card, count in (
                belief.counts.items()
            ):
                if count <= 0:
                    continue

                probability = (
                    count / belief.total
                )

                outcome_state = (
                    apply_search_dev_card_outcome(
                        state,
                        player_id,
                        card,
                    )
                )

                if depth <= 1:
                    outcome_value = (
                        evaluate_search_state(
                            outcome_state,
                            player_id,
                        )
                    )
                else:
                    (
                        outcome_value,
                        _,
                    ) = self._search_line(
                        outcome_state,
                        player_id,
                        depth - 1,
                    )

                expected_value += (
                    probability
                    * outcome_value
                )

            # A chance node has multiple possible
            # continuations, so there is no single
            # deterministic principal variation after
            # the purchase to report.
            return (
                expected_value,
                (),
            )

        next_state = apply_search_action(
            state,
            player_id,
            action,
        )

        if (
            action.action_type
            == ActionType.PASS
            or depth <= 1
        ):
            return (
                evaluate_search_state(
                    next_state,
                    player_id,
                ),
                (),
            )

        return self._search_line(
            next_state,
            player_id,
            depth - 1,
        )

    def _search_line(
        self,
        state: SearchState,
        player_id: int,
        depth: int,
    ) -> tuple[
        float,
        tuple[TurnAction, ...],
    ]:
        if depth <= 0:
            return (
                evaluate_search_state(
                    state,
                    player_id,
                ),
                (),
            )

        actions = enumerate_search_actions(
            state,
            player_id,
        )

        candidates = []

        for action in actions:
            (
                value,
                continuation,
            ) = self._evaluate_action(
                state,
                player_id,
                action,
                depth,
            )

            candidates.append(
                (
                    value,
                    action.action_type.value,
                    repr(action),
                    action,
                    continuation,
                )
            )

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            )
        )

        (
            value,
            _,
            _,
            action,
            continuation,
        ) = candidates[0]

        return (
            value,
            (
                action,
                *continuation,
            ),
        )

    def evaluate_actions(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck,
        bank,
    ) -> SearchDecision:
        state = self._make_search_state(
            board,
            players,
            player,
            inventory,
            dev_deck,
            bank,
        )

        actions = enumerate_search_actions(
            state,
            player_id=player.player_id,
        )

        candidates = []

        for action in actions:
            (
                value,
                continuation,
            ) = self._evaluate_action(
                state,
                player.player_id,
                action,
                self.search_depth,
            )

            candidates.append(
                SearchActionEvaluation(
                    action=action,
                    value=value,
                    continuation=continuation,
                )
            )

        candidates.sort(
            key=lambda candidate: (
                -candidate.value,
                candidate.action.action_type.value,
                repr(candidate.action),
            )
        )

        best = candidates[0]

        return SearchDecision(
            action=best.action,
            value=best.value,
            candidates=tuple(candidates),
            principal_variation=best.line,
        )

    def choose_action(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck=None,
        bank=None,
    ):
        if dev_deck is None or bank is None:
            return super().choose_action(
                board,
                players,
                player,
                inventory,
                dev_deck=dev_deck,
                bank=bank,
            )

        return self.evaluate_actions(
            board,
            players,
            player,
            inventory,
            dev_deck,
            bank,
        ).action
