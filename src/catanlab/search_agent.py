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
        use_transposition_cache: bool = True,
        search_maritime_trades: bool = False,
    ):
        super().__init__(strategy)

        if search_depth < 1:
            raise ValueError(
                "search_depth must be at least 1"
            )

        self.search_depth = search_depth
        self.use_transposition_cache = (
            use_transposition_cache
        )
        self.search_maritime_trades = (
            search_maritime_trades
        )

        self._evaluation_cache = {}
        self._search_cache = {}

        self.cache_hits = 0
        self.cache_misses = 0

    @staticmethod
    def _state_key(
        state: SearchState,
        player_id: int,
    ) -> tuple:
        """
        Return an information-safe immutable key for
        search-equivalent states.

        Hidden development-card identities belonging
        to opponents and the hidden deck order are
        deliberately excluded.
        """
        from catanlab.economy import (
            PRODUCING_RESOURCES,
        )

        player_keys = []

        for index, player in enumerate(
            state.players
        ):
            if index == player_id:
                dev_cards = tuple(
                    sorted(player.dev_cards)
                )
                new_dev_cards = tuple(
                    sorted(player.new_dev_cards)
                )
            else:
                # Opponent identities are private.
                # Only the publicly observable count
                # may influence the cache key.
                dev_cards = (
                    len(player.dev_cards),
                )
                new_dev_cards = ()

            roads = tuple(
                sorted(
                    tuple(sorted(edge))
                    for edge in player.roads
                )
            )

            player_keys.append(
                (
                    player.player_id,
                    tuple(
                        sorted(
                            player.settlements
                        )
                    ),
                    tuple(
                        sorted(player.cities)
                    ),
                    roads,
                    dev_cards,
                    new_dev_cards,
                    tuple(
                        sorted(
                            player.played_dev_cards
                        )
                    ),
                    player.knights_played,
                    player.has_largest_army,
                    player.has_longest_road,
                )
            )

        inventory_keys = tuple(
            tuple(
                inventory.count(resource)
                for resource
                in PRODUCING_RESOURCES
            )
            for inventory in state.inventories
        )

        bank_key = tuple(
            state.bank.count(resource)
            for resource in PRODUCING_RESOURCES
        )

        return (
            tuple(player_keys),
            inventory_keys,
            bank_key,

            # Deck size is public/useful. Its hidden
            # identities and ordering are not.
            len(state.dev_deck.cards),
        )

    def _evaluate_state(
        self,
        state: SearchState,
        player_id: int,
    ) -> float:
        """
        Evaluate a leaf, reusing an equivalent leaf
        value when caching is enabled.
        """
        if not self.use_transposition_cache:
            return evaluate_search_state(
                state,
                player_id,
            )

        key = self._state_key(
            state,
            player_id,
        )

        if key in self._evaluation_cache:
            self.cache_hits += 1
            return self._evaluation_cache[key]

        self.cache_misses += 1

        value = evaluate_search_state(
            state,
            player_id,
        )

        self._evaluation_cache[key] = value

        return value

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
                        self._evaluate_state(
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
                self._evaluate_state(
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
                self._evaluate_state(
                    state,
                    player_id,
                ),
                (),
            )

        cache_key = None

        if self.use_transposition_cache:
            cache_key = (
                depth,
                self._state_key(
                    state,
                    player_id,
                ),
            )

            if cache_key in self._search_cache:
                self.cache_hits += 1
                return self._search_cache[
                    cache_key
                ]

            self.cache_misses += 1

        actions = enumerate_search_actions(
            state,
            player_id,
            include_maritime_trades=(
                self.search_maritime_trades
            ),
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

        result = (
            value,
            (
                action,
                *continuation,
            ),
        )

        if (
            self.use_transposition_cache
            and cache_key is not None
        ):
            self._search_cache[
                cache_key
            ] = result

        return result

    def evaluate_actions(
        self,
        board,
        players,
        player,
        inventory,
        dev_deck,
        bank,
    ) -> SearchDecision:
        # Search caches are deliberately scoped to one
        # decision. Live game state may change between
        # calls, so nothing persists across turns.
        self._evaluation_cache = {}
        self._search_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

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
            include_maritime_trades=(
                self.search_maritime_trades
            ),
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
