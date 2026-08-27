from __future__ import annotations

from dataclasses import dataclass

from catanlab.search import (
    SearchState,
    apply_search_action,
    apply_search_dev_card_outcome,
    apply_search_monopoly_outcome,
    apply_search_road_building,
    apply_search_year_of_plenty,
    build_dev_card_belief,
    build_monopoly_gain_belief,
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
        search_maritime_trades: bool = True,
        search_year_of_plenty: bool = False,
        search_road_building: bool = True,
        search_monopoly: bool = False,
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
        self.search_year_of_plenty = (
            search_year_of_plenty
        )
        self.search_road_building = (
            search_road_building
        )
        self.search_monopoly = (
            search_monopoly
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

    def choose_dev_card_play(
        self,
        board,
        players,
        player,
        inventories,
        phase,
        dev_deck=None,
        bank=None,
    ):
        """
        Search selected post-roll development-card
        decisions while preserving the established
        heuristic policy for all other cards.
        """
        baseline = super().choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            phase,
        )

        if (
            not self.search_year_of_plenty
            and not self.search_road_building
            and not self.search_monopoly
        ):
            return baseline

        from catanlab.devcard_policy import (
            DevCardDecision,
            DevCardPhase,
        )
        from catanlab.devcards import (
            DevCardType,
            has_playable_dev_card,
        )
        from catanlab.resources import Resource
        from catanlab.turns import legal_road_edges

        # Pre-roll search would require a dice-roll
        # chance node. Preserve heuristic behavior.
        if phase != DevCardPhase.POST_ROLL:
            return baseline

        if dev_deck is None or bank is None:
            return baseline

        # ------------------------------------------------
        # Monopoly
        # ------------------------------------------------

        # For this first Monopoly-search version, only
        # replace an established heuristic Monopoly play.
        #
        # This deliberately avoids interfering with the
        # already-validated Road Building search when the
        # heuristic prefers another card or chooses HOLD.
        if (
            self.search_monopoly
            and baseline.card
            == DevCardType.MONOPOLY
        ):
            belief = build_monopoly_gain_belief(
                board,
                players,
                inventories,
                player.player_id,
            )

            self._evaluation_cache = {}
            self._search_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0

            state = self._make_search_state(
                board,
                players,
                player,
                inventories[player.player_id],
                dev_deck,
                bank,
            )

            # HOLD preserves the Monopoly card.
            (
                hold_value,
                _,
            ) = self._search_line(
                state,
                player.player_id,
                self.search_depth,
            )

            resources = (
                Resource.WOOD,
                Resource.BRICK,
                Resource.SHEEP,
                Resource.WHEAT,
                Resource.ORE,
            )

            candidates = []

            for resource in resources:
                distribution = belief[
                    resource
                ]

                expected_value = 0.0

                for (
                    collected,
                    probability,
                ) in distribution.items():
                    if probability <= 0.0:
                        continue

                    outcome_state = (
                        apply_search_monopoly_outcome(
                            state,
                            player.player_id,
                            resource,
                            collected,
                        )
                    )

                    (
                        outcome_value,
                        _,
                    ) = self._search_line(
                        outcome_state,
                        player.player_id,
                        self.search_depth,
                    )

                    expected_value += (
                        probability
                        * outcome_value
                    )

                candidates.append(
                    (
                        expected_value,
                        resource.value,
                        resource,
                    )
                )

            candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                )
            )

            (
                best_play_value,
                _,
                best_resource,
            ) = candidates[0]

            # Preserve the card on a tie.
            if best_play_value <= hold_value:
                return DevCardDecision(
                    card=None,
                    utility=hold_value,
                )

            return DevCardDecision(
                card=DevCardType.MONOPOLY,
                utility=best_play_value,
                resource=best_resource,
            )

        # ------------------------------------------------
        # Year of Plenty
        # ------------------------------------------------
        if (
            self.search_year_of_plenty
            and baseline.card in (
                None,
                DevCardType.YEAR_OF_PLENTY,
            )
            and has_playable_dev_card(
                player,
                DevCardType.YEAR_OF_PLENTY,
            )
        ):
            self._evaluation_cache = {}
            self._search_cache = {}
            self.cache_hits = 0
            self.cache_misses = 0

            state = self._make_search_state(
                board,
                players,
                player,
                inventories[player.player_id],
                dev_deck,
                bank,
            )

            (
                hold_value,
                _,
            ) = self._search_line(
                state,
                player.player_id,
                self.search_depth,
            )

            resources = (
                Resource.WOOD,
                Resource.BRICK,
                Resource.SHEEP,
                Resource.WHEAT,
                Resource.ORE,
            )

            play_candidates = []

            for index, resource_a in enumerate(
                resources
            ):
                for resource_b in resources[
                    index:
                ]:
                    required_a = (
                        2
                        if resource_a == resource_b
                        else 1
                    )

                    if not state.bank.can_supply(
                        resource_a,
                        required_a,
                    ):
                        continue

                    if (
                        resource_b != resource_a
                        and not state.bank.can_supply(
                            resource_b,
                            1,
                        )
                    ):
                        continue

                    next_state = (
                        apply_search_year_of_plenty(
                            state,
                            player.player_id,
                            resource_a,
                            resource_b,
                        )
                    )

                    (
                        play_value,
                        continuation,
                    ) = self._search_line(
                        next_state,
                        player.player_id,
                        self.search_depth,
                    )

                    play_candidates.append(
                        (
                            play_value,
                            resource_a.value,
                            resource_b.value,
                            resource_a,
                            resource_b,
                            continuation,
                        )
                    )

            if not play_candidates:
                return DevCardDecision(
                    card=None,
                    utility=hold_value,
                )

            play_candidates.sort(
                key=lambda item: (
                    -item[0],
                    item[1],
                    item[2],
                )
            )

            (
                best_play_value,
                _,
                _,
                resource_a,
                resource_b,
                _,
            ) = play_candidates[0]

            if best_play_value <= hold_value:
                return DevCardDecision(
                    card=None,
                    utility=hold_value,
                )

            return DevCardDecision(
                card=DevCardType.YEAR_OF_PLENTY,
                utility=best_play_value,
                resources=(
                    resource_a,
                    resource_b,
                ),
            )

        # ------------------------------------------------
        # Road Building
        # ------------------------------------------------

        # Preserve Knight, Monopoly, and YOP if the
        # existing policy prefers one of them.
        if baseline.card not in (
            None,
            DevCardType.ROAD_BUILDING,
        ):
            return baseline

        if not self.search_road_building:
            return baseline

        if not has_playable_dev_card(
            player,
            DevCardType.ROAD_BUILDING,
        ):
            return baseline

        self._evaluation_cache = {}
        self._search_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0

        state = self._make_search_state(
            board,
            players,
            player,
            inventories[player.player_id],
            dev_deck,
            bank,
        )

        # HOLD: keep the card and search ordinary
        # same-turn actions.
        (
            hold_value,
            _,
        ) = self._search_line(
            state,
            player.player_id,
            self.search_depth,
        )

        first_edges = legal_road_edges(
            state.board,
            state.players,
            state.players[player.player_id],
        )

        if not first_edges:
            return DevCardDecision(
                card=None,
                utility=hold_value,
            )

        play_candidates = []
        seen_sequences = set()

        for first in first_edges:
            # A one-road Road Building play is legal
            # under the existing simulator API.
            single_key = (first,)

            if single_key not in seen_sequences:
                seen_sequences.add(single_key)

                single_state = (
                    apply_search_road_building(
                        state,
                        player.player_id,
                        first,
                    )
                )

                (
                    single_value,
                    continuation,
                ) = self._search_line(
                    single_state,
                    player.player_id,
                    self.search_depth,
                )

                play_candidates.append(
                    (
                        single_value,
                        (first,),
                        continuation,
                    )
                )

            # Probe legality after the first free road
            # without consuming the development card.
            probe = (
                state.fast_clone_for_ordinary_search()
            )

            probe_player = (
                probe.players[player.player_id]
            )

            probe_player.roads.append(first)

            second_edges = legal_road_edges(
                probe.board,
                probe.players,
                probe_player,
            )

            for second in second_edges:
                # The same final pair may be reachable
                # in both orders. Evaluate it only once,
                # retaining the first known-legal order.
                final_key = tuple(
                    sorted(
                        (
                            first,
                            second,
                        )
                    )
                )

                if final_key in seen_sequences:
                    continue

                seen_sequences.add(final_key)

                next_state = (
                    apply_search_road_building(
                        state,
                        player.player_id,
                        first,
                        second,
                    )
                )

                (
                    play_value,
                    continuation,
                ) = self._search_line(
                    next_state,
                    player.player_id,
                    self.search_depth,
                )

                play_candidates.append(
                    (
                        play_value,
                        (
                            first,
                            second,
                        ),
                        continuation,
                    )
                )

        if not play_candidates:
            return DevCardDecision(
                card=None,
                utility=hold_value,
            )

        play_candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        (
            best_play_value,
            best_edges,
            _,
        ) = play_candidates[0]

        # Preserve the card on ties.
        if best_play_value <= hold_value:
            return DevCardDecision(
                card=None,
                utility=hold_value,
            )

        return DevCardDecision(
            card=DevCardType.ROAD_BUILDING,
            utility=best_play_value,
            road_edges=best_edges,
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
