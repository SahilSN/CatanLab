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
        search_robber_decisions: bool = False,
        search_discard_decisions: bool = False,
        search_domestic_trades: bool = False,
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
        self.search_robber_decisions = (
            search_robber_decisions
        )
        self.search_discard_decisions = (
            search_discard_decisions
        )
        self.search_domestic_trades = (
            search_domestic_trades
        )

        self._evaluation_cache = {}
        self._search_cache = {}

        self.cache_hits = 0
        self.cache_misses = 0

        # Short-lived arguments selected while evaluating
        # a development-card play. These are consumed by
        # the specialized execution hooks immediately after
        # choose_dev_card_play() returns.
        self._pending_monopoly_resource = None
        self._pending_year_of_plenty_resources = None
        self._pending_road_building_edges = None

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

    @staticmethod
    def _copy_resource_inventory(
        inventory,
    ):
        """
        Return an independent copy of a resource hand.
        """
        from catanlab.economy import PlayerInventory
        from catanlab.resources import Resource

        copied = PlayerInventory()

        for resource in (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        ):
            amount = inventory.count(
                resource
            )

            if amount:
                copied.add(
                    resource,
                    amount,
                )

        return copied

    def _domestic_trade_hand_value(
        self,
        board,
        players,
        player,
        inventory,
    ) -> float:
        """
        Evaluate this player's resource hand for domestic
        trade decisions.

        The value mirrors the build-readiness model used
        by Search-v2 proposal and discard decisions.
        """
        from catanlab.economy import (
            BUILD_COSTS,
            BuildType,
        )
        from catanlab.resources import Resource
        from catanlab.turns import (
            legal_road_edges,
            legal_settlement_vertices,
        )

        build_available = {
            BuildType.CITY: bool(
                player.settlements
            ),
            BuildType.SETTLEMENT: bool(
                legal_settlement_vertices(
                    board,
                    players,
                    player,
                )
            ),
            BuildType.ROAD: bool(
                legal_road_edges(
                    board,
                    players,
                    player,
                )
            ),
            BuildType.DEV_CARD: True,
        }

        build_weights = {
            BuildType.CITY: 2.0,
            BuildType.SETTLEMENT: 1.8,
            BuildType.DEV_CARD: 1.2,
            BuildType.ROAD: 0.8,
        }

        value = 0.0

        for build_type in (
            BuildType.CITY,
            BuildType.SETTLEMENT,
            BuildType.DEV_CARD,
            BuildType.ROAD,
        ):
            if not build_available[
                build_type
            ]:
                continue

            cost = BUILD_COSTS[
                build_type
            ]

            total_required = sum(
                cost.values()
            )

            satisfied = sum(
                min(
                    inventory.count(
                        resource
                    ),
                    required,
                )
                for resource, required
                in cost.items()
            )

            weight = build_weights[
                build_type
            ]

            if total_required:
                value += (
                    weight
                    * satisfied
                    / total_required
                )

            if inventory.can_afford(
                build_type
            ):
                value += weight

        value += 0.05 * sum(
            inventory.count(resource) > 0
            for resource in (
                Resource.WOOD,
                Resource.BRICK,
                Resource.SHEEP,
                Resource.WHEAT,
                Resource.ORE,
            )
        )

        return value

    def _simulate_domestic_trade_hand(
        self,
        inventory,
        *,
        outgoing,
        incoming,
    ):
        """
        Return this player's hypothetical post-trade hand.

        `outgoing` is what this player gives and `incoming`
        is what this player receives.
        """
        simulated = self._copy_resource_inventory(
            inventory
        )

        for resource, amount in outgoing:
            if (
                simulated.count(resource)
                < amount
            ):
                return None

            simulated.remove(
                resource,
                amount,
            )

        for resource, amount in incoming:
            simulated.add(
                resource,
                amount,
            )

        return simulated

    def propose_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        excluded_recipients=None,
        agents=None,
    ):
        """
        Search-v2 domestic-trade proposal policy.

        Enumerate compact 1-for-1 offers and select an
        exchange that materially improves the acting
        player's own build readiness.

        Recipient hidden resource identities are never
        inspected. A recipient may therefore be asked for
        a card they do not hold; normal trade validation
        and negotiation handle that case.
        """
        if not self.search_domestic_trades:
            return super().propose_player_trade(
                board,
                players,
                player,
                inventories,
                excluded_recipients=(
                    excluded_recipients
                ),
                agents=agents,
            )

        from catanlab.resources import Resource
        from catanlab.trading import (
            TradeOffer,
            validate_trade_terms,
        )

        if excluded_recipients is None:
            excluded_recipients = set()
        else:
            excluded_recipients = set(
                excluded_recipients
            )

        inventory = inventories[
            player.player_id
        ]

        resources = (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        )

        recipients = [
            other.player_id
            for other in players
            if (
                other.player_id
                != player.player_id
                and other.player_id
                not in excluded_recipients
            )
        ]

        if not recipients:
            return None

        before_value = (
            self._domestic_trade_hand_value(
                board,
                players,
                player,
                inventory,
            )
        )

        candidates = []

        for give_resource in resources:
            if inventory.count(
                give_resource
            ) <= 0:
                continue

            for receive_resource in resources:
                if receive_resource == give_resource:
                    continue

                simulated = (
                    self._simulate_domestic_trade_hand(
                        inventory,
                        outgoing=(
                            (
                                give_resource,
                                1,
                            ),
                        ),
                        incoming=(
                            (
                                receive_resource,
                                1,
                            ),
                        ),
                    )
                )

                if simulated is None:
                    continue

                after_value = (
                    self._domestic_trade_hand_value(
                        board,
                        players,
                        player,
                        simulated,
                    )
                )

                gain = (
                    after_value
                    - before_value
                )

                # Do not negotiate for a merely equivalent
                # hand. Search must identify a real benefit.
                if gain <= 1e-9:
                    continue

                for recipient_id in recipients:
                    offer = TradeOffer(
                        proposer_id=(
                            player.player_id
                        ),
                        recipient_id=recipient_id,
                        give=(
                            (
                                give_resource,
                                1,
                            ),
                        ),
                        receive=(
                            (
                                receive_resource,
                                1,
                            ),
                        ),
                    )

                    if not validate_trade_terms(
                        offer
                    ):
                        continue

                    # Recipient VP is public. Prefer trading
                    # with the less threatening opponent when
                    # otherwise indifferent.
                    recipient_threat = (
                        players[
                            recipient_id
                        ].public_victory_points
                    )

                    candidates.append(
                        (
                            gain,
                            -recipient_threat,
                            -recipient_id,
                            give_resource.value,
                            receive_resource.value,
                            offer,
                        )
                    )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                -item[0],
                -item[1],
                -item[2],
                item[3],
                item[4],
            )
        )

        return candidates[0][-1]

    def evaluate_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
    ) -> bool:
        """
        Evaluate an incoming domestic trade directly with
        Search-v2's own hand-value model.
        """
        if not self.search_domestic_trades:
            return super().evaluate_player_trade(
                board,
                players,
                player,
                inventories,
                offer,
            )

        from catanlab.trading import (
            validate_trade_offer,
        )

        if (
            offer.recipient_id
            != player.player_id
        ):
            return False

        # The recipient is allowed to verify that the
        # offered transaction is actually feasible because
        # they know their own hand. The game engine also
        # performs this validation.
        if not validate_trade_offer(
            offer,
            inventories,
        ):
            return False

        proposer = players[
            offer.proposer_id
        ]

        # Preserve the established public-threat safeguard.
        if proposer.public_victory_points >= 9:
            return False

        inventory = inventories[
            player.player_id
        ]

        before_value = (
            self._domestic_trade_hand_value(
                board,
                players,
                player,
                inventory,
            )
        )

        # From recipient perspective:
        #   give     = offer.receive
        #   receive  = offer.give
        simulated = (
            self._simulate_domestic_trade_hand(
                inventory,
                outgoing=offer.receive,
                incoming=offer.give,
            )
        )

        if simulated is None:
            return False

        after_value = (
            self._domestic_trade_hand_value(
                board,
                players,
                player,
                simulated,
            )
        )

        gain = (
            after_value
            - before_value
        )

        # Require a real improvement rather than accepting
        # neutral exchanges that can create excessive
        # negotiation churn.
        return gain > 1e-9

    def counter_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
        attempted_offers=None,
    ):
        """
        Generate the best improving 1-for-1 Search-v2
        counteroffer.

        Candidate requests do not inspect the other
        player's hidden resource identities.
        """
        if not self.search_domestic_trades:
            return super().counter_player_trade(
                board,
                players,
                player,
                inventories,
                offer,
                attempted_offers=(
                    attempted_offers
                ),
            )

        from catanlab.resources import Resource
        from catanlab.trading import (
            TradeOffer,
            validate_trade_terms,
        )

        if (
            offer.recipient_id
            != player.player_id
        ):
            return None

        if attempted_offers is None:
            attempted_offers = set()

        inventory = inventories[
            player.player_id
        ]

        resources = (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        )

        before_value = (
            self._domestic_trade_hand_value(
                board,
                players,
                player,
                inventory,
            )
        )

        candidates = []

        for give_resource in resources:
            if inventory.count(
                give_resource
            ) <= 0:
                continue

            for receive_resource in resources:
                if (
                    receive_resource
                    == give_resource
                ):
                    continue

                simulated = (
                    self._simulate_domestic_trade_hand(
                        inventory,
                        outgoing=(
                            (
                                give_resource,
                                1,
                            ),
                        ),
                        incoming=(
                            (
                                receive_resource,
                                1,
                            ),
                        ),
                    )
                )

                if simulated is None:
                    continue

                after_value = (
                    self._domestic_trade_hand_value(
                        board,
                        players,
                        player,
                        simulated,
                    )
                )

                gain = (
                    after_value
                    - before_value
                )

                if gain <= 1e-9:
                    continue

                candidate = TradeOffer(
                    proposer_id=(
                        player.player_id
                    ),
                    recipient_id=(
                        offer.proposer_id
                    ),
                    give=(
                        (
                            give_resource,
                            1,
                        ),
                    ),
                    receive=(
                        (
                            receive_resource,
                            1,
                        ),
                    ),
                )

                if candidate in attempted_offers:
                    continue

                # Structural validation only. Do not use
                # validate_trade_offer(), because doing so
                # here would reveal whether the opponent
                # actually owns the requested card.
                if not validate_trade_terms(
                    candidate
                ):
                    continue

                candidates.append(
                    (
                        gain,
                        give_resource.value,
                        receive_resource.value,
                        candidate,
                    )
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
                item[2],
            )
        )

        return candidates[0][-1]

    def choose_discards_with_context(
        self,
        board,
        players,
        inventories,
        player,
        inventory,
        count,
    ):
        """
        Search over legal discard multisets and retain
        the remaining hand with the greatest strategic
        build readiness.

        Only the acting player's private resource hand
        is inspected. Opponent resource identities are
        irrelevant to this decision.
        """
        if not self.search_discard_decisions:
            return super().choose_discards_with_context(
                board,
                players,
                inventories,
                player,
                inventory,
                count,
            )

        if count <= 0:
            return []

        from catanlab.economy import (
            BUILD_COSTS,
            BuildType,
            PlayerInventory,
        )
        from catanlab.resources import Resource
        from catanlab.turns import (
            legal_road_edges,
            legal_settlement_vertices,
        )

        resources = (
            Resource.WOOD,
            Resource.BRICK,
            Resource.SHEEP,
            Resource.WHEAT,
            Resource.ORE,
        )

        if count > inventory.total():
            raise ValueError(
                "Cannot discard more cards than are held."
            )

        # ------------------------------------------------
        # Which build goals are structurally reachable?
        # ------------------------------------------------

        build_available = {
            BuildType.CITY: bool(
                player.settlements
            ),
            BuildType.SETTLEMENT: bool(
                legal_settlement_vertices(
                    board,
                    players,
                    player,
                )
            ),
            BuildType.ROAD: bool(
                legal_road_edges(
                    board,
                    players,
                    player,
                )
            ),
            BuildType.DEV_CARD: True,
        }

        weights = {
            BuildType.CITY: 2.0,
            BuildType.SETTLEMENT: 1.8,
            BuildType.DEV_CARD: 1.2,
            BuildType.ROAD: 0.8,
        }

        def remaining_inventory(discard_counts):
            remaining = PlayerInventory()

            for resource, discarded_count in zip(
                resources,
                discard_counts,
            ):
                kept = (
                    inventory.count(resource)
                    - discarded_count
                )

                if kept:
                    remaining.add(
                        resource,
                        kept,
                    )

            return remaining

        def score_remaining(remaining):
            """
            Mirror the resource-readiness component of
            the ordinary search evaluator while adding
            a bonus for already-affordable builds.
            """
            value = 0.0

            for build_type in (
                BuildType.CITY,
                BuildType.SETTLEMENT,
                BuildType.DEV_CARD,
                BuildType.ROAD,
            ):
                if not build_available[
                    build_type
                ]:
                    continue

                cost = BUILD_COSTS[
                    build_type
                ]

                total_required = sum(
                    cost.values()
                )

                satisfied = sum(
                    min(
                        remaining.count(
                            resource
                        ),
                        required,
                    )
                    for resource, required
                    in cost.items()
                )

                weight = weights[
                    build_type
                ]

                if total_required:
                    value += (
                        weight
                        * satisfied
                        / total_required
                    )

                if remaining.can_afford(
                    build_type
                ):
                    value += weight

            # Secondary flexibility reward:
            # preserve diversity once immediate build
            # readiness has been accounted for.
            value += 0.05 * sum(
                remaining.count(resource) > 0
                for resource in resources
            )

            return value

        candidates = []

        held = tuple(
            inventory.count(resource)
            for resource in resources
        )

        def enumerate_counts(
            index,
            remaining_to_discard,
            prefix,
        ):
            if index == len(resources):
                if remaining_to_discard == 0:
                    discard_counts = tuple(
                        prefix
                    )

                    remaining = (
                        remaining_inventory(
                            discard_counts
                        )
                    )

                    value = score_remaining(
                        remaining
                    )

                    candidates.append(
                        (
                            value,
                            discard_counts,
                        )
                    )

                return

            max_take = min(
                held[index],
                remaining_to_discard,
            )

            for take in range(
                max_take + 1
            ):
                enumerate_counts(
                    index + 1,
                    remaining_to_discard
                    - take,
                    (
                        *prefix,
                        take,
                    ),
                )

        enumerate_counts(
            0,
            count,
            (),
        )

        if not candidates:
            raise ValueError(
                "No legal discard multiset found."
            )

        # Highest strategic value wins.
        #
        # On equal values, prefer the lexicographically
        # smallest discard-count vector for completely
        # deterministic behavior.
        candidates.sort(
            key=lambda item: (
                -item[0],
                item[1],
            )
        )

        _, best_counts = candidates[0]

        discarded = []

        for resource, amount in zip(
            resources,
            best_counts,
        ):
            discarded.extend(
                [resource] * amount
            )

        return discarded

    def choose_robber_tile(
        self,
        board,
        players,
        inventories,
        player,
    ):
        """
        Choose a robber destination using only legally
        observable information.

        Search-v2 explicitly evaluates every destination
        rather than delegating the choice to the inherited
        Core-v1 policy.
        """
        if not self.search_robber_decisions:
            return super().choose_robber_tile(
                board,
                players,
                inventories,
                player,
            )

        from catanlab.dice import production_weight

        candidates = [
            tile
            for tile in board.tiles
            if tile.id != board.robber_tile_id
        ]

        if not candidates:
            return None

        def buildings_on_tile(
            candidate_player,
            tile_id,
        ):
            settlements = sum(
                1
                for vertex_id
                in candidate_player.settlements
                if tile_id
                in board.vertices[
                    vertex_id
                ].adjacent_tiles
            )

            cities = sum(
                1
                for vertex_id
                in candidate_player.cities
                if tile_id
                in board.vertices[
                    vertex_id
                ].adjacent_tiles
            )

            return settlements, cities

        def tile_score(tile):
            probability_weight = (
                production_weight(tile.number)
                if tile.number is not None
                else 0.0
            )

            opponent_denial = 0.0
            self_denial = 0.0
            steal_value = 0.0

            for other in players:
                settlements, cities = (
                    buildings_on_tile(
                        other,
                        tile.id,
                    )
                )

                production_units = (
                    settlements
                    + 2 * cities
                )

                if production_units <= 0:
                    continue

                blocked_value = (
                    production_units
                    * probability_weight
                )

                if (
                    other.player_id
                    == player.player_id
                ):
                    self_denial += blocked_value
                    continue

                threat = (
                    1.0
                    + 0.20
                    * other.public_victory_points
                )

                opponent_denial += (
                    blocked_value
                    * threat
                )

                # Resource identities are private.
                # Total hand size is public and is the
                # only opponent inventory information used.
                public_hand_size = (
                    inventories[
                        other.player_id
                    ].total()
                )

                if public_hand_size > 0:
                    steal_value = max(
                        steal_value,
                        1.0
                        + 0.10
                        * min(
                            public_hand_size,
                            7,
                        )
                        + 0.20
                        * other.public_victory_points,
                    )

            value = (
                opponent_denial
                - 1.75 * self_denial
                + 0.85 * steal_value
            )

            return (
                value,
                -tile.id,
            )

        return max(
            candidates,
            key=tile_score,
        ).id

    def choose_robber_victim(
        self,
        board,
        players,
        inventories,
        player,
    ):
        """
        Choose a robber victim using only public VP and
        public resource-card count.
        """
        if not self.search_robber_decisions:
            return super().choose_robber_victim(
                board,
                players,
                inventories,
                player,
            )

        from catanlab.devcards import (
            players_adjacent_to_tile,
        )

        if board.robber_tile_id is None:
            return None

        adjacent = players_adjacent_to_tile(
            board,
            players,
            board.robber_tile_id,
            exclude_player_id=player.player_id,
        )

        eligible = [
            victim_id
            for victim_id in adjacent
            if inventories[victim_id].total() > 0
        ]

        if not eligible:
            return None

        return max(
            eligible,
            key=lambda victim_id: (
                players[
                    victim_id
                ].public_victory_points,
                min(
                    inventories[
                        victim_id
                    ].total(),
                    7,
                ),
                -victim_id,
            ),
        )

    def choose_monopoly_resource(
        self,
        board,
        players,
        inventories,
        player,
        suggested_resource=None,
    ):
        """
        Consume a Monopoly resource selected by Search v2.

        Fall back to the normal TurnAgent contract when no
        search-owned choice is pending.
        """
        if self._pending_monopoly_resource is not None:
            resource = self._pending_monopoly_resource
            self._pending_monopoly_resource = None
            return resource

        return super().choose_monopoly_resource(
            board,
            players,
            inventories,
            player,
            suggested_resource=suggested_resource,
        )

    def choose_year_of_plenty_resources(
        self,
        board,
        players,
        inventories,
        player,
        bank=None,
        suggested_resources=None,
    ):
        """
        Consume a Year of Plenty pair selected by Search v2.
        """
        if (
            self._pending_year_of_plenty_resources
            is not None
        ):
            resources = (
                self._pending_year_of_plenty_resources
            )
            self._pending_year_of_plenty_resources = None
            return resources

        return super().choose_year_of_plenty_resources(
            board,
            players,
            inventories,
            player,
            bank=bank,
            suggested_resources=suggested_resources,
        )

    def choose_road_building_edges(
        self,
        board,
        players,
        inventories,
        player,
        suggested_edges=None,
    ):
        """
        Consume Road Building edges selected by Search v2.
        """
        if self._pending_road_building_edges is not None:
            edges = self._pending_road_building_edges
            self._pending_road_building_edges = None
            return edges

        return super().choose_road_building_edges(
            board,
            players,
            inventories,
            player,
            suggested_edges=suggested_edges,
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
        self._pending_monopoly_resource = None
        self._pending_year_of_plenty_resources = None
        self._pending_road_building_edges = None

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

            self._pending_monopoly_resource = (
                best_resource
            )

            return DevCardDecision(
                card=DevCardType.MONOPOLY,
                utility=best_play_value,
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

            self._pending_year_of_plenty_resources = (
                resource_a,
                resource_b,
            )

            return DevCardDecision(
                card=DevCardType.YEAR_OF_PLENTY,
                utility=best_play_value,
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

        self._pending_road_building_edges = (
            best_edges
        )

        return DevCardDecision(
            card=DevCardType.ROAD_BUILDING,
            utility=best_play_value,
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
