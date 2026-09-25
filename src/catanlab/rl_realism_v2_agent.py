from __future__ import annotations

from collections import Counter

from catanlab.economy import Resource
from catanlab.observation import (
    game_observation,
)
from catanlab.observation_encoder import (
    encode_game_observation,
)
from catanlab.rl_agent import (
    NeuralPolicyAgent,
)
from catanlab.rl_decision_policy import (
    LearnedDecisionRequest,
    TorchRealismV2DecisionPolicy,
    choose_decision_value,
)
from catanlab.rl_special_actions import (
    discard_decision_input,
    monopoly_resource_decision_input,
    road_building_decision_input,
    robber_tile_decision_input,
    robber_victim_decision_input,
    trade_counter_decision_input,
    trade_proposal_decision_input,
    trade_response_decision_input,
    year_of_plenty_decision_input,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)


_RESOURCE_ORDER = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


class RealismV2PolicyAgent(
    NeuralPolicyAgent
):
    """
    Full-game inference adapter for RealismV2ActorCritic.

    ORDINARY_ACTION continues through NeuralPolicyAgent's
    established 202-action inference path.

    Every realism-v2 special decision is routed through
    TorchRealismV2DecisionPolicy using the same categorical
    decision-input builders used by teacher-v2 generation.

    No heuristic fallback is permitted for an encountered
    realism-v2 special decision.
    """

    def __init__(
        self,
        strategy,
        model,
        deterministic: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            strategy,
            model=model,
            deterministic=deterministic,
            seed=seed,
        )

        self.decision_policy = (
            TorchRealismV2DecisionPolicy(
                model,
                deterministic=deterministic,
                seed=seed,
            )
        )

        self.decision_counts = Counter(
            {
                kind.value: 0
                for kind in TeacherDecisionKind
            }
        )

        self.illegal_decisions = 0
        self.fallbacks = 0

        # Some historical dev-card parameter hooks do not
        # receive bank/dev_deck. Cache the complete context
        # from choose_dev_card_play(), immediately before
        # those hooks are exercised.
        self._dev_decision_context = None

    def _observation(
        self,
        board,
        players,
        inventories,
        player,
        bank,
        dev_deck,
    ) -> tuple[float, ...]:
        if bank is None:
            raise ValueError(
                "RealismV2PolicyAgent requires bank "
                "for learned special decisions."
            )

        if dev_deck is None:
            raise ValueError(
                "RealismV2PolicyAgent requires dev_deck "
                "for learned special decisions."
            )

        observation = game_observation(
            board,
            players,
            inventories,
            player.player_id,
            bank,
            dev_deck,
        )

        return encode_game_observation(
            observation
        )

    def _choose_special(
        self,
        *,
        decision_kind,
        observation,
        decision_input,
    ):
        request = LearnedDecisionRequest(
            decision_kind=decision_kind,
            observation=observation,
            decision_input=decision_input,
        )

        try:
            value = choose_decision_value(
                self.decision_policy,
                request,
            )
        except Exception:
            # Do not recover or silently delegate to a
            # heuristic. The smoke benchmark must expose
            # every integration failure.
            self.illegal_decisions += 1
            raise

        self.decision_counts[
            decision_kind.value
        ] += 1

        return value

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
        action = super().choose_action(
            board,
            players,
            player,
            inventory,
            dev_deck=dev_deck,
            bank=bank,
            inventories=inventories,
        )

        self.decision_counts[
            TeacherDecisionKind
            .ORDINARY_ACTION
            .value
        ] += 1

        return action

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
        if bank is None:
            raise ValueError(
                "RealismV2PolicyAgent requires bank "
                "during development-card play."
            )

        if dev_deck is None:
            raise ValueError(
                "RealismV2PolicyAgent requires dev_deck "
                "during development-card play."
            )

        self._dev_decision_context = (
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        return super().choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            phase,
        )

    def _cached_dev_context(
        self,
        player,
    ):
        if self._dev_decision_context is None:
            raise RuntimeError(
                "A learned development-card parameter "
                "decision was requested without a cached "
                "choose_dev_card_play context."
            )

        (
            board,
            players,
            inventories,
            cached_player,
            bank,
            dev_deck,
        ) = self._dev_decision_context

        if (
            cached_player.player_id
            != player.player_id
        ):
            raise RuntimeError(
                "Cached development-card context belongs "
                "to a different player."
            )

        return (
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

    def choose_robber_tile(
        self,
        board,
        players,
        inventories,
        player,
        bank=None,
        dev_deck=None,
    ):
        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            robber_tile_decision_input(
                board
            )
        )

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind.ROBBER_TILE
            ),
            observation=observation,
            decision_input=decision_input,
        )

    def choose_robber_victim(
        self,
        board,
        players,
        inventories,
        player,
        bank=None,
        dev_deck=None,
    ):
        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            robber_victim_decision_input(
                board,
                players,
                inventories,
                player,
            )
        )

        # Moving the robber does not always produce a
        # victim choice. If no adjacent opponent has a
        # legally robbable hand, there is no categorical
        # decision for the learned policy to make.
        if not decision_input.legal_action_ids:
            return None

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind.ROBBER_VICTIM
            ),
            observation=observation,
            decision_input=decision_input,
        )

    def choose_monopoly_resource(
        self,
        board,
        players,
        inventories,
        player,
        suggested_resource=None,
    ):
        (
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        ) = self._cached_dev_context(
            player
        )

        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind
                .MONOPOLY_RESOURCE
            ),
            observation=observation,
            decision_input=(
                monopoly_resource_decision_input()
            ),
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
        (
            board,
            players,
            inventories,
            player,
            cached_bank,
            dev_deck,
        ) = self._cached_dev_context(
            player
        )

        if bank is None:
            bank = cached_bank

        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind
                .YEAR_OF_PLENTY
            ),
            observation=observation,
            decision_input=(
                year_of_plenty_decision_input(
                    bank
                )
            ),
        )

    def choose_road_building_edges(
        self,
        board,
        players,
        inventories,
        player,
        suggested_edges=None,
    ):
        (
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        ) = self._cached_dev_context(
            player
        )

        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            road_building_decision_input(
                board,
                players,
                player,
            )
        )

        # Road Building can be resolved in a state where
        # no legal road placements remain. In that case,
        # there is no categorical choice for the learned
        # policy to make.
        if decision_input.action_dim == 0:
            return ()

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind
                .ROAD_BUILDING
            ),
            observation=observation,
            decision_input=decision_input,
        )
    def choose_discards_with_context(
        self,
        board,
        players,
        inventories,
        player,
        inventory,
        count,
        bank=None,
        dev_deck=None,
    ):
        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        counts = self._choose_special(
            decision_kind=(
                TeacherDecisionKind.DISCARD
            ),
            observation=observation,
            decision_input=(
                discard_decision_input(
                    inventory,
                    count,
                )
            ),
        )

        # discard_decision_input's simulator-facing
        # categorical value is the canonical five-resource
        # count vector used by teacher-v2 generation.
        if (
            not isinstance(
                counts,
                (tuple, list),
            )
            or len(counts) != 5
        ):
            raise RuntimeError(
                "Decoded discard decision must be a "
                "five-resource count vector: "
                f"{counts!r}"
            )

        discarded = []

        for resource, resource_count in zip(
            _RESOURCE_ORDER,
            counts,
        ):
            if (
                not isinstance(
                    resource_count,
                    int,
                )
                or resource_count < 0
            ):
                raise RuntimeError(
                    "Invalid decoded discard count: "
                    f"{counts!r}"
                )

            discarded.extend(
                [resource] * resource_count
            )

        if len(discarded) != count:
            raise RuntimeError(
                "Decoded discard decision has wrong "
                "total count: "
                f"expected={count}, "
                f"decoded={len(discarded)}, "
                f"counts={counts!r}"
            )

        return discarded

    def propose_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        excluded_recipients=None,
        agents=None,
        bank=None,
        dev_deck=None,
    ):
        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind
                .TRADE_PROPOSAL
            ),
            observation=observation,
            decision_input=(
                trade_proposal_decision_input(
                    players,
                    player,
                    inventories[
                        player.player_id
                    ],
                    excluded_recipients=(
                        excluded_recipients
                    ),
                )
            ),
        )

    def evaluate_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
        bank=None,
        dev_deck=None,
    ) -> bool:
        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        return bool(
            self._choose_special(
                decision_kind=(
                    TeacherDecisionKind
                    .TRADE_RESPONSE
                ),
                observation=observation,
                decision_input=(
                    trade_response_decision_input(
                        offer,
                        inventories,
                    )
                ),
            )
        )

    def counter_player_trade(
        self,
        board,
        players,
        player,
        inventories,
        offer,
        attempted_offers=None,
        bank=None,
        dev_deck=None,
    ):
        observation = self._observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        return self._choose_special(
            decision_kind=(
                TeacherDecisionKind
                .TRADE_COUNTER
            ),
            observation=observation,
            decision_input=(
                trade_counter_decision_input(
                    players,
                    player,
                    inventories[
                        player.player_id
                    ],
                    offer,
                    attempted_offers=(
                        attempted_offers
                    ),
                )
            ),
        )
