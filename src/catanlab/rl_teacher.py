from __future__ import annotations

from dataclasses import dataclass

from catanlab.action_space import (
    turn_action_id,
)
from catanlab.rl_interface import (
    build_rl_decision_context,
)
from catanlab.observation import (
    game_observation,
)
from catanlab.observation_encoder import (
    encode_game_observation,
)
from catanlab.rl_special_actions import (
    discard_counts,
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
from catanlab.search_agent import (
    OneStepLookaheadAgent,
)
from catanlab.rl_agent import (
    NeuralPolicyAgent,
)


@dataclass(frozen=True)
class TeacherExample:
    """
    One supervised ordinary-action training example.
    """

    observation: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    action_id: int

    player_id: int


class RecordingSearchAgent(
    OneStepLookaheadAgent
):
    """
    Frozen search policy that records its ordinary
    decisions in the RL representation.

    Search behavior itself is unchanged.
    """

    def __init__(
        self,
        strategy,
        search_depth: int = 2,
        use_transposition_cache: bool = False,
        search_maritime_trades: bool = True,
        search_year_of_plenty: bool = False,
        search_road_building: bool = True,
        search_monopoly: bool = False,
        search_robber_decisions: bool = False,
        search_discard_decisions: bool = False,
        search_domestic_trades: bool = False,
    ):
        super().__init__(
            strategy,
            search_depth=search_depth,
            use_transposition_cache=(
                use_transposition_cache
            ),
            search_maritime_trades=(
                search_maritime_trades
            ),
            search_year_of_plenty=(
                search_year_of_plenty
            ),
            search_road_building=(
                search_road_building
            ),
            search_monopoly=(
                search_monopoly
            ),
            search_robber_decisions=(
                search_robber_decisions
            ),
            search_discard_decisions=(
                search_discard_decisions
            ),
            search_domestic_trades=(
                search_domestic_trades
            ),
        )

        self.examples: list[
            TeacherExample
        ] = []

        self.v2_examples: list[
            "TeacherV2Example"
        ] = []

    def _v2_observation(
        self,
        board,
        players,
        inventories,
        player,
        bank,
        dev_deck,
    ) -> tuple[float, ...] | None:
        """
        Encode the same information-safe observation used
        by the ordinary learned policy.

        Compatibility/direct calls without bank or dev_deck
        preserve teacher behavior but are not recorded.
        """
        if bank is None or dev_deck is None:
            return None

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

    def _record_categorical_v2(
        self,
        *,
        decision_kind,
        observation,
        player_id,
        decision_input,
        value,
        allow_none_value: bool = False,
    ) -> None:
        """
        Record one categorical Search-v2 choice after
        validating it against its legal mask.
        """
        if observation is None:
            return

        if (
            value is None
            and not allow_none_value
        ):
            return

        label = decision_input.encode(
            value
        )

        if not decision_input.legal_mask[
            label
        ]:
            raise RuntimeError(
                "Search-v2 teacher selected a special "
                "decision marked illegal by its mask: "
                f"kind={decision_kind.value}, "
                f"label={label}, "
                f"value={value!r}"
            )

        from catanlab.rl_candidate_features import (
            encode_dynamic_decision_input,
            is_dynamic_decision_kind,
        )

        candidate_features = None

        if is_dynamic_decision_kind(
            decision_kind
        ):
            candidate_features = (
                encode_dynamic_decision_input(
                    decision_kind,
                    decision_input,
                )
            )

            if (
                len(candidate_features)
                != decision_input.action_dim
            ):
                raise RuntimeError(
                    "Teacher candidate encoding changed "
                    "categorical dimension: "
                    f"kind={decision_kind.value}, "
                    f"action_dim="
                    f"{decision_input.action_dim}, "
                    f"candidate_count="
                    f"{len(candidate_features)}"
                )

        self.v2_examples.append(
            TeacherV2Example(
                decision_kind=decision_kind,
                observation=observation,
                player_id=player_id,
                label=label,
                legal_mask=(
                    decision_input.legal_mask
                ),
                candidate_features=(
                    candidate_features
                ),
            )
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
        Preserve Search-v2 development-card behavior and
        record a Monopoly resource when Search-v2 actually
        commits one.
        """
        observation = self._v2_observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision = super().choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            phase,
            dev_deck=dev_deck,
            bank=bank,
        )

        if (
            self.search_monopoly
            and self._pending_monopoly_resource
            is not None
        ):
            decision_input = (
                monopoly_resource_decision_input()
            )

            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind
                    .MONOPOLY_RESOURCE
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=(
                    self._pending_monopoly_resource
                ),
            )

        if (
            self.search_year_of_plenty
            and self._pending_year_of_plenty_resources
            is not None
            and bank is not None
        ):
            decision_input = (
                year_of_plenty_decision_input(
                    bank
                )
            )

            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind
                    .YEAR_OF_PLENTY
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=(
                    self._pending_year_of_plenty_resources
                ),
            )

        if (
            self.search_road_building
            and self._pending_road_building_edges
            is not None
        ):
            decision_input = (
                road_building_decision_input(
                    board,
                    players,
                    player,
                )
            )

            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind
                    .ROAD_BUILDING
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=(
                    self._pending_road_building_edges
                ),
            )

        return decision

    def choose_robber_tile(
        self,
        board,
        players,
        inventories,
        player,
        bank=None,
        dev_deck=None,
    ):
        """
        Record Search-v2's categorical robber destination.
        """
        observation = self._v2_observation(
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

        tile_id = super().choose_robber_tile(
            board,
            players,
            inventories,
            player,
            bank=bank,
            dev_deck=dev_deck,
        )

        if self.search_robber_decisions:
            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind.ROBBER_TILE
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=tile_id,
            )

        return tile_id

    def choose_robber_victim(
        self,
        board,
        players,
        inventories,
        player,
        bank=None,
        dev_deck=None,
    ):
        """
        Record Search-v2's categorical robber victim when
        one is legally available.
        """
        observation = self._v2_observation(
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

        victim_id = (
            super().choose_robber_victim(
                board,
                players,
                inventories,
                player,
                bank=bank,
                dev_deck=dev_deck,
            )
        )

        if self.search_robber_decisions:
            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind.ROBBER_VICTIM
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=victim_id,
            )

        return victim_id

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
        """
        Record Search-v2's domestic-trade proposal.
        """
        observation = self._v2_observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            trade_proposal_decision_input(
                players,
                player,
                inventories[player.player_id],
                excluded_recipients=(
                    excluded_recipients
                ),
            )
        )

        offer = super().propose_player_trade(
            board,
            players,
            player,
            inventories,
            excluded_recipients=(
                excluded_recipients
            ),
            agents=agents,
        )

        if self.search_domestic_trades:
            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind
                    .TRADE_PROPOSAL
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=offer,
                allow_none_value=True,
            )

        return offer

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
        """
        Record Search-v2's accept/reject decision.
        """
        observation = self._v2_observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            trade_response_decision_input(
                offer,
                inventories,
            )
        )

        accepted = (
            super().evaluate_player_trade(
                board,
                players,
                player,
                inventories,
                offer,
            )
        )

        if self.search_domestic_trades:
            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind
                    .TRADE_RESPONSE
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=accepted,
            )

        return accepted

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
        """
        Record Search-v2's counteroffer decision.
        """
        observation = self._v2_observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            trade_counter_decision_input(
                players,
                player,
                inventories[player.player_id],
                offer,
                attempted_offers=(
                    attempted_offers
                ),
            )
        )

        counteroffer = (
            super().counter_player_trade(
                board,
                players,
                player,
                inventories,
                offer,
                attempted_offers=(
                    attempted_offers
                ),
            )
        )

        if self.search_domestic_trades:
            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind
                    .TRADE_COUNTER
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=counteroffer,
                allow_none_value=True,
            )

        return counteroffer

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
        """
        Record Search-v2's strategic discard multiset.
        """
        observation = self._v2_observation(
            board,
            players,
            inventories,
            player,
            bank,
            dev_deck,
        )

        decision_input = (
            discard_decision_input(
                inventory,
                count,
            )
        )

        discarded = (
            super().choose_discards_with_context(
                board,
                players,
                inventories,
                player,
                inventory,
                count,
                bank=bank,
                dev_deck=dev_deck,
            )
        )

        if self.search_discard_decisions:
            self._record_categorical_v2(
                decision_kind=(
                    TeacherDecisionKind.DISCARD
                ),
                observation=observation,
                player_id=player.player_id,
                decision_input=decision_input,
                value=discard_counts(
                    discarded
                ),
            )

        return discarded

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
        if dev_deck is None:
            raise ValueError(
                "RecordingSearchAgent requires "
                "dev_deck."
            )

        if bank is None:
            raise ValueError(
                "RecordingSearchAgent requires bank."
            )

        if inventories is None:
            raise ValueError(
                "RecordingSearchAgent requires "
                "inventories."
            )

        # Capture the policy input BEFORE the action
        # mutates any simulator state.
        context = build_rl_decision_context(
            board,
            players,
            inventories,
            player.player_id,
            bank,
            dev_deck,
        )

        action = super().choose_action(
            board,
            players,
            player,
            inventory,
            dev_deck=dev_deck,
            bank=bank,
        )

        action_id = turn_action_id(
            action,
            context.vocabulary,
        )

        if not (
            context.policy_input.legal_mask[
                action_id
            ]
        ):
            raise RuntimeError(
                "Search teacher selected an action "
                "that the RL legal mask marks illegal: "
                f"action_id={action_id}, "
                f"action={action}"
            )

        self.examples.append(
            TeacherExample(
                observation=(
                    context
                    .policy_input
                    .observation
                ),
                legal_mask=(
                    context
                    .policy_input
                    .legal_mask
                ),
                action_id=action_id,
                player_id=player.player_id,
            )
        )

        return action



@dataclass(frozen=True)
class DAggerExample:
    """
    One DAgger ordinary-action example.

    action_id is the search-teacher label.
    learner_action_id is the action actually selected
    by the neural policy on that state.
    """

    observation: tuple[float, ...]
    legal_mask: tuple[bool, ...]
    action_id: int
    player_id: int
    learner_action_id: int

    @property
    def agrees(self) -> bool:
        return (
            self.action_id
            == self.learner_action_id
        )


class DAggerAgent(
    NeuralPolicyAgent
):
    """
    Neural policy that controls the actual trajectory
    while a frozen search teacher labels every ordinary
    decision state.

    The teacher never controls the returned action.
    """

    def __init__(
        self,
        strategy,
        model,
        deterministic: bool = True,
        seed: int | None = None,
        search_depth: int = 2,
        use_transposition_cache: bool = False,
        search_maritime_trades: bool = True,
        search_year_of_plenty: bool = False,
        search_road_building: bool = True,
        search_monopoly: bool = False,
    ):
        super().__init__(
            strategy,
            model=model,
            deterministic=deterministic,
            seed=seed,
        )

        self.teacher = OneStepLookaheadAgent(
            strategy,
            search_depth=search_depth,
            use_transposition_cache=(
                use_transposition_cache
            ),
            search_maritime_trades=(
                search_maritime_trades
            ),
            search_year_of_plenty=(
                search_year_of_plenty
            ),
            search_road_building=(
                search_road_building
            ),
            search_monopoly=(
                search_monopoly
            ),
        )

        self.examples: list[
            DAggerExample
        ] = []

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
        if dev_deck is None:
            raise ValueError(
                "DAggerAgent requires dev_deck."
            )

        if bank is None:
            raise ValueError(
                "DAggerAgent requires bank."
            )

        if inventories is None:
            raise ValueError(
                "DAggerAgent requires inventories."
            )

        # Snapshot the learner-visible state before
        # either policy returns an action.
        context = build_rl_decision_context(
            board,
            players,
            inventories,
            player.player_id,
            bank,
            dev_deck,
        )

        # Query the frozen search policy on exactly the
        # state reached by the learner.
        teacher_action = (
            self.teacher.choose_action(
                board,
                players,
                player,
                inventory,
                dev_deck=dev_deck,
                bank=bank,
            )
        )

        teacher_action_id = turn_action_id(
            teacher_action,
            context.vocabulary,
        )

        if not context.policy_input.legal_mask[
            teacher_action_id
        ]:
            raise RuntimeError(
                "DAgger teacher selected an action "
                "marked illegal by the RL mask: "
                f"action_id={teacher_action_id}, "
                f"action={teacher_action}"
            )

        # The neural policy, not the teacher, controls
        # the trajectory.
        learner_action = super().choose_action(
            board,
            players,
            player,
            inventory,
            dev_deck=dev_deck,
            bank=bank,
            inventories=inventories,
        )

        learner_action_id = self.last_action_id

        if learner_action_id is None:
            raise RuntimeError(
                "Neural policy did not record its "
                "selected action ID."
            )

        if not context.policy_input.legal_mask[
            learner_action_id
        ]:
            raise RuntimeError(
                "DAgger learner selected an action "
                "marked illegal by the RL mask."
            )

        self.examples.append(
            DAggerExample(
                observation=(
                    context
                    .policy_input
                    .observation
                ),
                legal_mask=(
                    context
                    .policy_input
                    .legal_mask
                ),
                action_id=teacher_action_id,
                player_id=player.player_id,
                learner_action_id=(
                    learner_action_id
                ),
            )
        )

        return learner_action


# ---------------------------------------------------------------------------
# Realism-v2 teacher data
# ---------------------------------------------------------------------------

from enum import Enum


class TeacherDecisionKind(str, Enum):
    """
    Phase-specific decision types supervised by a
    realism-v2 teacher.

    ORDINARY_ACTION preserves the existing flat RL action
    space. The remaining kinds deliberately use their own
    small decision spaces rather than expanding that flat
    vocabulary.
    """

    ORDINARY_ACTION = "ordinary_action"

    ROBBER_TILE = "robber_tile"
    ROBBER_VICTIM = "robber_victim"

    DISCARD = "discard"

    MONOPOLY_RESOURCE = "monopoly_resource"
    YEAR_OF_PLENTY = "year_of_plenty"
    ROAD_BUILDING = "road_building"

    TRADE_PROPOSAL = "trade_proposal"
    TRADE_RESPONSE = "trade_response"
    TRADE_COUNTER = "trade_counter"


@dataclass(frozen=True)
class TeacherV2Example:
    """
    One phase-specific Search-v2 supervision example.

    `observation` remains the information-safe encoded
    game observation.

    `legal_mask` is optional because not every structured
    decision maps naturally to a single fixed categorical
    vocabulary.

    `label` contains the canonical teacher choice for the
    decision kind. Its interpretation is defined by the
    corresponding phase-specific codec.
    """

    decision_kind: TeacherDecisionKind
    observation: tuple[float, ...]
    player_id: int

    label: object

    legal_mask: tuple[bool, ...] | None = None

    # Dynamic decision kinds require the ordered candidate
    # representation that gives the categorical label its
    # meaning. Fixed-size heads leave this as None because
    # their vocabularies are globally defined.
    candidate_features: (
        tuple[
            tuple[float, ...],
            ...
        ]
        | None
    ) = None

    @property
    def has_legal_mask(self) -> bool:
        return self.legal_mask is not None

    @property
    def has_candidate_features(self) -> bool:
        return self.candidate_features is not None
