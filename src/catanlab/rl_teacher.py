from __future__ import annotations

from dataclasses import dataclass

from catanlab.action_space import (
    turn_action_id,
)
from catanlab.rl_interface import (
    build_rl_decision_context,
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
        )

        self.examples: list[
            TeacherExample
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
