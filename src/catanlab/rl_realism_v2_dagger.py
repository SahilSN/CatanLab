from __future__ import annotations

from catanlab.rl_realism_v2_agent import (
    RealismV2PolicyAgent,
)
from catanlab.rl_teacher import (
    DAggerExample,
    RecordingSearchAgent,
)


class RealismV2DAggerAgent(
    RealismV2PolicyAgent
):
    """
    Full realism-v2 learner whose ordinary-action states
    are labelled online by the frozen Search-v2 teacher.

    The learner controls the actual trajectory.

    Only ordinary actions receive DAgger supervision.
    All realism-v2 special decision families continue to
    use the learned policy inherited from
    RealismV2PolicyAgent.
    """

    def __init__(
        self,
        strategy,
        model,
        *,
        deterministic=True,
        seed=None,
    ):
        super().__init__(
            strategy,
            model=model,
            deterministic=deterministic,
            seed=seed,
        )

        # Exact frozen realism-v2 teacher configuration.
        self.teacher = RecordingSearchAgent(
            strategy,
            search_depth=2,
            use_transposition_cache=False,
            search_maritime_trades=True,
            search_year_of_plenty=True,
            search_road_building=True,
            search_monopoly=True,
            search_robber_decisions=True,
            search_discard_decisions=True,
            search_domestic_trades=True,
        )

        self.dagger_examples = []

    @property
    def dagger_agreements(self):
        return sum(
            example.agrees
            for example
            in self.dagger_examples
        )

    @property
    def dagger_disagreements(self):
        return (
            len(self.dagger_examples)
            - self.dagger_agreements
        )

    @property
    def dagger_agreement_rate(self):
        if not self.dagger_examples:
            return 0.0

        return (
            self.dagger_agreements
            / len(self.dagger_examples)
        )

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
                "RealismV2DAggerAgent requires "
                "dev_deck."
            )

        if bank is None:
            raise ValueError(
                "RealismV2DAggerAgent requires bank."
            )

        if inventories is None:
            raise ValueError(
                "RealismV2DAggerAgent requires "
                "inventories."
            )

        teacher_count_before = len(
            self.teacher.examples
        )

        # Query Search-v2 on exactly the state reached by
        # the learner. RecordingSearchAgent validates the
        # resulting action against the ordinary RL legal
        # mask and stores the canonical TeacherExample.
        self.teacher.choose_action(
            board,
            players,
            player,
            inventory,
            dev_deck=dev_deck,
            bank=bank,
            inventories=inventories,
        )

        teacher_count_after = len(
            self.teacher.examples
        )

        if (
            teacher_count_after
            != teacher_count_before + 1
        ):
            raise RuntimeError(
                "Search-v2 teacher did not record "
                "exactly one ordinary-action example."
            )

        teacher_example = (
            self.teacher.examples[-1]
        )

        # The learner, not Search-v2, controls the actual
        # game trajectory.
        learner_action = (
            super().choose_action(
                board,
                players,
                player,
                inventory,
                dev_deck=dev_deck,
                bank=bank,
                inventories=inventories,
            )
        )

        learner_action_id = (
            self.last_action_id
        )

        if learner_action_id is None:
            raise RuntimeError(
                "Realism-v2 learner did not record "
                "its selected ordinary action ID."
            )

        if not (
            teacher_example
            .legal_mask[
                learner_action_id
            ]
        ):
            raise RuntimeError(
                "Realism-v2 learner selected an "
                "ordinary action marked illegal by "
                "the teacher-state legal mask."
            )

        self.dagger_examples.append(
            DAggerExample(
                observation=(
                    teacher_example
                    .observation
                ),
                legal_mask=(
                    teacher_example
                    .legal_mask
                ),
                action_id=(
                    teacher_example
                    .action_id
                ),
                player_id=(
                    teacher_example
                    .player_id
                ),
                learner_action_id=(
                    learner_action_id
                ),
            )
        )

        return learner_action
