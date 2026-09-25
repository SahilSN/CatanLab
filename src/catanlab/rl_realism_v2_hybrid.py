from __future__ import annotations

from collections import Counter

from catanlab.rl_realism_v2_agent import (
    RealismV2PolicyAgent,
)
from catanlab.rl_teacher import (
    RecordingSearchAgent,
)


VALID_INTERVENTIONS = frozenset(
    {
        "ordinary",
        "robber",
        "discard",
        "trades",
        "dev_cards",
    }
)


class RealismV2HybridAgent(
    RealismV2PolicyAgent
):
    """
    Learned realism-v2 policy with selected decision
    families replaced by the canonical Search-v2 teacher.

    This is an evaluation/diagnostic agent only.
    """

    def __init__(
        self,
        strategy,
        model,
        *,
        interventions=(),
        deterministic=True,
        seed=None,
    ):
        super().__init__(
            strategy,
            model=model,
            deterministic=deterministic,
            seed=seed,
        )

        interventions = frozenset(
            interventions
        )

        unknown = (
            interventions
            - VALID_INTERVENTIONS
        )

        if unknown:
            raise ValueError(
                "Unknown realism-v2 intervention(s): "
                f"{sorted(unknown)}"
            )

        self.interventions = (
            interventions
        )

        # Match the frozen realism-v2 teacher.
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

        self.teacher_takeovers = Counter()

    def _uses_teacher(
        self,
        family,
    ):
        return (
            family
            in self.interventions
        )

    def _record_takeover(
        self,
        family,
    ):
        self.teacher_takeovers[
            family
        ] += 1

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
        if self._uses_teacher(
            "ordinary"
        ):
            self._record_takeover(
                "ordinary"
            )

            return (
                self.teacher.choose_action(
                    board,
                    players,
                    player,
                    inventory,
                    dev_deck=dev_deck,
                    bank=bank,
                    inventories=inventories,
                )
            )

        return super().choose_action(
            board,
            players,
            player,
            inventory,
            dev_deck=dev_deck,
            bank=bank,
            inventories=inventories,
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
        if self._uses_teacher(
            "robber"
        ):
            self._record_takeover(
                "robber"
            )

            return (
                self.teacher
                .choose_robber_tile(
                    board,
                    players,
                    inventories,
                    player,
                    bank=bank,
                    dev_deck=dev_deck,
                )
            )

        return (
            super()
            .choose_robber_tile(
                board,
                players,
                inventories,
                player,
                bank=bank,
                dev_deck=dev_deck,
            )
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
        if self._uses_teacher(
            "robber"
        ):
            self._record_takeover(
                "robber"
            )

            return (
                self.teacher
                .choose_robber_victim(
                    board,
                    players,
                    inventories,
                    player,
                    bank=bank,
                    dev_deck=dev_deck,
                )
            )

        return (
            super()
            .choose_robber_victim(
                board,
                players,
                inventories,
                player,
                bank=bank,
                dev_deck=dev_deck,
            )
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
        if self._uses_teacher(
            "discard"
        ):
            self._record_takeover(
                "discard"
            )

            return (
                self.teacher
                .choose_discards_with_context(
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

        return (
            super()
            .choose_discards_with_context(
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
        if self._uses_teacher(
            "trades"
        ):
            self._record_takeover(
                "trades"
            )

            return (
                self.teacher
                .propose_player_trade(
                    board,
                    players,
                    player,
                    inventories,
                    excluded_recipients=(
                        excluded_recipients
                    ),
                    agents=agents,
                    bank=bank,
                    dev_deck=dev_deck,
                )
            )

        return (
            super()
            .propose_player_trade(
                board,
                players,
                player,
                inventories,
                excluded_recipients=(
                    excluded_recipients
                ),
                agents=agents,
                bank=bank,
                dev_deck=dev_deck,
            )
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
    ):
        if self._uses_teacher(
            "trades"
        ):
            self._record_takeover(
                "trades"
            )

            return (
                self.teacher
                .evaluate_player_trade(
                    board,
                    players,
                    player,
                    inventories,
                    offer,
                    bank=bank,
                    dev_deck=dev_deck,
                )
            )

        return (
            super()
            .evaluate_player_trade(
                board,
                players,
                player,
                inventories,
                offer,
                bank=bank,
                dev_deck=dev_deck,
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
        if self._uses_teacher(
            "trades"
        ):
            self._record_takeover(
                "trades"
            )

            return (
                self.teacher
                .counter_player_trade(
                    board,
                    players,
                    player,
                    inventories,
                    offer,
                    attempted_offers=(
                        attempted_offers
                    ),
                    bank=bank,
                    dev_deck=dev_deck,
                )
            )

        return (
            super()
            .counter_player_trade(
                board,
                players,
                player,
                inventories,
                offer,
                attempted_offers=(
                    attempted_offers
                ),
                bank=bank,
                dev_deck=dev_deck,
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
        if self._uses_teacher(
            "dev_cards"
        ):
            self._record_takeover(
                "dev_cards"
            )

            # The teacher's Monopoly/YOP/Road-Building
            # parameter choices are established by its
            # dev-card-play search, so delegate this
            # complete decision path together.
            return (
                self.teacher
                .choose_dev_card_play(
                    board,
                    players,
                    player,
                    inventories,
                    phase,
                )
            )

        return super().choose_dev_card_play(
            board,
            players,
            player,
            inventories,
            phase,
            dev_deck=dev_deck,
            bank=bank,
        )

    def choose_monopoly_resource(
        self,
        board,
        players,
        inventories,
        player,
        suggested_resource=None,
    ):
        if self._uses_teacher(
            "dev_cards"
        ):
            return (
                self.teacher
                .choose_monopoly_resource(
                    board,
                    players,
                    inventories,
                    player,
                    suggested_resource=(
                        suggested_resource
                    ),
                )
            )

        return (
            super()
            .choose_monopoly_resource(
                board,
                players,
                inventories,
                player,
                suggested_resource=(
                    suggested_resource
                ),
            )
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
        if self._uses_teacher(
            "dev_cards"
        ):
            return (
                self.teacher
                .choose_year_of_plenty_resources(
                    board,
                    players,
                    inventories,
                    player,
                    bank=bank,
                    suggested_resources=(
                        suggested_resources
                    ),
                )
            )

        return (
            super()
            .choose_year_of_plenty_resources(
                board,
                players,
                inventories,
                player,
                bank=bank,
                suggested_resources=(
                    suggested_resources
                ),
            )
        )

    def choose_road_building_edges(
        self,
        board,
        players,
        inventories,
        player,
        suggested_edges=None,
    ):
        if self._uses_teacher(
            "dev_cards"
        ):
            return (
                self.teacher
                .choose_road_building_edges(
                    board,
                    players,
                    inventories,
                    player,
                    suggested_edges=(
                        suggested_edges
                    ),
                )
            )

        return (
            super()
            .choose_road_building_edges(
                board,
                players,
                inventories,
                player,
                suggested_edges=(
                    suggested_edges
                ),
            )
        )
