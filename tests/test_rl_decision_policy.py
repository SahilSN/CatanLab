import pytest

from catanlab.rl_decision_policy import (
    LearnedDecisionRequest,
    choose_decision_value,
    validate_decision_action_id,
)
from catanlab.rl_special_actions import (
    CategoricalDecisionInput,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)


def make_request():
    return LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=(
            0.0,
            1.0,
            2.0,
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=(
                    "wood",
                    "brick",
                    "ore",
                ),
                legal_mask=(
                    True,
                    False,
                    True,
                ),
            )
        ),
    )


def test_learned_decision_request_exposes_action_space():
    request = make_request()

    assert request.action_dim == 3

    assert request.legal_mask == (
        True,
        False,
        True,
    )

    assert request.legal_action_ids == (
        0,
        2,
    )


def test_learned_decision_request_decodes_legal_action():
    request = make_request()

    assert request.decode(2) == "ore"


def test_learned_decision_request_rejects_illegal_action():
    request = make_request()

    with pytest.raises(
        ValueError,
        match="illegal",
    ):
        request.decode(1)


def test_learned_decision_request_rejects_out_of_range_action():
    request = make_request()

    with pytest.raises(
        ValueError,
        match="Invalid learned decision action ID",
    ):
        request.decode(3)


def test_learned_decision_request_requires_legal_choice():
    with pytest.raises(
        ValueError,
        match="at least one legal",
    ):
        LearnedDecisionRequest(
            decision_kind=(
                TeacherDecisionKind
                .TRADE_RESPONSE
            ),
            observation=(1.0,),
            decision_input=(
                CategoricalDecisionInput(
                    vocabulary=(
                        False,
                        True,
                    ),
                    legal_mask=(
                        False,
                        False,
                    ),
                )
            ),
        )


def test_validate_decision_action_id_returns_legal_id():
    request = make_request()

    assert (
        validate_decision_action_id(
            request,
            2,
        )
        == 2
    )


def test_validate_decision_action_id_rejects_masked_id():
    request = make_request()

    with pytest.raises(
        ValueError,
        match="illegal",
    ):
        validate_decision_action_id(
            request,
            1,
        )


def test_choose_decision_value_decodes_policy_choice():
    request = make_request()

    class Policy:
        def choose_decision(
            self,
            request,
        ):
            assert (
                request.decision_kind
                == TeacherDecisionKind
                .MONOPOLY_RESOURCE
            )

            return 2

    assert (
        choose_decision_value(
            Policy(),
            request,
        )
        == "ore"
    )


def test_choose_decision_value_rejects_buggy_policy():
    request = make_request()

    class Policy:
        def choose_decision(
            self,
            request,
        ):
            return 1

    with pytest.raises(
        ValueError,
        match="illegal",
    ):
        choose_decision_value(
            Policy(),
            request,
        )


def test_all_realism_v2_categorical_kinds_fit_request_contract():
    from catanlab.board import Board, Edge, Tile, Vertex
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.graph import HexCoord
    from catanlab.resources import Resource
    from catanlab.rl_decision_policy import (
        LearnedDecisionRequest,
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
    from catanlab.simulation import PlayerState
    from catanlab.trading import TradeOffer

    observation = (
        0.0,
        1.0,
        2.0,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=5,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.BRICK,
                number=6,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[0, 1],
            ),
            Vertex(
                id=2,
                position=(2.0, 0.0),
                adjacent_tiles=[1],
            ),
            Vertex(
                id=3,
                position=(0.0, 1.0),
                adjacent_tiles=[0],
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
            Edge(
                vertex_a=1,
                vertex_b=2,
            ),
        ],
        robber_tile_id=0,
    )

    players = [
        PlayerState(
            player_id=0,
            roads=[(0, 1)],
        ),
        PlayerState(
            player_id=1,
            settlements=[3],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WOOD,
        2,
    )
    inventories[0].add(
        Resource.BRICK,
        2,
    )
    inventories[0].add(
        Resource.ORE,
        1,
    )

    inventories[1].add(
        Resource.SHEEP,
        1,
    )

    bank = ResourceBank()

    incoming = TradeOffer(
        proposer_id=1,
        recipient_id=0,
        give=((Resource.ORE, 1),),
        receive=((Resource.WOOD, 1),),
    )

    decision_inputs = {
        TeacherDecisionKind.ROBBER_TILE: (
            robber_tile_decision_input(
                board
            )
        ),
        TeacherDecisionKind.ROBBER_VICTIM: (
            robber_victim_decision_input(
                board,
                players,
                inventories,
                players[0],
            )
        ),
        TeacherDecisionKind.DISCARD: (
            discard_decision_input(
                inventories[0],
                2,
            )
        ),
        TeacherDecisionKind.MONOPOLY_RESOURCE: (
            monopoly_resource_decision_input()
        ),
        TeacherDecisionKind.YEAR_OF_PLENTY: (
            year_of_plenty_decision_input(
                bank
            )
        ),
        TeacherDecisionKind.ROAD_BUILDING: (
            road_building_decision_input(
                board,
                players,
                players[0],
            )
        ),
        TeacherDecisionKind.TRADE_PROPOSAL: (
            trade_proposal_decision_input(
                players,
                players[0],
                inventories[0],
            )
        ),
        TeacherDecisionKind.TRADE_RESPONSE: (
            trade_response_decision_input(
                incoming,
                inventories,
            )
        ),
        TeacherDecisionKind.TRADE_COUNTER: (
            trade_counter_decision_input(
                players,
                players[0],
                inventories[0],
                incoming,
            )
        ),
    }

    expected = {
        TeacherDecisionKind.ROBBER_TILE,
        TeacherDecisionKind.ROBBER_VICTIM,
        TeacherDecisionKind.DISCARD,
        TeacherDecisionKind.MONOPOLY_RESOURCE,
        TeacherDecisionKind.YEAR_OF_PLENTY,
        TeacherDecisionKind.ROAD_BUILDING,
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_RESPONSE,
        TeacherDecisionKind.TRADE_COUNTER,
    }

    assert set(
        decision_inputs
    ) == expected

    for decision_kind, decision_input in (
        decision_inputs.items()
    ):
        request = LearnedDecisionRequest(
            decision_kind=decision_kind,
            observation=observation,
            decision_input=decision_input,
        )

        assert request.action_dim > 0
        assert request.legal_action_ids

        for action_id in (
            request.legal_action_ids
        ):
            value = request.decode(
                action_id
            )

            assert (
                decision_input.encode(
                    value
                )
                == action_id
            )


def test_fixed_head_policy_satisfies_learned_policy_contract():
    import torch

    from catanlab.rl_decision_policy import (
        TorchFixedHeadDecisionPolicy,
        choose_decision_value,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        monopoly_resource_decision_input,
    )

    model = RealismV2ActorCritic()

    # Force deterministic preference for ORE.
    with torch.no_grad():
        model.monopoly_resource_head.weight.zero_()

        model.monopoly_resource_head.bias.copy_(
            torch.tensor([
                0.0,
                1.0,
                2.0,
                3.0,
                10.0,
            ])
        )

    decision_input = (
        monopoly_resource_decision_input()
    )

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=decision_input,
    )

    policy = TorchFixedHeadDecisionPolicy(
        model,
        deterministic=True,
    )

    action_id = policy.choose_decision(
        request
    )

    assert action_id == 4

    assert (
        choose_decision_value(
            policy,
            request,
        )
        == Resource.ORE
    )


def test_fixed_head_policy_respects_legal_mask():
    import torch

    from catanlab.rl_decision_policy import (
        TorchFixedHeadDecisionPolicy,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_special_actions import (
        CategoricalDecisionInput,
    )

    model = RealismV2ActorCritic()

    with torch.no_grad():
        model.trade_response_head.weight.zero_()

        # Strongly prefer ACCEPT, but mask it.
        model.trade_response_head.bias.copy_(
            torch.tensor([
                0.0,
                100.0,
            ])
        )

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .TRADE_RESPONSE
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=(
                    False,
                    True,
                ),
                legal_mask=(
                    True,
                    False,
                ),
            )
        ),
    )

    policy = TorchFixedHeadDecisionPolicy(
        model,
        deterministic=True,
    )

    assert (
        policy.choose_decision(
            request
        )
        == 0
    )


def test_fixed_head_policy_rejects_dynamic_kind():
    from catanlab.rl_decision_policy import (
        TorchFixedHeadDecisionPolicy,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_special_actions import (
        CategoricalDecisionInput,
    )

    model = RealismV2ActorCritic()

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .ROBBER_TILE
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=(0, 1),
                legal_mask=(True, True),
            )
        ),
    )

    policy = TorchFixedHeadDecisionPolicy(
        model,
    )

    with pytest.raises(
        ValueError,
        match="does not support dynamic",
    ):
        policy.choose_decision(
            request
        )


def test_fixed_head_policy_rejects_dimension_mismatch():
    from catanlab.rl_decision_policy import (
        TorchFixedHeadDecisionPolicy,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_special_actions import (
        CategoricalDecisionInput,
    )

    model = RealismV2ActorCritic()

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=(
                    "a",
                    "b",
                ),
                legal_mask=(
                    True,
                    True,
                ),
            )
        ),
    )

    policy = TorchFixedHeadDecisionPolicy(
        model,
    )

    with pytest.raises(
        ValueError,
        match="dimension does not match",
    ):
        policy.choose_decision(
            request
        )


def test_realism_v2_policy_routes_fixed_decision():
    import torch

    from catanlab.rl_decision_policy import (
        TorchRealismV2DecisionPolicy,
        choose_decision_value,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.resources import Resource
    from catanlab.rl_special_actions import (
        monopoly_resource_decision_input,
    )

    model = RealismV2ActorCritic()

    with torch.no_grad():
        model.monopoly_resource_head.weight.zero_()
        model.monopoly_resource_head.bias.copy_(
            torch.tensor(
                [
                    0.0,
                    1.0,
                    2.0,
                    9.0,
                    3.0,
                ]
            )
        )

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .MONOPOLY_RESOURCE
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=(
            monopoly_resource_decision_input()
        ),
    )

    policy = TorchRealismV2DecisionPolicy(
        model,
        deterministic=True,
    )

    assert (
        choose_decision_value(
            policy,
            request,
        )
        == Resource.WHEAT
    )


def test_realism_v2_policy_routes_dynamic_decision():
    import torch

    from catanlab.rl_decision_policy import (
        TorchRealismV2DecisionPolicy,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_special_actions import (
        CategoricalDecisionInput,
    )

    model = RealismV2ActorCritic(
        hidden_dim=8,
    )

    key = (
        TeacherDecisionKind
        .DISCARD
        .value
    )

    projector = (
        model.dynamic_candidate_projectors[
            key
        ]
    )

    scorer = (
        model.dynamic_candidate_scorers[
            key
        ]
    )

    with torch.no_grad():
        # Candidate embedding dimension 0 becomes the
        # candidate's WOOD discard count.
        projector.weight.zero_()
        projector.bias.zero_()
        projector.weight[0, 0] = 1.0

        # Ignore observation features. Copy candidate
        # embedding dimension 0 through the scoring MLP.
        scorer[0].weight.zero_()
        scorer[0].bias.zero_()

        # Joint feature layout is:
        # [observation H | candidate H]
        scorer[0].weight[
            0,
            model.hidden_dim,
        ] = 1.0

        scorer[2].weight.zero_()
        scorer[2].bias.zero_()
        scorer[2].weight[0, 0] = 1.0

    decision_input = (
        CategoricalDecisionInput(
            vocabulary=(
                (
                    0,
                    0,
                    0,
                    0,
                    2,
                ),
                (
                    1,
                    1,
                    0,
                    0,
                    0,
                ),
                (
                    2,
                    0,
                    0,
                    0,
                    0,
                ),
            ),
            legal_mask=(
                True,
                True,
                True,
            ),
        )
    )

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=decision_input,
    )

    policy = TorchRealismV2DecisionPolicy(
        model,
        deterministic=True,
    )

    # Highest WOOD count is candidate 2.
    assert (
        policy.choose_decision(
            request
        )
        == 2
    )

    assert request.decode(2) == (
        2,
        0,
        0,
        0,
        0,
    )


def test_realism_v2_dynamic_policy_respects_legal_mask():
    import torch

    from catanlab.rl_decision_policy import (
        TorchRealismV2DecisionPolicy,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_special_actions import (
        CategoricalDecisionInput,
    )

    model = RealismV2ActorCritic(
        hidden_dim=8,
    )

    key = (
        TeacherDecisionKind
        .DISCARD
        .value
    )

    projector = (
        model.dynamic_candidate_projectors[
            key
        ]
    )

    scorer = (
        model.dynamic_candidate_scorers[
            key
        ]
    )

    with torch.no_grad():
        projector.weight.zero_()
        projector.bias.zero_()
        projector.weight[0, 0] = 1.0

        scorer[0].weight.zero_()
        scorer[0].bias.zero_()
        scorer[0].weight[
            0,
            model.hidden_dim,
        ] = 1.0

        scorer[2].weight.zero_()
        scorer[2].bias.zero_()
        scorer[2].weight[0, 0] = 1.0

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind.DISCARD
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=(
                    (
                        0,
                        0,
                        0,
                        0,
                        2,
                    ),
                    (
                        1,
                        1,
                        0,
                        0,
                        0,
                    ),
                    (
                        2,
                        0,
                        0,
                        0,
                        0,
                    ),
                ),
                legal_mask=(
                    True,
                    True,
                    False,
                ),
            )
        ),
    )

    policy = TorchRealismV2DecisionPolicy(
        model,
        deterministic=True,
    )

    # Candidate 2 has the highest raw score but is masked,
    # so candidate 1 must win.
    assert (
        policy.choose_decision(
            request
        )
        == 1
    )


def test_realism_v2_policy_supports_all_special_kinds():
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )

    model = RealismV2ActorCritic()

    special_kinds = {
        TeacherDecisionKind.ROBBER_TILE,
        TeacherDecisionKind.ROBBER_VICTIM,
        TeacherDecisionKind.DISCARD,
        TeacherDecisionKind.MONOPOLY_RESOURCE,
        TeacherDecisionKind.YEAR_OF_PLENTY,
        TeacherDecisionKind.ROAD_BUILDING,
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_RESPONSE,
        TeacherDecisionKind.TRADE_COUNTER,
    }

    supported = {
        decision_kind
        for decision_kind in special_kinds
        if (
            model.is_fixed_decision_kind(
                decision_kind
            )
            or model.is_dynamic_decision_kind(
                decision_kind
            )
        )
    }

    assert supported == special_kinds


def test_realism_v2_policy_rejects_ordinary_action_kind():
    from catanlab.rl_decision_policy import (
        TorchRealismV2DecisionPolicy,
    )
    from catanlab.rl_model import (
        RealismV2ActorCritic,
    )
    from catanlab.rl_special_actions import (
        CategoricalDecisionInput,
    )

    model = RealismV2ActorCritic()

    request = LearnedDecisionRequest(
        decision_kind=(
            TeacherDecisionKind
            .ORDINARY_ACTION
        ),
        observation=tuple(
            0.0
            for _ in range(1138)
        ),
        decision_input=(
            CategoricalDecisionInput(
                vocabulary=("pass",),
                legal_mask=(True,),
            )
        ),
    )

    policy = TorchRealismV2DecisionPolicy(
        model,
    )

    with pytest.raises(
        ValueError,
        match="does not support",
    ):
        policy.choose_decision(
            request
        )
