from catanlab.board import build_random_board
from catanlab.devcards import (
    build_dev_card_deck,
)
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.rl_teacher import (
    DAggerAgent,
    RecordingSearchAgent,
)
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType


def make_fixture():
    board = build_random_board(
        seed=123
    )

    players = [
        PlayerState(player_id=i)
        for i in range(4)
    ]

    inventories = [
        PlayerInventory()
        for _ in range(4)
    ]

    bank = ResourceBank()

    deck = build_dev_card_deck(
        seed=456
    )

    return (
        board,
        players,
        inventories,
        bank,
        deck,
    )


def test_recording_search_agent_records_example():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    assert agent.examples == []

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    assert action is not None
    assert len(agent.examples) == 1

    example = agent.examples[0]

    assert len(example.observation) == 1138
    assert len(example.legal_mask) == 202

    assert example.legal_mask[
        example.action_id
    ]

    assert example.player_id == 0


def test_recorded_example_is_snapshot():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    example = agent.examples[0]

    before = example.observation

    # Mutating the live state must not mutate the
    # already-recorded training example.
    from catanlab.resources import Resource

    inventories[0].add(
        Resource.ORE,
        4,
    )

    assert example.observation == before


def test_teacher_label_is_always_mask_legal():
    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = RecordingSearchAgent(
        StrategyType.FIVE_RESOURCE
    )

    for _ in range(5):
        agent.choose_action(
            board,
            players,
            players[0],
            inventories[0],
            dev_deck=deck,
            bank=bank,
            inventories=inventories,
        )

    assert len(agent.examples) == 5

    assert all(
        example.legal_mask[
            example.action_id
        ]
        for example in agent.examples
    )


def test_dagger_agent_records_teacher_label():
    from catanlab.rl_model import (
        CatanActorCritic,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = DAggerAgent(
        StrategyType.FIVE_RESOURCE,
        model=CatanActorCritic(),
        deterministic=True,
        seed=1,
    )

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    assert action is not None
    assert len(agent.examples) == 1

    example = agent.examples[0]

    assert len(example.observation) == 1138
    assert len(example.legal_mask) == 202

    assert example.legal_mask[
        example.action_id
    ]

    assert example.legal_mask[
        example.learner_action_id
    ]

    assert example.player_id == 0


def test_dagger_returned_action_is_learner_action():
    from catanlab.action_space import (
        turn_action_id,
    )
    from catanlab.rl_interface import (
        build_rl_decision_context,
    )
    from catanlab.rl_model import (
        CatanActorCritic,
    )

    (
        board,
        players,
        inventories,
        bank,
        deck,
    ) = make_fixture()

    agent = DAggerAgent(
        StrategyType.FIVE_RESOURCE,
        model=CatanActorCritic(),
        deterministic=True,
        seed=2,
    )

    context = build_rl_decision_context(
        board,
        players,
        inventories,
        0,
        bank,
        deck,
    )

    returned = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
        inventories=inventories,
    )

    returned_id = turn_action_id(
        returned,
        context.vocabulary,
    )

    assert returned_id == (
        agent.examples[
            0
        ].learner_action_id
    )


def test_teacher_v2_example_supports_flat_masked_decision():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
        TeacherV2Example,
    )

    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.ORDINARY_ACTION
        ),
        observation=(1.0, 2.0),
        player_id=0,
        label=17,
        legal_mask=(
            True,
            False,
            True,
        ),
    )

    assert (
        example.decision_kind
        == TeacherDecisionKind.ORDINARY_ACTION
    )
    assert example.label == 17
    assert example.has_legal_mask


def test_teacher_v2_example_supports_structured_decision():
    from catanlab.resources import Resource
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
        TeacherV2Example,
    )

    example = TeacherV2Example(
        decision_kind=(
            TeacherDecisionKind.YEAR_OF_PLENTY
        ),
        observation=(0.0,),
        player_id=2,
        label=(
            Resource.ORE,
            Resource.WHEAT,
        ),
    )

    assert example.label == (
        Resource.ORE,
        Resource.WHEAT,
    )
    assert example.legal_mask is None
    assert not example.has_legal_mask


def test_teacher_v2_decision_kinds_are_distinct():
    from catanlab.rl_teacher import (
        TeacherDecisionKind,
    )

    values = [
        kind.value
        for kind in TeacherDecisionKind
    ]

    assert len(values) == len(set(values))

    assert (
        TeacherDecisionKind.ROBBER_TILE
        != TeacherDecisionKind.ROBBER_VICTIM
    )

    assert (
        TeacherDecisionKind.TRADE_PROPOSAL
        != TeacherDecisionKind.TRADE_RESPONSE
    )
