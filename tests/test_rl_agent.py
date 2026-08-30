from catanlab.board import build_random_board
from catanlab.devcards import (
    build_dev_card_deck,
)
from catanlab.economy import (
    PlayerInventory,
    ResourceBank,
)
from catanlab.rl_agent import (
    NeuralPolicyAgent,
    RandomMaskedAgent,
)
from catanlab.simulation import PlayerState
from catanlab.strategies import StrategyType
from catanlab.turns import ActionType


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

    deck = build_dev_card_deck(
        seed=456
    )

    bank = ResourceBank()

    return (
        board,
        players,
        inventories,
        deck,
        bank,
    )


def test_random_masked_agent_can_choose_pass():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_fixture()

    agent = RandomMaskedAgent(
        StrategyType.FIVE_RESOURCE,
        seed=1,
    )

    action = agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    # With no pieces/resources, PASS should be the only
    # ordinary legal action.
    assert (
        action.action_type
        == ActionType.PASS
    )

    assert agent.last_action_id == 0
    assert agent.last_legal_count == 1


def test_random_masked_agent_is_seed_reproducible():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_fixture()

    agent_a = RandomMaskedAgent(
        StrategyType.FIVE_RESOURCE,
        seed=99,
    )

    agent_b = RandomMaskedAgent(
        StrategyType.FIVE_RESOURCE,
        seed=99,
    )

    action_a = agent_a.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    action_b = agent_b.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    assert action_a == action_b
    assert (
        agent_a.last_action_id
        == agent_b.last_action_id
    )


def test_random_masked_agent_tracks_selection():
    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_fixture()

    agent = RandomMaskedAgent(
        StrategyType.FIVE_RESOURCE,
        seed=7,
    )

    assert agent.actions_selected == 0

    agent.choose_action(
        board,
        players,
        players[0],
        inventories[0],
        dev_deck=deck,
        bank=bank,
    )

    assert agent.actions_selected == 1
    assert agent.last_action_id is not None
    assert agent.last_legal_count >= 1


def test_neural_agent_receives_full_inventories():
    import torch

    from catanlab.rl_model import (
        CatanActorCritic,
    )

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_fixture()

    model = CatanActorCritic()

    agent = NeuralPolicyAgent(
        StrategyType.FIVE_RESOURCE,
        model=model,
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
    assert agent.actions_selected == 1
    assert agent.last_action_id is not None
    assert agent.last_legal_count >= 1


def test_neural_agent_runs_through_real_turn_path():
    from catanlab.rl_model import (
        CatanActorCritic,
    )
    from catanlab.turns import run_turn

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_fixture()

    agents = [
        NeuralPolicyAgent(
            StrategyType.FIVE_RESOURCE,
            model=CatanActorCritic(),
            deterministic=True,
            seed=1,
        )
    ]

    # Fill remaining seats with ordinary agents.
    from catanlab.turns import (
        AdaptiveStrategyAgent,
    )

    agents.extend(
        AdaptiveStrategyAgent(
            StrategyType.FIVE_RESOURCE
        )
        for _ in range(3)
    )

    result = run_turn(
        board,
        players,
        inventories,
        agents,
        player_id=0,
        roll=6,
        dev_deck=deck,
        bank=bank,
    )

    assert result is not None
    assert agents[0].actions_selected >= 1


def test_factorized_model_output_dimensions():
    import torch

    from catanlab.rl_model import (
        FactorizedCatanActorCritic,
    )

    model = FactorizedCatanActorCritic()

    observation = torch.zeros(
        3,
        1138,
    )

    logits, value = model(
        observation
    )

    assert logits.shape == (
        3,
        202,
    )

    assert value.shape == (
        3,
    )


def test_factorized_model_works_with_neural_agent():
    from catanlab.rl_model import (
        FactorizedCatanActorCritic,
    )

    (
        board,
        players,
        inventories,
        deck,
        bank,
    ) = make_fixture()

    model = FactorizedCatanActorCritic()

    agent = NeuralPolicyAgent(
        StrategyType.FIVE_RESOURCE,
        model=model,
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
    assert agent.actions_selected == 1
    assert agent.last_action_id is not None
