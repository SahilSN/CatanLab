import pytest

from catanlab.economy import (
    PRODUCING_RESOURCES,
    STANDARD_RESOURCE_SUPPLY,
    ResourceBank,
)
from catanlab.resources import Resource


def test_standard_bank_starts_with_19_of_each_resource():
    bank = ResourceBank()

    for resource in PRODUCING_RESOURCES:
        assert (
            bank.count(resource)
            == STANDARD_RESOURCE_SUPPLY
        )

    assert bank.total() == 95


def test_bank_remove_reduces_supply():
    bank = ResourceBank()

    bank.remove(
        Resource.ORE,
        3,
    )

    assert (
        bank.count(Resource.ORE)
        == 16
    )


def test_bank_add_returns_resource():
    bank = ResourceBank()

    bank.remove(
        Resource.WOOD,
        4,
    )

    bank.add(
        Resource.WOOD,
        2,
    )

    assert (
        bank.count(Resource.WOOD)
        == 17
    )


def test_bank_cannot_supply_more_than_available():
    bank = ResourceBank()

    bank.remove(
        Resource.BRICK,
        18,
    )

    assert bank.can_supply(
        Resource.BRICK,
        1,
    )

    assert not bank.can_supply(
        Resource.BRICK,
        2,
    )


def test_bank_rejects_overdraw():
    bank = ResourceBank()

    with pytest.raises(
        ValueError
    ):
        bank.remove(
            Resource.SHEEP,
            20,
        )


def test_bank_rejects_desert():
    bank = ResourceBank()

    assert (
        bank.count(
            Resource.DESERT
        )
        == 0
    )

    with pytest.raises(
        ValueError
    ):
        bank.remove(
            Resource.DESERT,
            1,
        )

    with pytest.raises(
        ValueError
    ):
        bank.add(
            Resource.DESERT,
            1,
        )


def test_bank_rejects_negative_amounts():
    bank = ResourceBank()

    with pytest.raises(
        ValueError
    ):
        bank.remove(
            Resource.WHEAT,
            -1,
        )

    with pytest.raises(
        ValueError
    ):
        bank.add(
            Resource.WHEAT,
            -1,
        )

    with pytest.raises(
        ValueError
    ):
        bank.can_supply(
            Resource.WHEAT,
            -1,
        )


def test_bank_aware_production_pays_all_claims_when_supply_is_sufficient():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
        produce_for_roll,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.WOOD,
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
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
            cities=[1],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    bank = ResourceBank()

    produce_for_roll(
        board,
        players,
        inventories,
        roll=6,
        bank=bank,
    )

    assert inventories[0].count(
        Resource.WOOD
    ) == 1

    assert inventories[1].count(
        Resource.WOOD
    ) == 2

    assert bank.count(
        Resource.WOOD
    ) == 16


def test_bank_scarcity_denies_entire_resource_type():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
        produce_for_roll,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.WOOD,
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
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
            cities=[1],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    bank = ResourceBank()

    bank.remove(
        Resource.WOOD,
        17,
    )

    assert bank.count(
        Resource.WOOD
    ) == 2

    produce_for_roll(
        board,
        players,
        inventories,
        roll=6,
        bank=bank,
    )

    # Total wood demand is 3 but the bank has only 2,
    # so neither player receives wood.
    assert inventories[0].count(
        Resource.WOOD
    ) == 0

    assert inventories[1].count(
        Resource.WOOD
    ) == 0

    assert bank.count(
        Resource.WOOD
    ) == 2


def test_scarcity_is_resolved_independently_by_resource_type():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
        produce_for_roll,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.WOOD,
                number=6,
            ),
            Tile(
                id=1,
                coord=(1, 0),
                resource=Resource.ORE,
                number=6,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0, 1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
        PlayerState(
            player_id=1,
            cities=[1],
        ),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    bank = ResourceBank()

    bank.remove(
        Resource.WOOD,
        17,
    )

    produce_for_roll(
        board,
        players,
        inventories,
        roll=6,
        bank=bank,
    )

    # Wood demand is 3 and cannot be satisfied.
    assert inventories[0].count(
        Resource.WOOD
    ) == 0
    assert inventories[1].count(
        Resource.WOOD
    ) == 0

    # Ore demand is only 1 and should still be paid.
    assert inventories[0].count(
        Resource.ORE
    ) == 1

    assert bank.count(
        Resource.ORE
    ) == 18


def test_legacy_production_without_bank_still_works():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
        produce_for_roll,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=(0, 0),
                resource=Resource.SHEEP,
                number=8,
            ),
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        ),
    ]

    inventories = [
        PlayerInventory(),
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=8,
    )

    assert inventories[0].count(
        Resource.SHEEP
    ) == 1


def test_build_cost_returns_resources_to_bank():
    from catanlab.economy import (
        BuildType,
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource

    inventory = PlayerInventory()
    bank = ResourceBank()

    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.BRICK,
        1,
    )

    # These cards conceptually came from the bank.
    bank.remove(
        Resource.WOOD,
        1,
    )
    bank.remove(
        Resource.BRICK,
        1,
    )

    inventory.spend(
        BuildType.ROAD,
        bank=bank,
    )

    assert (
        inventory.count(
            Resource.WOOD
        )
        == 0
    )
    assert (
        inventory.count(
            Resource.BRICK
        )
        == 0
    )

    assert (
        bank.count(
            Resource.WOOD
        )
        == STANDARD_RESOURCE_SUPPLY
    )
    assert (
        bank.count(
            Resource.BRICK
        )
        == STANDARD_RESOURCE_SUPPLY
    )


def test_spend_without_bank_preserves_legacy_behavior():
    from catanlab.economy import (
        BuildType,
        PlayerInventory,
    )
    from catanlab.resources import Resource

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.BRICK,
        1,
    )

    inventory.spend(
        BuildType.ROAD
    )

    assert inventory.total() == 0


def test_execute_paid_road_returns_cost_to_bank():
    from catanlab.board import (
        Board,
        Edge,
        Vertex,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        execute_action,
    )

    board = Board(
        tiles=[],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                neighbors=[1],
            ),
            Vertex(
                id=1,
                position=(1.0, 0.0),
                neighbors=[0],
            ),
        ],
        edges=[
            Edge(
                vertex_a=0,
                vertex_b=1,
            ),
        ],
    )

    player = PlayerState(
        player_id=0,
        settlements=[0],
    )

    players = [player]

    inventory = PlayerInventory()
    bank = ResourceBank()

    # Transfer a road's resource cost from the
    # bank to the player's hand.
    bank.remove(
        Resource.WOOD,
        1,
    )
    bank.remove(
        Resource.BRICK,
        1,
    )

    inventory.add(
        Resource.WOOD,
        1,
    )
    inventory.add(
        Resource.BRICK,
        1,
    )

    execute_action(
        board,
        players,
        player,
        inventory,
        TurnAction(
            action_type=ActionType.BUILD_ROAD,
            edge=(0, 1),
        ),
        bank=bank,
    )

    assert player.roads == [
        (0, 1)
    ]

    assert inventory.total() == 0

    assert (
        bank.count(Resource.WOOD)
        == STANDARD_RESOURCE_SUPPLY
    )

    assert (
        bank.count(Resource.BRICK)
        == STANDARD_RESOURCE_SUPPLY
    )


def test_roll_seven_discards_return_cards_to_bank():
    from catanlab.board import Board
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState
    from catanlab.turns import (
        ActionType,
        TurnAction,
        TurnAgent,
        run_turn,
    )

    board = Board(
        tiles=[],
        vertices=[],
        edges=[],
    )

    players = [
        PlayerState(player_id=0),
        PlayerState(player_id=1),
    ]

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    bank = ResourceBank()

    # Move 8 wood cards from the bank into
    # player 1's hand.
    bank.remove(
        Resource.WOOD,
        8,
    )

    inventories[1].add(
        Resource.WOOD,
        8,
    )

    class PassAgent(TurnAgent):
        def choose_action(
            self,
            board,
            players,
            player,
            inventory,
            dev_deck=None,
        ):
            return TurnAction(
                action_type=ActionType.PASS
            )

    run_turn(
        board,
        players,
        inventories,
        [
            PassAgent(),
            PassAgent(),
        ],
        player_id=0,
        roll=7,
        bank=bank,
    )

    # Player 1 discards half: 4 cards.
    assert (
        inventories[1].count(
            Resource.WOOD
        )
        == 4
    )

    # Those 4 discarded cards return to the bank.
    assert (
        bank.count(
            Resource.WOOD
        )
        == 15
    )

    # Conservation still holds.
    assert (
        bank.count(
            Resource.WOOD
        )
        + inventories[0].count(
            Resource.WOOD
        )
        + inventories[1].count(
            Resource.WOOD
        )
        == STANDARD_RESOURCE_SUPPLY
    )


def test_maritime_trade_exchanges_with_bank():
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.ports import maritime_trade
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    # Use the normal test board constructor already
    # used elsewhere in the suite.
    from catanlab.board import build_random_board

    board = build_random_board(
        seed=123
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()
    bank = ResourceBank()

    # No port ownership -> ordinary 4:1 trade.
    bank.remove(
        Resource.WOOD,
        4,
    )
    inventory.add(
        Resource.WOOD,
        4,
    )

    starting_ore = bank.count(
        Resource.ORE
    )

    ratio = maritime_trade(
        board,
        player,
        inventory,
        give=Resource.WOOD,
        receive=Resource.ORE,
        bank=bank,
    )

    assert ratio == 4

    assert (
        inventory.count(
            Resource.WOOD
        )
        == 0
    )
    assert (
        inventory.count(
            Resource.ORE
        )
        == 1
    )

    assert (
        bank.count(
            Resource.WOOD
        )
        == STANDARD_RESOURCE_SUPPLY
    )
    assert (
        bank.count(
            Resource.ORE
        )
        == starting_ore - 1
    )


def test_maritime_trade_fails_atomically_when_bank_empty():
    import pytest

    from catanlab.board import build_random_board
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.ports import maritime_trade
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    board = build_random_board(
        seed=123
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()
    bank = ResourceBank()

    bank.remove(
        Resource.WOOD,
        4,
    )
    inventory.add(
        Resource.WOOD,
        4,
    )

    # Exhaust all ore from the bank.
    bank.remove(
        Resource.ORE,
        STANDARD_RESOURCE_SUPPLY,
    )

    wood_before = inventory.count(
        Resource.WOOD
    )
    ore_before = inventory.count(
        Resource.ORE
    )

    with pytest.raises(
        ValueError
    ):
        maritime_trade(
            board,
            player,
            inventory,
            give=Resource.WOOD,
            receive=Resource.ORE,
            bank=bank,
        )

    # Failed trade changed nothing.
    assert (
        inventory.count(
            Resource.WOOD
        )
        == wood_before
    )
    assert (
        inventory.count(
            Resource.ORE
        )
        == ore_before
    )

    assert (
        bank.count(
            Resource.WOOD
        )
        == STANDARD_RESOURCE_SUPPLY - 4
    )
    assert (
        bank.count(
            Resource.ORE
        )
        == 0
    )


def test_year_of_plenty_draws_resources_from_bank():
    from catanlab.devcards import (
        DevCardType,
        play_year_of_plenty,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.YEAR_OF_PLENTY.value
        ],
    )

    inventory = PlayerInventory()
    bank = ResourceBank()

    play_year_of_plenty(
        player,
        inventory,
        Resource.ORE,
        Resource.WHEAT,
        bank=bank,
    )

    assert (
        inventory.count(Resource.ORE)
        == 1
    )
    assert (
        inventory.count(Resource.WHEAT)
        == 1
    )

    assert (
        bank.count(Resource.ORE)
        == STANDARD_RESOURCE_SUPPLY - 1
    )
    assert (
        bank.count(Resource.WHEAT)
        == STANDARD_RESOURCE_SUPPLY - 1
    )

    assert (
        DevCardType.YEAR_OF_PLENTY.value
        not in player.dev_cards
    )


def test_year_of_plenty_can_take_same_resource_twice():
    from catanlab.devcards import (
        DevCardType,
        play_year_of_plenty,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.YEAR_OF_PLENTY.value
        ],
    )

    inventory = PlayerInventory()
    bank = ResourceBank()

    play_year_of_plenty(
        player,
        inventory,
        Resource.ORE,
        Resource.ORE,
        bank=bank,
    )

    assert (
        inventory.count(Resource.ORE)
        == 2
    )

    assert (
        bank.count(Resource.ORE)
        == STANDARD_RESOURCE_SUPPLY - 2
    )


def test_year_of_plenty_failure_is_atomic():
    import pytest

    from catanlab.devcards import (
        DevCardType,
        play_year_of_plenty,
    )
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.resources import Resource
    from catanlab.simulation import PlayerState

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.YEAR_OF_PLENTY.value
        ],
    )

    inventory = PlayerInventory()
    bank = ResourceBank()

    # Leave only one ore in the bank.
    bank.remove(
        Resource.ORE,
        STANDARD_RESOURCE_SUPPLY - 1,
    )

    with pytest.raises(ValueError):
        play_year_of_plenty(
            player,
            inventory,
            Resource.ORE,
            Resource.ORE,
            bank=bank,
        )

    # Neither the development card nor the one
    # remaining ore was consumed.
    assert (
        DevCardType.YEAR_OF_PLENTY.value
        in player.dev_cards
    )

    assert (
        inventory.count(Resource.ORE)
        == 0
    )

    assert (
        bank.count(Resource.ORE)
        == 1
    )


def test_resource_conservation_validator_accepts_valid_state():
    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
        validate_resource_conservation,
    )
    from catanlab.resources import Resource

    bank = ResourceBank()
    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    bank.remove(
        Resource.WOOD,
        3,
    )
    inventories[0].add(
        Resource.WOOD,
        2,
    )
    inventories[1].add(
        Resource.WOOD,
        1,
    )

    validate_resource_conservation(
        bank,
        inventories,
    )


def test_resource_conservation_validator_detects_created_card():
    import pytest

    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
        validate_resource_conservation,
    )
    from catanlab.resources import Resource

    bank = ResourceBank()
    inventories = [
        PlayerInventory(),
    ]

    # Deliberately manufacture a card.
    inventories[0].add(
        Resource.ORE,
        1,
    )

    with pytest.raises(
        ValueError,
        match="ore=20",
    ):
        validate_resource_conservation(
            bank,
            inventories,
        )


def test_complete_games_preserve_resource_conservation():
    from catanlab.economy import (
        validate_resource_conservation,
    )
    from catanlab.game import run_game
    from catanlab.strategies import StrategyType

    strategies = [
        StrategyType.FULL_OWS,
        StrategyType.HYBRID_OWS,
        StrategyType.ROAD_BUILDING,
        StrategyType.PORT,
    ]

    for seed in range(10):
        result = run_game(
            strategies,
            seed=seed,
            validate_conservation=True,
        )

        assert result.bank is not None

        validate_resource_conservation(
            result.bank,
            result.inventories,
        )
