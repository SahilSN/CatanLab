from catanlab.board import (
    Board,
    Tile,
    Vertex,
)
from catanlab.economy import (
    PlayerInventory,
    produce_for_roll,
)
from catanlab.graph import HexCoord
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


def test_inventory_adds_resource():
    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD
    )

    assert inventory.count(
        Resource.WOOD
    ) == 1

    assert inventory.total() == 1


def test_matching_roll_produces_resource():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=6,
    )

    assert inventories[0].count(
        Resource.WOOD
    ) == 1


def test_nonmatching_roll_produces_nothing():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WHEAT,
                number=9,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=8,
    )

    assert inventories[0].total() == 0


def test_seven_produces_nothing():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=7,
    )

    assert inventories[0].total() == 0


def test_multiple_settlements_can_produce():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.WHEAT,
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
                position=(2.0, 0.0),
                adjacent_tiles=[1],
            ),
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0, 1],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=6,
    )

    assert inventories[0].count(
        Resource.WOOD
    ) == 1

    assert inventories[0].count(
        Resource.WHEAT
    ) == 1

    assert inventories[0].total() == 2


def test_can_afford_road():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD
    )

    inventory.add(
        Resource.BRICK
    )

    assert inventory.can_afford(
        BuildType.ROAD
    )


def test_cannot_afford_road():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD
    )

    assert not inventory.can_afford(
        BuildType.ROAD
    )


def test_spend_road():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        amount=2,
    )

    inventory.add(
        Resource.BRICK,
        amount=2,
    )

    inventory.spend(
        BuildType.ROAD
    )

    assert inventory.count(
        Resource.WOOD
    ) == 1

    assert inventory.count(
        Resource.BRICK
    ) == 1


def test_spend_settlement():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    for resource in (
        Resource.WOOD,
        Resource.BRICK,
        Resource.SHEEP,
        Resource.WHEAT,
    ):
        inventory.add(
            resource
        )

    assert inventory.can_afford(
        BuildType.SETTLEMENT
    )

    inventory.spend(
        BuildType.SETTLEMENT
    )

    assert inventory.total() == 0


def test_city_cost():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    inventory.add(
        Resource.WHEAT,
        amount=2,
    )

    inventory.add(
        Resource.ORE,
        amount=3,
    )

    assert inventory.can_afford(
        BuildType.CITY
    )

    inventory.spend(
        BuildType.CITY
    )

    assert inventory.total() == 0


def test_dev_card_cost():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP
    )

    inventory.add(
        Resource.WHEAT
    )

    inventory.add(
        Resource.ORE
    )

    assert inventory.can_afford(
        BuildType.DEV_CARD
    )


def test_spend_unaffordable_build_raises():
    from catanlab.economy import BuildType

    inventory = PlayerInventory()

    try:
        inventory.spend(
            BuildType.CITY
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected unaffordable build to fail"
        )


def test_negative_inventory_add_rejected():
    inventory = PlayerInventory()

    try:
        inventory.add(
            Resource.WOOD,
            amount=-1,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected negative amount to fail"
        )


def test_city_produces_two_resources():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=8,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
    )

    players = [
        PlayerState(
            player_id=0,
            cities=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=8,
    )

    assert inventories[0].count(
        Resource.ORE
    ) == 2


def test_robber_blocks_settlement_production():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.WOOD,
                number=6,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
        robber_tile_id=0,
    )

    players = [
        PlayerState(
            player_id=0,
            settlements=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=6,
    )

    assert inventories[0].total() == 0


def test_robber_blocks_city_production():
    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=8,
            )
        ],
        vertices=[
            Vertex(
                id=0,
                position=(0.0, 0.0),
                adjacent_tiles=[0],
            )
        ],
        edges=[],
        robber_tile_id=0,
    )

    players = [
        PlayerState(
            player_id=0,
            cities=[0],
        )
    ]

    inventories = [
        PlayerInventory()
    ]

    produce_for_roll(
        board,
        players,
        inventories,
        roll=8,
    )

    assert inventories[0].total() == 0


def test_inventory_remove_resource():
    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        2,
    )

    inventory.remove(
        Resource.WOOD
    )

    assert inventory.count(
        Resource.WOOD
    ) == 1


def test_inventory_cannot_remove_too_many():
    inventory = PlayerInventory()

    inventory.add(
        Resource.BRICK
    )

    try:
        inventory.remove(
            Resource.BRICK,
            2,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected excessive removal to fail"
        )
