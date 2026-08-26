from collections import Counter

from catanlab.board import (
    build_random_board,
)
from catanlab.ports import (
    STANDARD_PORT_RESOURCES,
    best_maritime_ratio,
    coastal_edges,
    player_ports,
    standard_port_edges,
)
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


def test_standard_board_has_30_coastal_edges():
    board = build_random_board(
        seed=42
    )

    assert len(
        coastal_edges(board)
    ) == 30


def test_standard_board_has_nine_ports():
    board = build_random_board(
        seed=42
    )

    assert len(
        board.ports
    ) == 9


def test_standard_port_distribution():
    board = build_random_board(
        seed=42
    )

    actual = Counter(
        port.resource
        for port in board.ports
    )

    expected = Counter(
        STANDARD_PORT_RESOURCES
    )

    assert actual == expected


def test_ports_do_not_share_vertices():
    board = build_random_board(
        seed=42
    )

    vertices = []

    for port in board.ports:
        vertices.extend(
            [
                port.vertex_a,
                port.vertex_b,
            ]
        )

    assert len(vertices) == 18
    assert len(set(vertices)) == 18


def test_standard_port_edges_are_coastal():
    board = build_random_board(
        seed=42
    )

    coast = set(
        coastal_edges(board)
    )

    for edge in standard_port_edges(
        board
    ):
        assert edge in coast


def test_player_controls_port_from_settlement():
    board = build_random_board(
        seed=42
    )

    port = board.ports[0]

    player = PlayerState(
        player_id=0,
        settlements=[
            port.vertex_a,
        ],
    )

    owned = player_ports(
        board,
        player,
    )

    assert port in owned


def test_player_controls_port_from_city():
    board = build_random_board(
        seed=42
    )

    port = board.ports[0]

    player = PlayerState(
        player_id=0,
        cities=[
            port.vertex_b,
        ],
    )

    assert port in player_ports(
        board,
        player,
    )


def test_no_port_has_four_to_one_ratio():
    board = build_random_board(
        seed=42
    )

    player = PlayerState(
        player_id=0
    )

    assert best_maritime_ratio(
        board,
        player,
        Resource.ORE,
    ) == 4


def test_generic_port_gives_three_to_one():
    board = build_random_board(
        seed=42
    )

    generic = next(
        port
        for port in board.ports
        if port.resource is None
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            generic.vertex_a,
        ],
    )

    assert best_maritime_ratio(
        board,
        player,
        Resource.ORE,
    ) == 3


def test_resource_port_gives_two_to_one():
    board = build_random_board(
        seed=42
    )

    ore_port = next(
        port
        for port in board.ports
        if port.resource == Resource.ORE
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            ore_port.vertex_a,
        ],
    )

    assert best_maritime_ratio(
        board,
        player,
        Resource.ORE,
    ) == 2


def test_resource_port_only_applies_to_matching_resource():
    board = build_random_board(
        seed=42
    )

    ore_port = next(
        port
        for port in board.ports
        if port.resource == Resource.ORE
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            ore_port.vertex_a,
        ],
    )

    assert best_maritime_ratio(
        board,
        player,
        Resource.WOOD,
    ) == 4


def test_default_maritime_trade_is_four_to_one():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )

    spent = maritime_trade(
        board,
        player,
        inventory,
        give=Resource.WOOD,
        receive=Resource.ORE,
    )

    assert spent == 4

    assert inventory.count(
        Resource.WOOD
    ) == 0

    assert inventory.count(
        Resource.ORE
    ) == 1


def test_generic_port_trade_is_three_to_one():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    port = next(
        port
        for port in board.ports
        if port.resource is None
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            port.vertex_a,
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.SHEEP,
        3,
    )

    spent = maritime_trade(
        board,
        player,
        inventory,
        give=Resource.SHEEP,
        receive=Resource.ORE,
    )

    assert spent == 3

    assert inventory.count(
        Resource.SHEEP
    ) == 0

    assert inventory.count(
        Resource.ORE
    ) == 1


def test_matching_port_trade_is_two_to_one():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    ore_port = next(
        port
        for port in board.ports
        if port.resource == Resource.ORE
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            ore_port.vertex_a,
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.ORE,
        2,
    )

    spent = maritime_trade(
        board,
        player,
        inventory,
        give=Resource.ORE,
        receive=Resource.WHEAT,
    )

    assert spent == 2

    assert inventory.count(
        Resource.ORE
    ) == 0

    assert inventory.count(
        Resource.WHEAT
    ) == 1


def test_wrong_resource_still_uses_four_to_one():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    ore_port = next(
        port
        for port in board.ports
        if port.resource == Resource.ORE
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            ore_port.vertex_a,
        ],
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )

    spent = maritime_trade(
        board,
        player,
        inventory,
        give=Resource.WOOD,
        receive=Resource.WHEAT,
    )

    assert spent == 4


def test_maritime_trade_rejects_insufficient_resources():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.BRICK,
        3,
    )

    try:
        maritime_trade(
            board,
            player,
            inventory,
            give=Resource.BRICK,
            receive=Resource.ORE,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected unaffordable trade to fail"
        )


def test_maritime_trade_requires_different_resources():
    from catanlab.economy import (
        PlayerInventory,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )

    try:
        maritime_trade(
            board,
            player,
            inventory,
            give=Resource.WOOD,
            receive=Resource.WOOD,
        )

    except ValueError:
        pass

    else:
        raise AssertionError(
            "Expected same-resource trade to fail"
        )


def test_ports_at_vertex():
    from catanlab.ports import (
        ports_at_vertex,
    )

    board = build_random_board(
        seed=42
    )

    port = board.ports[0]

    found = ports_at_vertex(
        board,
        port.vertex_a,
    )

    assert port in found


def test_matching_specific_port_beats_generic_port():
    """
    If a player controls both a generic 3:1 port
    and the matching specific 2:1 port, the better
    2:1 ratio must be used.
    """

    board = build_random_board(
        seed=42
    )

    generic = next(
        port
        for port in board.ports
        if port.resource is None
    )

    ore_port = next(
        port
        for port in board.ports
        if port.resource == Resource.ORE
    )

    player = PlayerState(
        player_id=0,
        settlements=[
            generic.vertex_a,
            ore_port.vertex_a,
        ],
    )

    assert best_maritime_ratio(
        board,
        player,
        Resource.ORE,
    ) == 2


def test_maritime_trade_with_finite_bank_conserves_resources():
    """
    A finite-bank maritime trade must return the
    offered cards to the bank and remove the
    received card from it.
    """

    from catanlab.economy import (
        PlayerInventory,
        ResourceBank,
    )
    from catanlab.ports import (
        maritime_trade,
    )

    board = build_random_board(
        seed=42
    )

    player = PlayerState(
        player_id=0
    )

    inventory = PlayerInventory()

    inventory.add(
        Resource.WOOD,
        4,
    )

    bank = ResourceBank()

    wood_before = bank.count(
        Resource.WOOD
    )
    ore_before = bank.count(
        Resource.ORE
    )

    maritime_trade(
        board,
        player,
        inventory,
        give=Resource.WOOD,
        receive=Resource.ORE,
        bank=bank,
    )

    assert inventory.count(
        Resource.WOOD
    ) == 0

    assert inventory.count(
        Resource.ORE
    ) == 1

    assert bank.count(
        Resource.WOOD
    ) == wood_before + 4

    assert bank.count(
        Resource.ORE
    ) == ore_before - 1
