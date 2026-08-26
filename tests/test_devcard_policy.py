from catanlab.devcard_policy import (
    choose_dev_card_play,
)
from catanlab.devcards import DevCardType
from catanlab.economy import PlayerInventory
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


def test_knight_prioritized_near_largest_army_win():
    player = PlayerState(
        player_id=0,
        settlements=list(
            range(8)
        ),
        knights_played=2,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
        knights_played=1,
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    decision = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
    )

    assert (
        decision.card
        == DevCardType.KNIGHT
    )


def test_monopoly_targets_strongest_public_production_signal():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.graph import HexCoord

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    opponent_1 = PlayerState(
        player_id=1,
        settlements=[0],
    )

    opponent_2 = PlayerState(
        player_id=2,
        settlements=[1],
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Only the public hand size matters now.
    inventories[1].add(
        Resource.WOOD,
        4,
    )

    inventories[2].add(
        Resource.BRICK,
        3,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
            ),
            Tile(
                id=1,
                coord=HexCoord(1, 0),
                resource=Resource.WOOD,
                number=3,
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
        ],
        edges=[],
    )

    decision = choose_dev_card_play(
        player,
        [
            player,
            opponent_1,
            opponent_2,
        ],
        inventories,
        board=board,
    )

    assert (
        decision.card
        == DevCardType.MONOPOLY
    )

    assert (
        decision.resource
        == Resource.ORE
    )


def test_road_building_prioritized_near_longest_road():
    player = PlayerState(
        player_id=0,
        settlements=list(
            range(8)
        ),
        roads=[
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),
        ],
        dev_cards=[
            DevCardType.ROAD_BUILDING.value,
        ],
    )

    inventories = [
        PlayerInventory()
    ]

    decision = choose_dev_card_play(
        player,
        [player],
        inventories,
    )

    assert (
        decision.card
        == DevCardType.ROAD_BUILDING
    )


def test_agent_holds_low_value_knight():
    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    inventories = [
        PlayerInventory()
    ]

    decision = choose_dev_card_play(
        player,
        [player],
        inventories,
    )

    assert decision.card is None


def test_monopoly_can_be_held_pre_roll_due_to_seven_risk():
    from catanlab.devcard_policy import (
        DevCardPhase,
    )

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Player already has six cards.
    inventories[0].add(
        Resource.WOOD,
        6,
    )

    # Monopoly would take four ore and push
    # the player to ten cards before rolling.
    inventories[1].add(
        Resource.ORE,
        4,
    )

    decision = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
        phase=DevCardPhase.PRE_ROLL,
    )

    assert decision.card is None


def test_same_monopoly_is_attractive_post_roll():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.devcard_policy import (
        DevCardPhase,
    )
    from catanlab.graph import HexCoord

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
        settlements=[0],
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WOOD,
        6,
    )

    # Exact composition is hidden; total card count
    # is what the heuristic uses.
    inventories[1].add(
        Resource.WOOD,
        4,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
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

    decision = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
        phase=DevCardPhase.POST_ROLL,
        board=board,
    )

    assert (
        decision.card
        == DevCardType.MONOPOLY
    )

    assert (
        decision.resource
        == Resource.ORE
    )


def test_pre_roll_and_post_roll_can_make_different_decisions():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.devcard_policy import (
        DevCardPhase,
    )
    from catanlab.graph import HexCoord

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
        settlements=[0],
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories[0].add(
        Resource.WHEAT,
        6,
    )

    inventories[1].add(
        Resource.BRICK,
        4,
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
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

    pre_roll = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
        phase=DevCardPhase.PRE_ROLL,
        board=board,
    )

    post_roll = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
        phase=DevCardPhase.POST_ROLL,
        board=board,
    )

    assert pre_roll.card is None

    assert (
        post_roll.card
        == DevCardType.MONOPOLY
    )

    assert (
        post_roll.resource
        == Resource.ORE
    )


def test_policy_ignores_new_knight():
    player = PlayerState(
        player_id=0,
        knights_played=2,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
        new_dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
        knights_played=1,
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    decision = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
    )

    assert decision.card is None


def test_policy_can_use_old_copy_when_new_copy_also_exists():
    player = PlayerState(
        player_id=0,
        knights_played=2,
        dev_cards=[
            DevCardType.KNIGHT.value,
            DevCardType.KNIGHT.value,
        ],
        new_dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
        knights_played=1,
    )

    inventories = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    decision = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories,
    )

    assert (
        decision.card
        == DevCardType.KNIGHT
    )


def test_full_ows_will_start_largest_army_progress():
    from catanlab.devcard_policy import (
        choose_dev_card_play,
    )
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    inventories = [
        PlayerInventory()
    ]

    decision = choose_dev_card_play(
        player,
        [player],
        inventories,
        strategy=StrategyType.FULL_OWS,
    )

    assert (
        decision.card
        == DevCardType.KNIGHT
    )


def test_road_strategy_can_still_hold_early_knight():
    from catanlab.devcard_policy import (
        choose_dev_card_play,
    )
    from catanlab.devcards import DevCardType
    from catanlab.economy import PlayerInventory
    from catanlab.simulation import PlayerState
    from catanlab.strategies import StrategyType

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.KNIGHT.value,
        ],
    )

    inventories = [
        PlayerInventory()
    ]

    decision = choose_dev_card_play(
        player,
        [player],
        inventories,
        strategy=StrategyType.ROAD_BUILDING,
    )

    assert decision.card is None


def test_monopoly_decision_ignores_hidden_resource_identity():
    from catanlab.board import (
        Board,
        Tile,
        Vertex,
    )
    from catanlab.graph import HexCoord

    player = PlayerState(
        player_id=0,
        dev_cards=[
            DevCardType.MONOPOLY.value,
        ],
    )

    opponent = PlayerState(
        player_id=1,
        settlements=[0],
    )

    board = Board(
        tiles=[
            Tile(
                id=0,
                coord=HexCoord(0, 0),
                resource=Resource.ORE,
                number=6,
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

    inventories_a = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    inventories_b = [
        PlayerInventory(),
        PlayerInventory(),
    ]

    # Same public hand size, completely different
    # hidden resource identities.
    inventories_a[1].add(
        Resource.ORE,
        4,
    )

    inventories_b[1].add(
        Resource.WOOD,
        4,
    )

    decision_a = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories_a,
        board=board,
    )

    decision_b = choose_dev_card_play(
        player,
        [
            player,
            opponent,
        ],
        inventories_b,
        board=board,
    )

    assert decision_a == decision_b

    assert (
        decision_a.card
        == DevCardType.MONOPOLY
    )

    assert (
        decision_a.resource
        == Resource.ORE
    )
