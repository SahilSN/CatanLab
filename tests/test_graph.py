from catanlab.graph import (
    HexCoord,
    STANDARD_HEX_COORDS,
)


def test_standard_board_has_19_hexes():
    assert len(STANDARD_HEX_COORDS) == 19


def test_standard_hex_coordinates_are_unique():
    assert len(
        set(STANDARD_HEX_COORDS)
    ) == 19


def test_standard_board_contains_center():
    assert HexCoord(
        0,
        0,
    ) in STANDARD_HEX_COORDS


def test_hex_has_six_corners():
    from catanlab.graph import hex_corners

    corners = hex_corners(
        HexCoord(0, 0)
    )

    assert len(corners) == 6
    assert len(set(corners)) == 6


def test_adjacent_hexes():
    from catanlab.graph import (
        are_hexes_adjacent,
    )

    assert are_hexes_adjacent(
        HexCoord(0, 0),
        HexCoord(1, 0),
    )

    assert are_hexes_adjacent(
        HexCoord(0, 0),
        HexCoord(1, -1),
    )


def test_non_adjacent_hexes():
    from catanlab.graph import (
        are_hexes_adjacent,
    )

    assert not are_hexes_adjacent(
        HexCoord(0, 0),
        HexCoord(2, 0),
    )
