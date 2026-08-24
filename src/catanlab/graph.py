from dataclasses import dataclass
from math import cos, pi, sin


@dataclass(frozen=True)
class HexCoord:
    q: int
    r: int


STANDARD_HEX_COORDS = [
    HexCoord(0, -2),
    HexCoord(1, -2),
    HexCoord(2, -2),

    HexCoord(-1, -1),
    HexCoord(0, -1),
    HexCoord(1, -1),
    HexCoord(2, -1),

    HexCoord(-2, 0),
    HexCoord(-1, 0),
    HexCoord(0, 0),
    HexCoord(1, 0),
    HexCoord(2, 0),

    HexCoord(-2, 1),
    HexCoord(-1, 1),
    HexCoord(0, 1),
    HexCoord(1, 1),

    HexCoord(-2, 2),
    HexCoord(-1, 2),
    HexCoord(0, 2),
]


def hex_center(
    coord: HexCoord,
) -> tuple[float, float]:
    """
    Convert axial coordinates to a pointy-top
    hex center in Cartesian coordinates.
    """
    x = 3 ** 0.5 * (
        coord.q + coord.r / 2
    )

    y = 1.5 * coord.r

    return x, y


def hex_corners(
    coord: HexCoord,
) -> list[tuple[float, float]]:
    """
    Return the six Cartesian corner coordinates
    of a pointy-top unit hex.
    """
    cx, cy = hex_center(coord)

    corners = []

    for i in range(6):
        angle = (
            pi / 180
        ) * (
            60 * i - 30
        )

        x = cx + cos(angle)
        y = cy + sin(angle)

        # Round so shared corners from neighboring
        # hexes deduplicate reliably.
        corners.append(
            (
                round(x, 8),
                round(y, 8),
            )
        )

    return corners


HEX_DIRECTIONS = (
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
)


def are_hexes_adjacent(
    a: HexCoord,
    b: HexCoord,
) -> bool:
    """Return whether two axial hexes share an edge."""

    dq = b.q - a.q
    dr = b.r - a.r

    return (
        dq,
        dr,
    ) in HEX_DIRECTIONS
