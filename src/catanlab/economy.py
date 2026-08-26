from collections import Counter
from dataclasses import dataclass, field
from enum import Enum

from catanlab.board import Board
from catanlab.resources import Resource
from catanlab.simulation import PlayerState


PRODUCING_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


class BuildType(str, Enum):
    ROAD = "road"
    SETTLEMENT = "settlement"
    CITY = "city"
    DEV_CARD = "dev_card"


BUILD_COSTS: dict[
    BuildType,
    Counter[Resource],
] = {
    BuildType.ROAD: Counter(
        {
            Resource.WOOD: 1,
            Resource.BRICK: 1,
        }
    ),
    BuildType.SETTLEMENT: Counter(
        {
            Resource.WOOD: 1,
            Resource.BRICK: 1,
            Resource.SHEEP: 1,
            Resource.WHEAT: 1,
        }
    ),
    BuildType.CITY: Counter(
        {
            Resource.WHEAT: 2,
            Resource.ORE: 3,
        }
    ),
    BuildType.DEV_CARD: Counter(
        {
            Resource.SHEEP: 1,
            Resource.WHEAT: 1,
            Resource.ORE: 1,
        }
    ),
}


@dataclass
class PlayerInventory:
    resources: Counter[Resource] = field(
        default_factory=Counter
    )

    def add(
        self,
        resource: Resource,
        amount: int = 1,
    ) -> None:
        if resource == Resource.DESERT:
            return

        if amount < 0:
            raise ValueError(
                "amount must be nonnegative"
            )

        self.resources[resource] += amount

    def count(
        self,
        resource: Resource,
    ) -> int:
        return self.resources[resource]

    def total(self) -> int:
        return sum(
            self.resources.values()
        )

    def remove(
        self,
        resource: Resource,
        amount: int = 1,
    ) -> None:
        if amount < 0:
            raise ValueError(
                "amount must be nonnegative"
            )

        if self.count(resource) < amount:
            raise ValueError(
                f"Not enough {resource.value}"
            )

        self.resources[resource] -= amount

    def can_afford(
        self,
        build_type: BuildType,
    ) -> bool:
        """
        Return whether the inventory contains
        enough resources for a build.
        """

        cost = BUILD_COSTS[
            build_type
        ]

        return all(
            self.count(resource) >= amount
            for resource, amount in cost.items()
        )

    def spend(
        self,
        build_type: BuildType,
    ) -> None:
        """
        Pay the resource cost for a build.

        Raises ValueError if the player cannot
        afford it.
        """

        if not self.can_afford(
            build_type
        ):
            raise ValueError(
                f"Cannot afford "
                f"{build_type.value}"
            )

        cost = BUILD_COSTS[
            build_type
        ]

        for resource, amount in cost.items():
            self.resources[
                resource
            ] -= amount


def produce_for_roll(
    board: Board,
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    roll: int,
) -> None:
    """
    Distribute resources for one dice roll.

    Each settlement produces one resource from
    each adjacent tile matching the rolled number.
    """

    if len(players) != len(inventories):
        raise ValueError(
            "players and inventories must have "
            "the same length"
        )

    if roll == 7:
        return

    for player, inventory in zip(
        players,
        inventories,
    ):
        for vertex_id in player.settlements:
            vertex = board.vertices[
                vertex_id
            ]

            for tile_id in vertex.adjacent_tiles:
                tile = board.tiles[
                    tile_id
                ]

                if tile.number != roll:
                    continue

                if tile.id == board.robber_tile_id:
                    continue

                if tile.resource == Resource.DESERT:
                    continue

                inventory.add(
                    tile.resource,
                    amount=1,
                )

        for vertex_id in player.cities:
            vertex = board.vertices[
                vertex_id
            ]

            for tile_id in vertex.adjacent_tiles:
                tile = board.tiles[
                    tile_id
                ]

                if tile.number != roll:
                    continue

                if tile.id == board.robber_tile_id:
                    continue

                if tile.resource == Resource.DESERT:
                    continue

                inventory.add(
                    tile.resource,
                    amount=2,
                )
