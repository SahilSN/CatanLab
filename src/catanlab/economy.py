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



STANDARD_RESOURCE_SUPPLY = 19


@dataclass
class ResourceBank:
    """
    Finite standard Catan resource-card bank.

    A four-player base game begins with 19 cards of
    each producing resource.
    """

    resources: Counter[Resource] = field(
        default_factory=lambda: Counter(
            {
                resource:
                    STANDARD_RESOURCE_SUPPLY
                for resource
                in PRODUCING_RESOURCES
            }
        )
    )

    def count(
        self,
        resource: Resource,
    ) -> int:
        if resource == Resource.DESERT:
            return 0

        return self.resources[
            resource
        ]

    def total(self) -> int:
        return sum(
            self.resources[
                resource
            ]
            for resource
            in PRODUCING_RESOURCES
        )

    def can_supply(
        self,
        resource: Resource,
        amount: int = 1,
    ) -> bool:
        if amount < 0:
            raise ValueError(
                "amount must be nonnegative"
            )

        if resource == Resource.DESERT:
            return False

        return (
            self.count(resource)
            >= amount
        )

    def remove(
        self,
        resource: Resource,
        amount: int = 1,
    ) -> None:
        """
        Take cards from the bank.
        """

        if amount < 0:
            raise ValueError(
                "amount must be nonnegative"
            )

        if resource == Resource.DESERT:
            raise ValueError(
                "Desert is not a resource card."
            )

        if not self.can_supply(
            resource,
            amount,
        ):
            raise ValueError(
                "Bank does not contain enough "
                f"{resource.value}: "
                f"requested={amount}, "
                f"available={self.count(resource)}"
            )

        self.resources[
            resource
        ] -= amount

    def add(
        self,
        resource: Resource,
        amount: int = 1,
    ) -> None:
        """
        Return cards to the bank.
        """

        if amount < 0:
            raise ValueError(
                "amount must be nonnegative"
            )

        if resource == Resource.DESERT:
            raise ValueError(
                "Desert is not a resource card."
            )

        self.resources[
            resource
        ] += amount


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
        bank: ResourceBank | None = None,
    ) -> None:
        """
        Pay the resource cost for a build.

        If a bank is supplied, spent resource cards
        are returned to that bank.

        Raises ValueError if the player cannot
        afford the build.
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

            if bank is not None:
                bank.add(
                    resource,
                    amount,
                )


def produce_for_roll(
    board: Board,
    players: list[PlayerState],
    inventories: list[PlayerInventory],
    roll: int,
    bank: ResourceBank | None = None,
) -> None:
    """
    Distribute resources for one dice roll.

    If bank is provided, production is resolved with
    standard finite-bank scarcity semantics:

    - first collect all claims,
    - aggregate demand per resource type,
    - if the bank cannot satisfy all claims for one
      resource, nobody receives that resource,
    - other resource types may still be paid normally.

    If bank is None, preserve legacy unlimited-bank
    behavior.
    """

    if len(players) != len(inventories):
        raise ValueError(
            "players and inventories must have "
            "the same length"
        )

    if roll == 7:
        return

    claims: dict[
        Resource,
        Counter[int],
    ] = {
        resource: Counter()
        for resource in PRODUCING_RESOURCES
    }

    for player in players:
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

                claims[
                    tile.resource
                ][
                    player.player_id
                ] += 1

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

                claims[
                    tile.resource
                ][
                    player.player_id
                ] += 2

    for resource, resource_claims in (
        claims.items()
    ):
        total_claim = sum(
            resource_claims.values()
        )

        if total_claim <= 0:
            continue

        if (
            bank is not None
            and not bank.can_supply(
                resource,
                total_claim,
            )
        ):
            # Standard scarcity rule:
            # nobody receives this resource.
            continue

        if bank is not None:
            bank.remove(
                resource,
                total_claim,
            )

        for player_id, amount in (
            resource_claims.items()
        ):
            inventories[
                player_id
            ].add(
                resource,
                amount,
            )


def resource_conservation_totals(
    bank: ResourceBank,
    inventories: list[PlayerInventory],
) -> dict[Resource, int]:
    """
    Return the total number of each producing
    resource across the bank and all player hands.
    """

    return {
        resource: (
            bank.count(resource)
            + sum(
                inventory.count(resource)
                for inventory in inventories
            )
        )
        for resource in PRODUCING_RESOURCES
    }


def validate_resource_conservation(
    bank: ResourceBank,
    inventories: list[PlayerInventory],
) -> None:
    """
    Raise ValueError if any resource does not total
    the standard 19 cards across bank + players.
    """

    totals = resource_conservation_totals(
        bank,
        inventories,
    )

    invalid = {
        resource: total
        for resource, total in totals.items()
        if total != STANDARD_RESOURCE_SUPPLY
    }

    if invalid:
        details = ", ".join(
            f"{resource.value}={total}"
            for resource, total in invalid.items()
        )

        raise ValueError(
            "Resource conservation violated: "
            + details
        )

