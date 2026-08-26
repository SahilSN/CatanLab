from collections import Counter
from dataclasses import dataclass

from catanlab.economy import PlayerInventory
from catanlab.resources import Resource


PRODUCING_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


ResourceBundle = tuple[
    tuple[Resource, int],
    ...,
]


def make_bundle(
    *pairs: tuple[Resource, int],
) -> ResourceBundle:
    """
    Create a canonical immutable resource bundle.

    Example:

        make_bundle(
            (Resource.WOOD, 1),
            (Resource.SHEEP, 1),
        )

    represents wood + sheep.
    """

    combined: Counter[
        Resource
    ] = Counter()

    for resource, amount in pairs:
        if amount <= 0:
            raise ValueError(
                "Trade bundle amounts must be positive."
            )

        if resource == Resource.DESERT:
            raise ValueError(
                "Desert cannot be traded."
            )

        combined[
            resource
        ] += amount

    return tuple(
        sorted(
            combined.items(),
            key=lambda item: item[0].value,
        )
    )


def bundle_counter(
    bundle: ResourceBundle,
) -> Counter[Resource]:
    return Counter(
        {
            resource: amount
            for resource, amount in bundle
        }
    )


def bundle_size(
    bundle: ResourceBundle,
) -> int:
    return sum(
        amount
        for _, amount in bundle
    )


@dataclass(frozen=True)
class TradeOffer:
    """
    Domestic Catan trade.

    `give` is what the proposer gives.
    `receive` is what the proposer requests.

    Both sides may contain multiple resource types
    and multiple cards.

    Examples:

        wood -> ore

        wood + sheep -> ore

        wood + brick -> ore + wheat
    """

    proposer_id: int
    recipient_id: int

    give: ResourceBundle
    receive: ResourceBundle


def validate_bundle(
    bundle: ResourceBundle,
) -> bool:
    if not bundle:
        return False

    seen: set[
        Resource
    ] = set()

    for resource, amount in bundle:
        if resource == Resource.DESERT:
            return False

        if amount <= 0:
            return False

        if resource in seen:
            # Bundles should already be canonical.
            return False

        seen.add(
            resource
        )

    return True


def inventory_has_bundle(
    inventory: PlayerInventory,
    bundle: ResourceBundle,
) -> bool:
    return all(
        inventory.count(
            resource
        )
        >= amount
        for resource, amount in bundle
    )


def validate_trade_offer(
    offer: TradeOffer,
    inventories: list[PlayerInventory],
) -> bool:
    """
    Validate only hard trade legality.

    Strategic willingness is handled by the agents.
    """

    if (
        offer.proposer_id
        == offer.recipient_id
    ):
        return False

    if not (
        0
        <= offer.proposer_id
        < len(inventories)
    ):
        return False

    if not (
        0
        <= offer.recipient_id
        < len(inventories)
    ):
        return False

    if not validate_bundle(
        offer.give
    ):
        return False

    if not validate_bundle(
        offer.receive
    ):
        return False

    give_resources = {
        resource
        for resource, _ in offer.give
    }

    receive_resources = {
        resource
        for resource, _ in offer.receive
    }

    # Exchanging the same resource in both
    # directions is strategically redundant and can
    # always be reduced to a simpler trade.
    if (
        give_resources
        & receive_resources
    ):
        return False

    proposer_inventory = inventories[
        offer.proposer_id
    ]

    recipient_inventory = inventories[
        offer.recipient_id
    ]

    if not inventory_has_bundle(
        proposer_inventory,
        offer.give,
    ):
        return False

    if not inventory_has_bundle(
        recipient_inventory,
        offer.receive,
    ):
        return False

    return True


def _remove_bundle(
    inventory: PlayerInventory,
    bundle: ResourceBundle,
) -> None:
    for resource, amount in bundle:
        inventory.remove(
            resource,
            amount,
        )


def _add_bundle(
    inventory: PlayerInventory,
    bundle: ResourceBundle,
) -> None:
    for resource, amount in bundle:
        inventory.add(
            resource,
            amount,
        )


def execute_player_trade(
    offer: TradeOffer,
    inventories: list[PlayerInventory],
) -> None:
    """
    Execute a legal domestic trade atomically.
    """

    if not validate_trade_offer(
        offer,
        inventories,
    ):
        raise ValueError(
            "Illegal player trade."
        )

    proposer_inventory = inventories[
        offer.proposer_id
    ]

    recipient_inventory = inventories[
        offer.recipient_id
    ]

    _remove_bundle(
        proposer_inventory,
        offer.give,
    )

    _remove_bundle(
        recipient_inventory,
        offer.receive,
    )

    _add_bundle(
        recipient_inventory,
        offer.give,
    )

    _add_bundle(
        proposer_inventory,
        offer.receive,
    )


def reverse_trade_offer(
    offer: TradeOffer,
) -> TradeOffer:
    """
    Convert an offer into the opposite perspective.

    Useful for counteroffers.
    """

    return TradeOffer(
        proposer_id=offer.recipient_id,
        recipient_id=offer.proposer_id,
        give=offer.receive,
        receive=offer.give,
    )


def generate_trade_bundles(
    inventory: PlayerInventory,
    max_cards: int = 4,
    max_types: int = 3,
) -> list[ResourceBundle]:
    """
    Generate legal non-empty resource bundles that
    could be offered from an inventory.

    Search is deliberately bounded because this
    function is used during negotiation.
    """

    from itertools import product

    resources = PRODUCING_RESOURCES

    limits = [
        min(
            inventory.count(
                resource
            ),
            max_cards,
        )
        for resource in resources
    ]

    bundles = []

    ranges = [
        range(
            limit + 1
        )
        for limit in limits
    ]

    for amounts in product(
        *ranges
    ):
        total_cards = sum(
            amounts
        )

        if (
            total_cards == 0
            or total_cards > max_cards
        ):
            continue

        distinct_types = sum(
            amount > 0
            for amount in amounts
        )

        if distinct_types > max_types:
            continue

        pairs = [
            (
                resource,
                amount,
            )
            for resource, amount
            in zip(
                resources,
                amounts,
            )
            if amount > 0
        ]

        bundles.append(
            make_bundle(
                *pairs
            )
        )

    bundles.sort(
        key=lambda bundle: (
            bundle_size(
                bundle
            ),
            tuple(
                (
                    resource.value,
                    amount,
                )
                for resource, amount
                in bundle
            ),
        )
    )

    return bundles
