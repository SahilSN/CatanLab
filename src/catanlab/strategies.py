from dataclasses import dataclass
from enum import Enum

from catanlab.resources import Resource


class StrategyType(str, Enum):
    FULL_OWS = "full_ows"
    HYBRID_OWS = "hybrid_ows"
    ROAD_BUILDING = "road_building"
    ROADS_AND_CITIES = "roads_and_cities"
    FIVE_RESOURCE = "five_resource"
    PORT = "port"


@dataclass(frozen=True)
class StrategyProfile:
    strategy: StrategyType

    resource_weights: dict[
        Resource,
        float,
    ]

    diversity_weight: float

    description: str


STRATEGY_PROFILES = {
    StrategyType.FULL_OWS: StrategyProfile(
        strategy=StrategyType.FULL_OWS,
        resource_weights={
            Resource.ORE: 1.4,
            Resource.WHEAT: 1.4,
            Resource.SHEEP: 1.1,
            Resource.WOOD: 0.3,
            Resource.BRICK: 0.3,
        },
        diversity_weight=0.3,
        description=(
            "Ore-wheat-sheep strategy focused "
            "on cities and development cards."
        ),
    ),

    StrategyType.HYBRID_OWS: StrategyProfile(
        strategy=StrategyType.HYBRID_OWS,
        resource_weights={
            Resource.ORE: 1.3,
            Resource.WHEAT: 1.3,
            Resource.SHEEP: 1.0,
            Resource.WOOD: 0.7,
            Resource.BRICK: 0.7,
        },
        diversity_weight=0.8,
        description=(
            "OWS strategy retaining enough "
            "wood and brick for expansion."
        ),
    ),

    StrategyType.ROAD_BUILDING: StrategyProfile(
        strategy=StrategyType.ROAD_BUILDING,
        resource_weights={
            Resource.WOOD: 1.5,
            Resource.BRICK: 1.5,
            Resource.WHEAT: 0.7,
            Resource.SHEEP: 0.7,
            Resource.ORE: 0.7,
        },
        diversity_weight=0.5,
        description=(
            "Wood-brick expansion strategy "
            "focused on roads and settlements."
        ),
    ),

    StrategyType.ROADS_AND_CITIES: StrategyProfile(
        strategy=StrategyType.ROADS_AND_CITIES,
        resource_weights={
            Resource.ORE: 1.2,
            Resource.WHEAT: 1.2,
            Resource.WOOD: 1.0,
            Resource.BRICK: 1.0,
            Resource.SHEEP: 0.5,
        },
        diversity_weight=0.8,
        description=(
            "Hybrid expansion strategy combining "
            "roads, settlements, and cities."
        ),
    ),

    StrategyType.FIVE_RESOURCE: StrategyProfile(
        strategy=StrategyType.FIVE_RESOURCE,
        resource_weights={
            Resource.WOOD: 1.0,
            Resource.BRICK: 1.0,
            Resource.SHEEP: 1.0,
            Resource.WHEAT: 1.0,
            Resource.ORE: 1.0,
        },
        diversity_weight=1.5,
        description=(
            "Balanced strategy emphasizing "
            "access to all five resources."
        ),
    ),

    StrategyType.PORT: StrategyProfile(
        strategy=StrategyType.PORT,
        resource_weights={
            Resource.WOOD: 1.0,
            Resource.BRICK: 1.0,
            Resource.SHEEP: 1.0,
            Resource.WHEAT: 1.0,
            Resource.ORE: 1.0,
        },
        diversity_weight=0.2,
        description=(
            "Specialist strategy centered around "
            "high production and favorable ports."
        ),
    ),
}
