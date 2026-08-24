from enum import Enum


class Resource(str, Enum):
    WOOD = "wood"
    BRICK = "brick"
    SHEEP = "sheep"
    WHEAT = "wheat"
    ORE = "ore"
    DESERT = "desert"


STANDARD_RESOURCE_COUNTS = {
    Resource.WOOD: 4,
    Resource.SHEEP: 4,
    Resource.WHEAT: 4,
    Resource.BRICK: 3,
    Resource.ORE: 3,
    Resource.DESERT: 1,
}
