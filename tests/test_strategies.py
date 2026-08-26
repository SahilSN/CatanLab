from catanlab.resources import Resource
from catanlab.strategies import (
    STRATEGY_PROFILES,
    StrategyType,
)


def test_all_strategy_profiles_exist():
    assert set(
        STRATEGY_PROFILES
    ) == set(
        StrategyType
    )


def test_full_ows_prefers_ore_to_wood():
    profile = STRATEGY_PROFILES[
        StrategyType.FULL_OWS
    ]

    assert (
        profile.resource_weights[
            Resource.ORE
        ]
        >
        profile.resource_weights[
            Resource.WOOD
        ]
    )


def test_road_strategy_prefers_wood_and_brick():
    profile = STRATEGY_PROFILES[
        StrategyType.ROAD_BUILDING
    ]

    assert (
        profile.resource_weights[
            Resource.WOOD
        ]
        >
        profile.resource_weights[
            Resource.ORE
        ]
    )

    assert (
        profile.resource_weights[
            Resource.BRICK
        ]
        >
        profile.resource_weights[
            Resource.ORE
        ]
    )


def test_five_resource_weights_are_equal():
    profile = STRATEGY_PROFILES[
        StrategyType.FIVE_RESOURCE
    ]

    assert len(
        set(
            profile.resource_weights.values()
        )
    ) == 1
