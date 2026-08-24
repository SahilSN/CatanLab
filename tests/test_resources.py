from catanlab.resources import Resource


def test_resource_values():
    assert Resource.WOOD.value == "wood"
    assert Resource.ORE.value == "ore"
    assert Resource.DESERT.value == "desert"
