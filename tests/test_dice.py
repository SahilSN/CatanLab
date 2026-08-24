from catanlab.dice import (
    production_probability,
    production_weight,
)


def test_common_number_weights():
    assert production_weight(6) == 5
    assert production_weight(8) == 5
    assert production_weight(5) == 4
    assert production_weight(9) == 4


def test_extreme_number_weights():
    assert production_weight(2) == 1
    assert production_weight(12) == 1


def test_seven_produces_nothing():
    assert production_weight(7) == 0


def test_desert_produces_nothing():
    assert production_weight(None) == 0


def test_probability():
    assert production_probability(6) == 5 / 36
