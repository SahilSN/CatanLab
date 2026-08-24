from random import randint


DICE_WEIGHTS = {
    2: 1,
    3: 2,
    4: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 5,
    9: 4,
    10: 3,
    11: 2,
    12: 1,
}


def roll_dice() -> int:
    """Roll two six-sided dice."""
    return randint(1, 6) + randint(1, 6)


def production_weight(number: int | None) -> int:
    """
    Return the standard Catan production weight
    for a number token.

    Desert tiles use None and have weight 0.
    """
    if number is None:
        return 0

    if number == 7:
        return 0

    return DICE_WEIGHTS[number]


def production_probability(number: int | None) -> float:
    """Return the probability that a number is rolled."""
    return production_weight(number) / 36


STANDARD_NUMBER_TOKENS = [
    2,
    3,
    3,
    4,
    4,
    5,
    5,
    6,
    6,
    8,
    8,
    9,
    9,
    10,
    10,
    11,
    11,
    12,
]
