from __future__ import annotations

from catanlab.resources import Resource
from catanlab.rl_teacher import (
    TeacherDecisionKind,
)


# These are feature-vector widths, not categorical action
# counts. Dynamic decisions may still have any number of
# candidates in a particular state.
DYNAMIC_CANDIDATE_FEATURE_DIMS = {
    TeacherDecisionKind.ROBBER_TILE: 1,
    TeacherDecisionKind.ROBBER_VICTIM: 1,
    TeacherDecisionKind.DISCARD: 5,
    TeacherDecisionKind.ROAD_BUILDING: 5,
    TeacherDecisionKind.TRADE_PROPOSAL: 13,
    TeacherDecisionKind.TRADE_COUNTER: 13,
}


_PRODUCING_RESOURCES = (
    Resource.WOOD,
    Resource.BRICK,
    Resource.SHEEP,
    Resource.WHEAT,
    Resource.ORE,
)


def is_dynamic_decision_kind(
    decision_kind: TeacherDecisionKind,
) -> bool:
    return (
        decision_kind
        in DYNAMIC_CANDIDATE_FEATURE_DIMS
    )


def dynamic_candidate_feature_dim(
    decision_kind: TeacherDecisionKind,
) -> int:
    try:
        return DYNAMIC_CANDIDATE_FEATURE_DIMS[
            decision_kind
        ]
    except KeyError as exc:
        raise ValueError(
            "Decision kind does not use dynamic "
            "candidate features: "
            f"{decision_kind!r}"
        ) from exc


def _scaled_nonnegative_id(
    value,
    *,
    scale: float,
    name: str,
) -> float:
    if not isinstance(value, int):
        raise ValueError(
            f"{name} must be an integer ID."
        )

    if value < 0:
        raise ValueError(
            f"{name} cannot be negative."
        )

    return float(value) / scale


def _resource_one_hot(
    resource,
) -> tuple[float, ...]:
    if resource not in _PRODUCING_RESOURCES:
        raise ValueError(
            "Dynamic trade candidate contains an "
            "unsupported resource: "
            f"{resource!r}"
        )

    return tuple(
        1.0 if resource == candidate else 0.0
        for candidate in _PRODUCING_RESOURCES
    )


def _encode_trade_candidate(
    value,
) -> tuple[float, ...]:
    """
    Encode Search-v2's dynamic domestic-trade candidate.

    Layout:
        [no_trade,
         proposer_id,
         recipient_id,
         give_resource one-hot (5),
         receive_resource one-hot (5)]

    The codec deliberately receives only the public
    TradeOffer value. No inventory or opponent hidden-hand
    information can enter this representation.
    """
    if value is None:
        return (
            1.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    from catanlab.trading import TradeOffer

    if not isinstance(value, TradeOffer):
        raise ValueError(
            "Dynamic trade candidate must be None or "
            "a TradeOffer."
        )

    # Search-v2 proposal/counter codecs are intentionally
    # restricted to one-resource, one-card-for-one-card
    # trades. Reject values outside that candidate family
    # rather than silently producing ambiguous features.
    if (
        len(value.give) != 1
        or len(value.receive) != 1
    ):
        raise ValueError(
            "Dynamic Search-v2 trade candidate must "
            "contain one give resource and one receive "
            "resource."
        )

    give_resource, give_amount = value.give[0]
    receive_resource, receive_amount = (
        value.receive[0]
    )

    if (
        give_amount != 1
        or receive_amount != 1
    ):
        raise ValueError(
            "Dynamic Search-v2 trade candidate must "
            "be 1-for-1."
        )

    return (
        0.0,
        _scaled_nonnegative_id(
            value.proposer_id,
            scale=4.0,
            name="Trade proposer ID",
        ),
        _scaled_nonnegative_id(
            value.recipient_id,
            scale=4.0,
            name="Trade recipient ID",
        ),
        *_resource_one_hot(
            give_resource
        ),
        *_resource_one_hot(
            receive_resource
        ),
    )


def encode_dynamic_candidate(
    decision_kind: TeacherDecisionKind,
    value,
) -> tuple[float, ...]:
    """
    Encode one simulator-facing dynamic categorical value
    into a fixed-width numeric candidate representation.

    The observation itself is encoded separately. These
    features describe only which candidate is being scored.
    """
    if (
        decision_kind
        == TeacherDecisionKind.ROBBER_TILE
    ):
        features = (
            _scaled_nonnegative_id(
                value,
                scale=19.0,
                name="Robber tile ID",
            ),
        )

    elif (
        decision_kind
        == TeacherDecisionKind.ROBBER_VICTIM
    ):
        features = (
            _scaled_nonnegative_id(
                value,
                scale=4.0,
                name="Robber victim player ID",
            ),
        )

    elif (
        decision_kind
        == TeacherDecisionKind.DISCARD
    ):
        if (
            not isinstance(value, tuple)
            or len(value) != 5
        ):
            raise ValueError(
                "Discard candidate must be a "
                "five-resource count tuple."
            )

        if any(
            not isinstance(count, int)
            or count < 0
            for count in value
        ):
            raise ValueError(
                "Discard counts must be nonnegative "
                "integers."
            )

        features = tuple(
            float(count)
            for count in value
        )

    elif (
        decision_kind
        == TeacherDecisionKind.ROAD_BUILDING
    ):
        if (
            not isinstance(value, tuple)
            or len(value) not in (1, 2)
        ):
            raise ValueError(
                "Road Building candidate must contain "
                "one or two edges."
            )

        def encode_edge(edge):
            if (
                not isinstance(edge, tuple)
                or len(edge) != 2
            ):
                raise ValueError(
                    "Road Building edges must be "
                    "two-vertex tuples."
                )

            return (
                _scaled_nonnegative_id(
                    edge[0],
                    scale=54.0,
                    name="Road vertex ID",
                ),
                _scaled_nonnegative_id(
                    edge[1],
                    scale=54.0,
                    name="Road vertex ID",
                ),
            )

        first_a, first_b = encode_edge(
            value[0]
        )

        if len(value) == 1:
            second_a = 0.0
            second_b = 0.0
            has_second = 0.0
        else:
            second_a, second_b = encode_edge(
                value[1]
            )
            has_second = 1.0

        features = (
            first_a,
            first_b,
            second_a,
            second_b,
            has_second,
        )

    elif decision_kind in {
        TeacherDecisionKind.TRADE_PROPOSAL,
        TeacherDecisionKind.TRADE_COUNTER,
    }:
        features = _encode_trade_candidate(
            value
        )

    else:
        raise ValueError(
            "Decision kind does not use dynamic "
            "candidate features: "
            f"{decision_kind!r}"
        )

    expected_dim = (
        dynamic_candidate_feature_dim(
            decision_kind
        )
    )

    if len(features) != expected_dim:
        raise RuntimeError(
            "Dynamic candidate encoder produced the "
            "wrong feature dimension: "
            f"kind={decision_kind.value}, "
            f"expected={expected_dim}, "
            f"actual={len(features)}"
        )

    return features


def encode_dynamic_decision_input(
    decision_kind: TeacherDecisionKind,
    decision_input,
) -> tuple[
    tuple[float, ...],
    ...,
]:
    """
    Encode every candidate in one CategoricalDecisionInput
    while preserving its categorical ordering exactly.
    """
    if not is_dynamic_decision_kind(
        decision_kind
    ):
        raise ValueError(
            "Decision kind does not use dynamic "
            "candidate features: "
            f"{decision_kind!r}"
        )

    return tuple(
        encode_dynamic_candidate(
            decision_kind,
            value,
        )
        for value in decision_input.vocabulary
    )
