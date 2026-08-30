from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from catanlab.rl_candidate_features import (
    dynamic_candidate_feature_dim,
    is_dynamic_decision_kind,
)
from catanlab.rl_teacher import (
    TeacherDecisionKind,
    TeacherV2Example,
)


TEACHER_V2_DATASET_VERSION = 1


def validate_teacher_v2_example(
    example: TeacherV2Example,
) -> None:
    if not example.observation:
        raise ValueError(
            "Teacher example observation cannot be empty."
        )

    if example.player_id < 0:
        raise ValueError(
            "Teacher example player_id cannot be negative."
        )

    if not isinstance(
        example.label,
        int,
    ):
        raise ValueError(
            "Teacher example label must be an integer "
            "categorical action ID."
        )

    if example.label < 0:
        raise ValueError(
            "Teacher example label cannot be negative."
        )

    if example.legal_mask is None:
        raise ValueError(
            "Categorical realism-v2 teacher example "
            "requires a legal mask."
        )

    if not example.legal_mask:
        raise ValueError(
            "Teacher example legal mask cannot be empty."
        )

    if (
        example.label
        >= len(example.legal_mask)
    ):
        raise ValueError(
            "Teacher example label is outside its "
            "legal-mask dimension."
        )

    if not example.legal_mask[
        example.label
    ]:
        raise ValueError(
            "Teacher example label is marked illegal."
        )

    if not any(
        example.legal_mask
    ):
        raise ValueError(
            "Teacher example must contain at least one "
            "legal categorical action."
        )

    if is_dynamic_decision_kind(
        example.decision_kind
    ):
        if example.candidate_features is None:
            raise ValueError(
                "Dynamic teacher example requires "
                "candidate features."
            )

        if (
            len(example.candidate_features)
            != len(example.legal_mask)
        ):
            raise ValueError(
                "Dynamic teacher candidate count must "
                "match legal-mask dimension."
            )

        expected_dim = (
            dynamic_candidate_feature_dim(
                example.decision_kind
            )
        )

        for candidate in (
            example.candidate_features
        ):
            if (
                len(candidate)
                != expected_dim
            ):
                raise ValueError(
                    "Dynamic teacher candidate feature "
                    "width does not match decision kind: "
                    f"kind={example.decision_kind.value}, "
                    f"expected={expected_dim}, "
                    f"actual={len(candidate)}"
                )

    elif (
        example.candidate_features
        is not None
    ):
        raise ValueError(
            "Fixed-size teacher example must not store "
            "dynamic candidate features."
        )


def teacher_v2_example_to_record(
    example: TeacherV2Example,
) -> dict:
    validate_teacher_v2_example(
        example
    )

    return {
        "version": TEACHER_V2_DATASET_VERSION,
        "decision_kind": (
            example.decision_kind.value
        ),
        "observation": list(
            example.observation
        ),
        "player_id": example.player_id,
        "label": example.label,
        "legal_mask": list(
            example.legal_mask
        ),
        "candidate_features": (
            None
            if example.candidate_features is None
            else [
                list(candidate)
                for candidate in (
                    example.candidate_features
                )
            ]
        ),
    }


def teacher_v2_example_from_record(
    record: dict,
) -> TeacherV2Example:
    if (
        record.get("version")
        != TEACHER_V2_DATASET_VERSION
    ):
        raise ValueError(
            "Unsupported teacher-v2 dataset version: "
            f"{record.get('version')!r}"
        )

    try:
        decision_kind = TeacherDecisionKind(
            record["decision_kind"]
        )
    except (
        KeyError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Invalid teacher-v2 decision kind."
        ) from exc

    try:
        example = TeacherV2Example(
            decision_kind=decision_kind,
            observation=tuple(
                float(value)
                for value in record[
                    "observation"
                ]
            ),
            player_id=int(
                record["player_id"]
            ),
            label=int(
                record["label"]
            ),
            legal_mask=tuple(
                bool(value)
                for value in record[
                    "legal_mask"
                ]
            ),
            candidate_features=(
                None
                if record.get(
                    "candidate_features"
                )
                is None
                else tuple(
                    tuple(
                        float(value)
                        for value in candidate
                    )
                    for candidate in record[
                        "candidate_features"
                    ]
                )
            ),
        )
    except (
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        raise ValueError(
            "Malformed teacher-v2 dataset record."
        ) from exc

    validate_teacher_v2_example(
        example
    )

    return example


def save_teacher_v2_jsonl(
    path,
    examples,
) -> int:
    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    count = 0

    with path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        for example in examples:
            record = (
                teacher_v2_example_to_record(
                    example
                )
            )

            handle.write(
                json.dumps(
                    record,
                    separators=(",", ":"),
                )
            )
            handle.write("\n")

            count += 1

    return count


def load_teacher_v2_jsonl(
    path,
) -> list[TeacherV2Example]:
    path = Path(path)

    examples = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line_number, line in enumerate(
            handle,
            start=1,
        ):
            if not line.strip():
                continue

            try:
                record = json.loads(
                    line
                )
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid JSON in teacher-v2 "
                    f"dataset at line {line_number}."
                ) from exc

            try:
                example = (
                    teacher_v2_example_from_record(
                        record
                    )
                )
            except ValueError as exc:
                raise ValueError(
                    "Invalid teacher-v2 example at "
                    f"line {line_number}: {exc}"
                ) from exc

            examples.append(
                example
            )

    return examples
