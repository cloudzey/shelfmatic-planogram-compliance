"""Compare a detected shelf structure with an expected planogram."""

from __future__ import annotations

from typing import Any


def parse_expected_facings(value: str) -> list[int]:
    """Parse comma-separated facing counts from top shelf to bottom."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Expected facing counts cannot be empty.")

    parts = [part.strip() for part in value.split(",")]

    if any(not part for part in parts):
        raise ValueError("Each shelf row must have a facing count.")

    try:
        counts = [int(part) for part in parts]
    except ValueError as error:
        raise ValueError("Facing counts must be whole numbers.") from error

    if any(count < 1 for count in counts):
        raise ValueError("Facing counts must be at least 1.")

    return counts


def compare_planogram(
    shelf_layout: dict[str, Any],
    expected_facings: list[int],
) -> dict[str, Any]:
    """Compare detected and expected facing counts row by row."""
    if not isinstance(shelf_layout, dict):
        raise ValueError("Shelf layout must be a mapping.")

    rows = shelf_layout.get("rows")

    if not isinstance(rows, list):
        raise ValueError("Shelf layout must contain a rows list.")

    if not expected_facings or any(
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 1
        for count in expected_facings
    ):
        raise ValueError(
            "Expected facings must contain positive integers."
        )

    detected_facings = []

    for row_index, row in enumerate(rows, start=1):
        if not isinstance(row, dict) or "detection_count" not in row:
            raise ValueError(
                f"Detected row {row_index} is missing detection_count."
            )

        count = row["detection_count"]

        if (
            not isinstance(count, int)
            or isinstance(count, bool)
            or count < 0
        ):
            raise ValueError(
                f"Detected row {row_index} has an invalid "
                "detection_count."
            )

        detected_facings.append(count)

    comparison_rows = []
    matched_total = 0
    missing_total = 0
    extra_total = 0
    comparison_slot_total = 0

    row_count = max(
        len(expected_facings),
        len(detected_facings),
    )

    for offset in range(row_count):
        expected = (
            expected_facings[offset]
            if offset < len(expected_facings)
            else 0
        )
        detected = (
            detected_facings[offset]
            if offset < len(detected_facings)
            else 0
        )

        matched = min(expected, detected)
        missing = max(expected - detected, 0)
        extra = max(detected - expected, 0)

        if expected == detected:
            status = "compliant"
        elif expected == 0:
            status = "unexpected_row"
        elif detected == 0:
            status = "missing_row"
        elif detected < expected:
            status = "missing_facings"
        else:
            status = "extra_facings"

        matched_total += matched
        missing_total += missing
        extra_total += extra
        comparison_slot_total += max(expected, detected)

        comparison_rows.append(
            {
                "row_index": offset + 1,
                "expected_facings": expected,
                "detected_facings": detected,
                "difference": detected - expected,
                "matched_facings": matched,
                "status": status,
            }
        )

    compliance_score = (
        1.0
        if comparison_slot_total == 0
        else matched_total / comparison_slot_total
    )

    is_compliant = missing_total == 0 and extra_total == 0

    return {
        "mode": "structural_facing_count",
        "status": (
            "compliant" if is_compliant else "non_compliant"
        ),
        "is_compliant": is_compliant,
        "compliance_score": round(compliance_score, 6),
        "compliance_percent": round(compliance_score * 100, 2),
        "expected_row_count": len(expected_facings),
        "detected_row_count": len(detected_facings),
        "expected_facing_count": sum(expected_facings),
        "detected_facing_count": sum(detected_facings),
        "matched_facing_count": matched_total,
        "missing_facing_count": missing_total,
        "extra_facing_count": extra_total,
        "rows": comparison_rows,
    }