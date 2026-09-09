"""Build an ordered shelf layout from product detections."""

from __future__ import annotations

from statistics import fmean
from typing import Any


def build_shelf_layout(
    detections: list[dict[str, Any]],
    row_tolerance: float = 0.08,
    min_detections_per_row: int=2,
) -> dict[str, Any]:
    """Group detections into shelf rows and sort each row left to right."""
    if not 0 < row_tolerance <= 1:
        raise ValueError("row_tolerance must be between 0 and 1.")

    if min_detections_per_row < 1:
        raise ValueError("min_detections_per_row must be at least 1.")

    if not detections:
        return {
            "row_count": 0,
            "row_tolerance": row_tolerance,
            "min_detections_per_row": min_detections_per_row,
            "unassigned_count": 0,
            "unassigned_detections": [],
            "rows": [],
        }

    required_fields = {
        "center_x_norm",
        "center_y_norm",
        "y1_norm",
        "y2_norm",
    }

    prepared_detections = []

    for detection in detections:
        missing_fields = required_fields.difference(detection)

        if missing_fields:
            missing_text = ", ".join(sorted(missing_fields))
            raise ValueError(
                f"Detection is missing required fields: {missing_text}"
            )

        for field in required_fields:
            value = float(detection[field])

            if not 0 <= value <= 1:
                raise ValueError(
                    f"{field} must be between 0 and 1: {value}"
                )

        prepared_detections.append(dict(detection))

    prepared_detections.sort(
        key=lambda item: (
            float(item["center_y_norm"]),
            float(item["center_x_norm"]),
        )
    )

    row_clusters: list[list[dict[str, Any]]] = []

    def calculate_row_center(row: list[dict[str, Any]]) -> float:
        return fmean(float(item["center_y_norm"]) for item in row)

    for detection in prepared_detections:
        detection_center_y = float(detection["center_y_norm"])

        if not row_clusters:
            row_clusters.append([detection])
            continue

        closest_row_index = min(
            range(len(row_clusters)),
            key=lambda index: abs(
                calculate_row_center(row_clusters[index])
                - detection_center_y
            ),
        )

        closest_distance = abs(
            calculate_row_center(row_clusters[closest_row_index])
            - detection_center_y
        )

        if closest_distance <= row_tolerance:
            row_clusters[closest_row_index].append(detection)
        else:
            row_clusters.append([detection])

        row_clusters.sort(key=calculate_row_center)

    accepted_row_clusters = []
    unassigned_detections = []

    for row_cluster in row_clusters:
        if len(row_cluster) < min_detections_per_row:
            for detection in row_cluster:
                unassigned_detection = dict(detection)
                unassigned_detection["row_index"] = None
                unassigned_detection["position_index"] = None
                unassigned_detection["slot_id"] = None
                unassigned_detection["layout_status"] = "review_required"
                unassigned_detections.append(unassigned_detection)
        else:
            accepted_row_clusters.append(row_cluster)

    rows = []

    for row_index, row_detections in enumerate(
        accepted_row_clusters,
        start=1,
    ):
        row_detections.sort(
            key=lambda item: float(item["center_x_norm"])
        )

        ordered_detections = []

        for position_index, detection in enumerate(
            row_detections,
            start=1,
        ):
            ordered_detection = dict(detection)
            ordered_detection["row_index"] = row_index
            ordered_detection["position_index"] = position_index
            ordered_detection["slot_id"] = (
                f"R{row_index}P{position_index}"
            )
            ordered_detection["layout_status"] = "assigned"
            ordered_detections.append(ordered_detection)

        rows.append(
            {
                "row_index": row_index,
                "center_y_norm": round(
                    calculate_row_center(ordered_detections),
                    6,
                ),
                "top_y_norm": round(
                    min(
                        float(item["y1_norm"])
                        for item in ordered_detections
                    ),
                    6,
                ),
                "bottom_y_norm": round(
                    max(
                        float(item["y2_norm"])
                        for item in ordered_detections
                    ),
                    6,
                ),
                "detection_count": len(ordered_detections),
                "detections": ordered_detections,
            }
        )

    return {
        "row_count": len(rows),
        "row_tolerance": row_tolerance,
        "min_detections_per_row": min_detections_per_row,
        "unassigned_count": len(unassigned_detections),
        "unassigned_detections": unassigned_detections,
        "rows": rows,
    }