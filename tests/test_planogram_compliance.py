"""Tests for structural planogram compliance."""

import unittest

from src.planogram_compliance import (
    compare_planogram,
    parse_expected_facings,
)


class PlanogramComplianceTests(unittest.TestCase):
    def test_parse_expected_facings(self) -> None:
        self.assertEqual(
            parse_expected_facings("15, 16, 17"),
            [15, 16, 17],
        )

    def test_exact_planogram_is_compliant(self) -> None:
        shelf_layout = {
            "rows": [
                {"detection_count": 15},
                {"detection_count": 16},
                {"detection_count": 17},
            ]
        }

        result = compare_planogram(
            shelf_layout,
            [15, 16, 17],
        )

        self.assertTrue(result["is_compliant"])
        self.assertEqual(result["compliance_percent"], 100.0)
        self.assertEqual(result["missing_facing_count"], 0)
        self.assertEqual(result["extra_facing_count"], 0)

    def test_missing_and_extra_facings(self) -> None:
        shelf_layout = {
            "rows": [
                {"detection_count": 15},
                {"detection_count": 16},
                {"detection_count": 17},
            ]
        }

        result = compare_planogram(
            shelf_layout,
            [15, 14, 20],
        )

        self.assertFalse(result["is_compliant"])
        self.assertEqual(result["compliance_percent"], 90.2)
        self.assertEqual(result["missing_facing_count"], 3)
        self.assertEqual(result["extra_facing_count"], 2)
        self.assertEqual(
            result["rows"][1]["status"],
            "extra_facings",
        )
        self.assertEqual(
            result["rows"][2]["status"],
            "missing_facings",
        )

    def test_invalid_planogram_input(self) -> None:
        invalid_values = (
            "",
            "15,,17",
            "15,a,17",
            "15,0,17",
        )

        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_expected_facings(value)


if __name__ == "__main__":
    unittest.main()