from __future__ import annotations

import unittest

import numpy as np

from paper_analysis.plot_calibration import calibration_coefficients, wilson_interval


class CalibrationTests(unittest.TestCase):
    def test_exact_group_frequencies_give_unit_slope(self) -> None:
        probabilities = []
        targets = []
        for probability in (0.1, 0.2, 0.4, 0.6, 0.8, 0.9):
            total = 100
            positives = round(total * probability)
            probabilities.extend([probability] * total)
            targets.extend([1.0] * positives + [0.0] * (total - positives))
        intercept, slope = calibration_coefficients(
            np.asarray(targets), np.asarray(probabilities)
        )
        self.assertAlmostEqual(intercept, 0.0, places=6)
        self.assertAlmostEqual(slope, 1.0, places=6)

    def test_wilson_interval_contains_observed_proportion(self) -> None:
        low, high = wilson_interval(30, 100)
        self.assertLess(low, 0.30)
        self.assertGreater(high, 0.30)
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 1.0)


if __name__ == "__main__":
    unittest.main()
