import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "horizon_capacity", ROOT / "experiments" / "analyze_horizon_capacity.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def points(values, horizons=(1, 2, 4, 8)):
    return [
        {"horizon": h, "normalized_r_star": v, "mean_r_star": v * 256,
         "normalized_ci": [v - 0.01, v + 0.01]}
        for h, v in zip(horizons, values)
    ]


class HorizonVerdictTests(unittest.TestCase):
    def test_rising_then_flat_curve_is_nondecreasing_and_saturating(self):
        v = MODULE.verdict(points([0.20, 0.45, 0.60, 0.62]), tolerance=0.0)
        self.assertTrue(v["nondecreasing"])
        self.assertEqual(v["decreases"], [])
        self.assertTrue(v["steps_are_shrinking"])
        self.assertAlmostEqual(v["total_change"], 0.42)

    def test_a_single_dip_is_reported_with_its_endpoints(self):
        v = MODULE.verdict(points([0.20, 0.45, 0.40, 0.62]), tolerance=0.0)
        self.assertFalse(v["nondecreasing"])
        self.assertEqual(len(v["decreases"]), 1)
        self.assertEqual(v["decreases"][0]["from"], 2)
        self.assertEqual(v["decreases"][0]["to"], 4)
        self.assertAlmostEqual(v["decreases"][0]["drop"], 0.05)

    def test_tolerance_absorbs_a_dip_smaller_than_it(self):
        curve = points([0.20, 0.45, 0.44, 0.62])
        self.assertFalse(MODULE.verdict(curve, tolerance=0.0)["nondecreasing"])
        self.assertTrue(MODULE.verdict(curve, tolerance=0.02)["nondecreasing"])

    def test_growing_steps_are_not_called_saturating(self):
        v = MODULE.verdict(points([0.10, 0.15, 0.25, 0.45]), tolerance=0.0)
        self.assertTrue(v["nondecreasing"])
        self.assertFalse(v["steps_are_shrinking"])

    def test_curve_reads_the_nested_prefix_spectrum_and_normalizes_the_ci(self):
        summary = {
            "d_c": 256,
            "horizon_capacity_spectrum": {
                "maximum": {
                    "1": {"mean_r_star": 64.0, "normalized_mean_r_star": 0.25,
                          "sequence_cluster_bootstrap_95pct_ci": [32.0, 96.0]},
                    "2": {"mean_r_star": 128.0, "normalized_mean_r_star": 0.5,
                          "sequence_cluster_bootstrap_95pct_ci": [96.0, 160.0]},
                }
            },
        }
        horizons, pts = MODULE.curve(summary, "maximum")
        self.assertEqual(horizons, [1, 2])
        self.assertEqual(pts[0]["normalized_ci"], [0.125, 0.375])
        self.assertEqual(pts[1]["normalized_r_star"], 0.5)

    def test_missing_spectrum_is_rejected_rather_than_silently_skipped(self):
        path = ROOT / "tests" / "__pycache__" / "_no_spectrum.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"d_c": 256}')
        with self.assertRaisesRegex(ValueError, "horizon_capacity_spectrum"):
            MODULE.load(path)


if __name__ == "__main__":
    unittest.main()
