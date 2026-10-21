"""Combining public-MLA runs: refuses unauthenticated or mismatched runs, and the diagnostic."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_SPEC = importlib.util.spec_from_file_location(
    "combine_public_mla", ROOT / "experiments" / "combine_public_mla_runs.py")
COMBINE = importlib.util.module_from_spec(_SPEC)
sys.modules["combine_public_mla"] = COMBINE
_SPEC.loader.exec_module(COMBINE)


def write_run(directory, name, positions, basis):
    raw_path = Path(directory) / f"{name}.raw.npz"
    np.savez(raw_path, eval_tokens=np.arange(6).reshape(2, 3), positions=positions,
             rank_grid=np.array([1, 2]), baseline=np.zeros((2, 2), dtype=np.float32),
             **{f"delta_{basis}": np.zeros((2, 2, 2, 4), dtype=np.float32)})
    run = {"provenance": {key: "x" for key in COMBINE.SHARED_PROVENANCE},
           "results_by_basis": {basis: {}}}
    run["provenance"]["raw_npz_sha256"] = COMBINE._RISK.file_sha256(raw_path)
    run.update({key: 0 for key in COMBINE.SHARED_SETTINGS})
    path = Path(directory) / f"{name}.json"
    path.write_text(json.dumps(run))
    return path


class CombinePublicMlaTest(unittest.TestCase):
    def test_refuses_tampered_raw_and_different_probes(self):
        with tempfile.TemporaryDirectory() as directory:
            first = write_run(directory, "a", np.array([[1, 2], [1, 2]]), "coord")
            same = write_run(directory, "b", np.array([[1, 2], [1, 2]]), "pca")
            other = write_run(directory, "c", np.array([[1, 3], [1, 2]]), "grad")
            COMBINE.check_same_probes([COMBINE.load_run(first), COMBINE.load_run(same)])
            with self.assertRaises(SystemExit):
                COMBINE.check_same_probes([COMBINE.load_run(first), COMBINE.load_run(other)])
            np.savez(Path(directory) / "b.raw.npz", eval_tokens=np.zeros(1))
            with self.assertRaises(SystemExit):
                COMBINE.load_run(same)

    def test_mean_safe_diagnostic_reads_the_accepted_rank(self):
        deltas = np.zeros((1, 2, 2, 4))
        deltas[0, 0, 1] = [0.3, -0.3, 0.1, -0.1]  # probe 0 accepted at rank index 1
        deltas[0, 1, 0] = [0.05, 0.05, 0.0, 0.0]  # probe 1 accepted at rank index 0
        result = {"signed_mean": {"r_star_by_sequence": {"0": [8, 4]}}}
        out = COMBINE.mean_safe_diagnostics(result, deltas, np.array([4, 8]), 0.1)
        self.assertAlmostEqual(out["signed_mean"], 0.0125)
        self.assertAlmostEqual(out["positive_part_mean"], (0.1 + 0.025) / 2)
        self.assertEqual(out["fraction_with_any_offset_above_epsilon"], 0.5)


if __name__ == "__main__":
    unittest.main()
