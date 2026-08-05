#!/usr/bin/env python3
"""Unit tests for the GPU PW replay diagnostic's pure decision layer."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase import Atoms

from . import diagnose_pw_replay as replay


class ReplayFrameTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pw-replay-test-")
        self.directory = Path(self.temporary.name)
        self.result_json = self.directory / "result.json"
        self.positions_xyz = self.directory / "positions.xyz"
        self.xyz_positions = [
            np.array([[0.10 + 0.01 * index, 0.20, 0.30],
                      [2.70, 2.60 + 0.02 * index, 2.50]], dtype=np.float64)
            for index in range(6)
        ]
        self.json_cells = [np.array([
            [5.429999929239518, 0.30999999596026717,
             0.16999999778466263],
            [0.0, 5.209999932106425, 0.36999999517838333],
            [0.0, 0.0, 5.569999927415123],
        ], dtype=np.float64) for _ in range(6)]
        self.payload = {
            "schema_version": 1,
            "ipi_version": "3.2.0",
            "backend": "pw",
            "device": "gpu",
            "precision": "double",
            "trajectory": {
                "steps": [self._stored_step(index) for index in range(6)]
            },
        }
        self._write_payload(self.payload)
        self._write_xyz(["Si2"] * 6)

    def tearDown(self):
        self.temporary.cleanup()

    def _stored_step(self, index):
        return {
            "cell_angstrom": self.json_cells[index].tolist(),
            "energy_ev": float(-10.0 + 0.01 * index),
            "forces_ev_per_angstrom": [
                [0.001 * index, 0.0, 0.0],
                [-0.001 * index, 0.0, 0.0],
            ],
            "ase_stress_ev_per_angstrom3": [
                float(0.001 * (index + component))
                for component in range(6)
            ],
        }

    def _write_payload(self, payload):
        self.result_json.write_text(json.dumps(payload) + "\n")

    def _write_xyz(self, symbols):
        lines = []
        for index, frame_symbols in enumerate(symbols):
            chemical_symbols = Atoms(frame_symbols).get_chemical_symbols()
            lines.extend([
                str(len(chemical_symbols)),
                ("# CELL(abcABC):    5.43000     5.21921     5.58486  "
                 "  86.10424    88.25568    86.59486  Step: "
                 "          {:d}  Bead:       0 positions{{angstrom}}  "
                 "cell{{angstrom}}".format(index)),
            ])
            for symbol, position in zip(
                    chemical_symbols, self.xyz_positions[index]):
                lines.append("{:>8s} {: .8e} {: .8e} {: .8e}".format(
                    symbol, *position))
        self.positions_xyz.write_text("\n".join(lines) + "\n")

    def _replace_first_header(self, replacement):
        lines = self.positions_xyz.read_text().splitlines()
        lines[1] = replacement
        self.positions_xyz.write_text("\n".join(lines) + "\n")

    def test_parse_frame_indices_requires_unique_sorted_in_range_values(self):
        self.assertEqual(replay.parse_frame_indices("0,1,5", 6), (0, 1, 5))
        for value in ("", "1,1", "2,1", "-1", "6"):
            with self.subTest(value=value):
                with self.assertRaises((ValueError, AssertionError)):
                    replay.parse_frame_indices(value, 6)

    def test_load_replay_frames_uses_xyz_positions_and_json_cell(self):
        payload, frames = replay.load_replay_frames(
            self.result_json, self.positions_xyz, (0, 1, 5))
        self.assertEqual([frame["index"] for frame in frames], [0, 1, 5])
        np.testing.assert_allclose(frames[1]["atoms"].positions,
                                   self.xyz_positions[1], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(frames[1]["atoms"].cell.array,
                                   self.json_cells[1], rtol=0.0, atol=0.0)
        np.testing.assert_array_equal(frames[1]["atoms"].pbc,
                                      [True, True, True])
        self.assertAlmostEqual(
            frames[0]["xyz_cell_max_abs_delta_angstrom"],
            4.428902746766994e-06, places=15)
        self.assertEqual(payload["ipi_version"], "3.2.0")

    def test_load_replay_frames_rejects_invalid_ipi_cell_metadata(self):
        invalid_headers = (
            "# Step: 0 Bead: 0 positions{angstrom} cell{angstrom}",
            "# CELL(abcABC): 5.43 5.21921 5.58486 86.10424 88.25568 Step: 0",
            ("# CELL(abcABC): nan 5.21921 5.58486 86.10424 88.25568 "
             "86.59486 Step: 0"),
            ("# CELL(abcABC): -5.43 5.21921 5.58486 86.10424 88.25568 "
             "86.59486 Step: 0"),
        )
        for header in invalid_headers:
            with self.subTest(header=header):
                self._write_xyz(["Si2"] * 6)
                self._replace_first_header(header)
                with self.assertRaises(AssertionError):
                    replay.load_replay_frames(
                        self.result_json, self.positions_xyz, (0, 1, 5))

    def test_actual_job_762695_frames_are_periodic_with_small_cell_delta(self):
        runtime = Path("/home/gengjianrui/bin/abacus-variable-cell-runtime")
        result = runtime / "results/gpu-pw-isotropic-762695.json"
        positions = runtime / (
            "work/gpu-pw-isotropic-762695/trajectory/"
            "isotropic-abacus_vc_isotropic_1775430_"
            "1785917255506141486.positions_0.xyz")
        if not result.is_file() or not positions.is_file():
            self.skipTest("read-only Job 762695 artifacts are unavailable")

        _, frames = replay.load_replay_frames(
            result, positions, replay.DEFAULT_FRAME_INDICES)

        self.assertTrue(all(np.all(frame["atoms"].pbc) for frame in frames))
        deltas = np.asarray([
            frame["xyz_cell_max_abs_delta_angstrom"] for frame in frames])
        self.assertTrue(np.all(np.isfinite(deltas)))
        self.assertLess(float(np.max(deltas)), 1.0e-5)

    def test_load_replay_frames_rejects_frame_count_mismatch(self):
        payload = copy.deepcopy(self.payload)
        payload["trajectory"]["steps"].pop()
        self._write_payload(payload)
        with self.assertRaises(AssertionError):
            replay.load_replay_frames(
                self.result_json, self.positions_xyz, (0, 1, 4))

    def test_load_replay_frames_rejects_nonfinite_cell(self):
        payload = copy.deepcopy(self.payload)
        payload["trajectory"]["steps"][1]["cell_angstrom"][0][0] = float("nan")
        self._write_payload(payload)
        with self.assertRaises(AssertionError):
            replay.load_replay_frames(
                self.result_json, self.positions_xyz, (0, 1, 5))

    def test_load_replay_frames_requires_gpu_pw_double_source(self):
        for key, wrong_value in (("backend", "lcao"),
                                 ("device", "cpu"),
                                 ("precision", "single")):
            with self.subTest(key=key):
                payload = copy.deepcopy(self.payload)
                payload[key] = wrong_value
                self._write_payload(payload)
                with self.assertRaises(AssertionError):
                    replay.load_replay_frames(
                        self.result_json, self.positions_xyz, (0, 1, 5))

    def test_load_replay_frames_requires_ipi_3_2_0(self):
        payload = copy.deepcopy(self.payload)
        payload["ipi_version"] = "3.1.0"
        self._write_payload(payload)
        with self.assertRaises(AssertionError):
            replay.load_replay_frames(
                self.result_json, self.positions_xyz, (0, 1, 5))

    def test_load_replay_frames_rejects_atom_symbol_mismatch(self):
        self._write_xyz(["Si2", "Si2", "Si2", "Si2", "Si2", "SiC"])
        with self.assertRaises(AssertionError):
            replay.load_replay_frames(
                self.result_json, self.positions_xyz, (0, 1, 5))


class ReplayDecisionTests(unittest.TestCase):
    def setUp(self):
        self.reference = {
            "energy_ev": -10.0,
            "forces_ev_per_angstrom": [
                [1.0, -2.0, 3.0],
                [-1.0, 2.0, -3.0],
            ],
            "ase_stress_ev_per_angstrom3": [
                0.10, -0.20, 0.30, -0.40, 0.50, -0.60,
            ],
        }

    def test_compare_records_uses_max_force_and_six_stress_errors(self):
        candidate = copy.deepcopy(self.reference)
        candidate["energy_ev"] += 5.0e-5
        candidate["forces_ev_per_angstrom"][0][2] += 4.0e-6
        stress_deltas = np.arange(1.0e-6, 7.0e-6, 1.0e-6)
        candidate["ase_stress_ev_per_angstrom3"] = (
            np.asarray(candidate["ase_stress_ev_per_angstrom3"])
            + stress_deltas).tolist()

        decision = replay.compare_records(self.reference, candidate)

        self.assertAlmostEqual(decision["energy_absolute_error_ev"], 5.0e-5)
        self.assertEqual(decision["energy_reference_abs_ev"], 10.0)
        self.assertAlmostEqual(
            decision["force_max_absolute_error_ev_per_angstrom"], 4.0e-6)
        self.assertEqual(
            decision["force_reference_max_abs_ev_per_angstrom"], 3.0)
        np.testing.assert_allclose(
            decision["stress_absolute_errors_ev_per_angstrom3"],
            stress_deltas, rtol=1.0e-10, atol=1.0e-17)
        np.testing.assert_allclose(
            decision["stress_reference_abs_ev_per_angstrom3"],
            [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
            rtol=0.0, atol=0.0)
        self.assertTrue(decision["pass"])

    def test_compare_records_rejects_invalid_numeric_records(self):
        for key, value in (
                ("energy_ev", float("nan")),
                ("forces_ev_per_angstrom", [[0.0, 0.0, 0.0]]),
                ("ase_stress_ev_per_angstrom3", [0.0] * 5)):
            with self.subTest(key=key):
                candidate = copy.deepcopy(self.reference)
                candidate[key] = value
                with self.assertRaises(AssertionError):
                    replay.compare_records(self.reference, candidate)

    def test_classification_order(self):
        all_pass = [{"pass": True}] * 6
        later_fail = ([{"pass": True}, {"pass": False}]
                      + [{"pass": True}] * 4)
        self.assertEqual(replay.classify_replay(
            [{"pass": False}] + all_pass[1:], all_pass,
            replay.DEFAULT_FRAME_INDICES),
            "general_gpu_backend_difference")
        self.assertEqual(replay.classify_replay(
            all_pass, later_fail, replay.DEFAULT_FRAME_INDICES),
            "continuous_socket_state_suspected")
        self.assertEqual(replay.classify_replay(
            all_pass, all_pass, replay.DEFAULT_FRAME_INDICES),
            "trajectory_geometry_explains_difference")

    def test_classification_is_inconclusive_when_stored_step_zero_fails(self):
        all_pass = [{"pass": True}] * 6
        stored_step_zero_fail = ([{"pass": False}]
                                 + [{"pass": True}] * 5)
        self.assertEqual(replay.classify_replay(
            all_pass, stored_step_zero_fail, replay.DEFAULT_FRAME_INDICES),
            "inconclusive")

    def test_classification_rejects_wrong_counts_and_types(self):
        all_pass = [{"pass": True}] * 6
        invalid_arguments = (
            (all_pass[:-1], all_pass, replay.DEFAULT_FRAME_INDICES),
            (tuple(all_pass), all_pass, replay.DEFAULT_FRAME_INDICES),
            (all_pass, [{"pass": 1}] * 6, replay.DEFAULT_FRAME_INDICES),
            (all_pass, all_pass, (1, 5, 8, 42, 50, 51)),
        )
        for cpu_gpu, stored_gpu, indices in invalid_arguments:
            with self.subTest(cpu_gpu=cpu_gpu, stored_gpu=stored_gpu,
                              indices=indices):
                with self.assertRaises(AssertionError):
                    replay.classify_replay(cpu_gpu, stored_gpu, indices)


if __name__ == "__main__":
    unittest.main()
