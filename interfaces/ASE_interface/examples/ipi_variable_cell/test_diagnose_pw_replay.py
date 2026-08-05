#!/usr/bin/env python3
"""Unit tests for the GPU PW replay diagnostic's pure decision layer."""
from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

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

    def test_load_replay_frames_requires_angstrom_ipi_metadata(self):
        invalid_headers = (
            ("# CELL(abcABC): 5.43 5.21921 5.58486 86.10424 88.25568 "
             "86.59486 Step: 0 Bead: 0 positions{angstrom} cell{bohr}"),
            ("# CELL(abcABC): 5.43 5.21921 5.58486 86.10424 88.25568 "
             "86.59486 Step: 0 Bead: 0 positions{bohr} cell{angstrom}"),
        )
        for header in invalid_headers:
            with self.subTest(header=header):
                self._write_xyz(["Si2"] * 6)
                self._replace_first_header(header)
                with self.assertRaises(AssertionError):
                    replay.load_replay_frames(
                        self.result_json, self.positions_xyz, (0, 1, 5))

    def test_load_replay_frames_rejects_angles_outside_open_domain(self):
        cellpar = [5.43, 5.21921, 5.58486,
                   86.10424, 88.25568, 86.59486]
        for angle_index in (3, 4, 5):
            with self.subTest(angle_index=angle_index):
                invalid_cellpar = cellpar.copy()
                invalid_cellpar[angle_index] += 360.0
                header = (
                    "# CELL(abcABC): {} Step: 0 Bead: 0 "
                    "positions{{angstrom}} cell{{angstrom}}".format(
                        " ".join(str(value) for value in invalid_cellpar)))
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


class ReplayExecutionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="pw-replay-execution-test-")
        self.directory = Path(self.temporary.name)
        self.cpu_input = self.directory / "INPUT.cpu"
        self.gpu_input = self.directory / "INPUT.gpu"
        self.cpu_input.write_text(self._input_text("cpu"))
        self.gpu_input.write_text(self._input_text("gpu"))

    def tearDown(self):
        self.temporary.cleanup()

    @staticmethod
    def _input_text(device):
        return "\n".join((
            "INPUT_PARAMETERS",
            "calculation scf",
            "basis_type pw",
            "device {}".format(device),
            "precision double",
            "ecutwfc 50",
            "symmetry 0",
            "kspacing 0.45",
            "scf_thr 1e-09",
            "scf_nmax 100",
            "chg_extrap atomic",
            "cal_force 1",
            "cal_stress 1",
            "",
        ))

    def write(self, name, text):
        path = self.directory / name
        path.write_text(text)
        return path

    def test_parse_raw_total_energy_requires_exactly_one_value(self):
        log = self.write(
            "running_scf.log", "#TOTAL ENERGY# -206.48515779761 eV\n")
        self.assertEqual(
            replay.parse_raw_total_energy(log), -206.48515779761)
        invalid_logs = (
            "",
            "#TOTAL ENERGY# -1 eV\n#TOTAL ENERGY# -2 eV\n",
            "#TOTAL ENERGY# nan eV\n",
            "!FINAL_ETOT_IS -3 eV\nE_KS(sigma->0) -4 eV\n",
        )
        for text in invalid_logs:
            with self.subTest(text=text):
                log.write_text(text)
                with self.assertRaises(AssertionError):
                    replay.parse_raw_total_energy(log)

    def test_paired_inputs_differ_only_by_device(self):
        replay.assert_paired_inputs(self.cpu_input, self.gpu_input)
        self.gpu_input.write_text(
            self.gpu_input.read_text() + "socket_driver 1\n")
        with self.assertRaises(AssertionError):
            replay.assert_paired_inputs(self.cpu_input, self.gpu_input)

    def test_paired_inputs_reject_socket_and_wrong_required_fields(self):
        mutations = (
            (self.cpu_input, "cal_stress 1", "cal_stress 0"),
            (self.gpu_input, "basis_type pw", "basis_type lcao"),
            (self.cpu_input, "scf_thr 1e-09", "scf_thr 1e-08"),
            (self.gpu_input, "cal_force 1", "cal_force 0"),
            (self.cpu_input, "", "socket_variable_cell 1\n"),
        )
        for path, old, new in mutations:
            with self.subTest(path=path.name, new=new):
                self.cpu_input.write_text(self._input_text("cpu"))
                self.gpu_input.write_text(self._input_text("gpu"))
                path.write_text(path.read_text().replace(old, new, 1))
                with self.assertRaises(AssertionError):
                    replay.assert_paired_inputs(
                        self.cpu_input, self.gpu_input)

    def test_run_fresh_frame_uses_raw_energy_and_one_device_frame(self):
        class FakeProfile:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class FakeAbacus(Calculator):
            implemented_properties = ["energy", "forces", "stress"]
            banner_device = "GPU"
            extra_frame = False

            def __init__(self, profile, directory, **kwargs):
                super().__init__()
                self.directory = Path(directory)
                self.device = kwargs["inp"]["device"]

            def calculate(self, atoms=None, properties=None,
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                directory = Path(self.directory)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "INPUT").write_text(
                    ReplayExecutionTests._input_text(self.device))
                output = directory / "OUT.ABACUS"
                output.mkdir()
                text = (
                    " RUNNING WITH DEVICE  : {} / fake-device (x1)\n".format(
                        type(self).banner_device) +
                    (
                    " #SCF IS CONVERGED#\n"
                    " #TOTAL ENERGY# -10.25 eV\n"
                    " #TOTAL-STRESS (kbar)#\n"
                    " 1 0 0\n 0 1 0\n 0 0 1\n"))
                if type(self).extra_frame:
                    text += (
                        " #SCF IS CONVERGED#\n"
                        " #TOTAL ENERGY# -10.24 eV\n"
                        " #TOTAL-STRESS (kbar)#\n"
                        " 1 0 0\n 0 1 0\n 0 0 1\n")
                (output / "running_scf.log").write_text(text)
                self.results = {
                    "energy": -10.0,
                    "forces": np.zeros((len(atoms), 3)),
                    "stress": np.zeros(6),
                }

        config = replay.ase_validation.Config(
            "fake-abacus", "pw", "gpu", "double",
            self.directory / "configured-gpu", self.directory / "unused.json",
            self.directory)
        atoms = Atoms("Si2", positions=[[0, 0, 0], [1, 1, 1]],
                      cell=np.eye(3) * 5.0, pbc=True)
        case = self.directory / "fresh-gpu"
        identity = {
            "executable_version": "v-test",
            "executable_sha256": "a" * 64,
            "source_commit": "b" * 40,
            "module": "test/module",
        }
        with mock.patch.object(
                replay.ase_validation, "_load_abacus_api",
                return_value=(FakeAbacus, FakeProfile, object)), \
                mock.patch.object(
                    replay.ase_validation, "_identity",
                    return_value=identity):
            record = replay.run_fresh_frame(config, atoms, case)

        self.assertEqual(record["energy_ev"], -10.25)
        self.assertEqual(record["fileio_energy_ev"], -10.0)
        self.assertEqual(
            record["energy_provenance"],
            "running_scf.log #TOTAL ENERGY#")
        self.assertTrue(record["scf_converged"])
        self.assertEqual(record["device"], "gpu")
        replay.run_validation.assert_json_ready(record)
        with self.assertRaises(AssertionError):
            replay.run_fresh_frame(config, atoms, case)

        for banner_device, extra_frame, name in (
                ("CPU", False, "wrong-device"),
                ("GPU", True, "multiple-frames")):
            with self.subTest(name=name):
                FakeAbacus.banner_device = banner_device
                FakeAbacus.extra_frame = extra_frame
                with mock.patch.object(
                        replay.ase_validation, "_load_abacus_api",
                        return_value=(FakeAbacus, FakeProfile, object)), \
                        mock.patch.object(
                            replay.ase_validation, "_identity",
                            return_value=identity):
                    with self.assertRaises(AssertionError):
                        replay.run_fresh_frame(
                            config, atoms, self.directory / name)


class ReplayOrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="pw-replay-orchestration-test-")
        self.directory = Path(self.temporary.name)
        self.validation_json = self.directory / "validation.json"
        self.positions_xyz = self.directory / "positions.xyz"
        self.pp_orb_root = self.directory / "PP_ORB"
        self.pp_orb_root.mkdir()
        (self.pp_orb_root / "Si_ONCV_PBE-1.2.upf").write_text(
            "synthetic pseudopotential\n")
        self.cpu_abacus = self._write_executable(
            "cpu-abacus", "v-test-cpu")
        self.gpu_abacus = self._write_executable(
            "gpu-abacus", "v-test-gpu")
        self.cell = np.array([
            [5.43, 0.31, 0.17],
            [0.00, 5.21, 0.37],
            [0.00, 0.00, 5.57],
        ], dtype=np.float64)
        self.steps = [self._stored_step(index) for index in range(51)]
        self.payload = {
            "schema_version": 1,
            "ipi_version": "3.2.0",
            "backend": "pw",
            "device": "gpu",
            "precision": "double",
            "trajectory": {"steps": self.steps},
        }
        self.validation_json.write_text(json.dumps(self.payload) + "\n")
        self._write_xyz()
        self.workdir = self.directory / "work"
        self.output = self.directory / "diagnostic.json"

    def tearDown(self):
        self.temporary.cleanup()

    def _write_executable(self, name, version):
        path = self.directory / name
        path.write_text(
            "#!/bin/sh\necho 'ABACUS version {}'\n".format(version))
        path.chmod(0o755)
        return path

    def _stored_step(self, index):
        energy = -10.0 + 0.001 * index
        return {
            "source": "socket-trajectory",
            "backend": "pw",
            "device": "gpu",
            "precision": "double",
            "scf_converged": True,
            "energy_ev": energy,
            "cell_angstrom": self.cell.tolist(),
            "forces_ev_per_angstrom": [
                [0.0001 * index, 0.0, 0.0],
                [-0.0001 * index, 0.0, 0.0],
            ],
            "ase_stress_ev_per_angstrom3": [
                0.001 * (index + component) for component in range(6)
            ],
            "executable_version": "v-source-gpu",
            "executable_sha256": "c" * 64,
            "source_commit": "d" * 40,
            "module": "source/module",
        }

    def _write_xyz(self):
        lines = []
        for index in range(51):
            lines.extend([
                "2",
                ("# CELL(abcABC): 5.43 5.21921 5.58486 "
                 "86.10424 88.25568 86.59486 Step: {} Bead: 0 "
                 "positions{{angstrom}} cell{{angstrom}}".format(index)),
                "Si {:.8f} 0.20000000 0.30000000".format(
                    0.1 + 0.001 * index),
                "Si 2.70000000 {:.8f} 2.50000000".format(
                    2.6 + 0.001 * index),
            ])
        self.positions_xyz.write_text("\n".join(lines) + "\n")

    def _argv(self, workdir=None, output=None):
        return [
            "--validation-json", str(self.validation_json),
            "--positions", str(self.positions_xyz),
            "--cpu-abacus", str(self.cpu_abacus),
            "--gpu-abacus", str(self.gpu_abacus),
            "--pp-orb-root", str(self.pp_orb_root),
            "--frames", "0,1,5,8,42,50",
            "--workdir", str(workdir or self.workdir),
            "--output", str(output or self.output),
        ]

    def _fresh_runner(self, calls, invalid=None):
        def fresh_runner(config, atoms, directory):
            directory = Path(directory)
            self.assertFalse(directory.exists())
            directory.mkdir(parents=True)
            (directory / "INPUT").write_text(
                ReplayExecutionTests._input_text(config.device))
            index = int(directory.name.split("-")[1])
            calls.append({
                "index": index,
                "device": config.device,
                "directory": str(directory.resolve()),
                "config_workdir": str(config.workdir.resolve()),
                "cell": atoms.cell.array.copy(),
                "positions": atoms.positions.copy(),
            })
            record = copy.deepcopy(self.steps[index])
            record.update({
                "source": "fileio",
                "device": config.device,
                "cell_angstrom": atoms.cell.array.tolist(),
                "fileio_energy_ev": record["energy_ev"] + 0.125,
                "energy_provenance": "running_scf.log #TOTAL ENERGY#",
            })
            if invalid == "unconverged" and len(calls) == 4:
                record["scf_converged"] = False
            elif invalid == "nonfinite" and len(calls) == 4:
                record["energy_ev"] = float("nan")
            elif invalid == "missing" and len(calls) == 4:
                record.pop("forces_ev_per_angstrom")
            return record
        return fresh_runner

    def test_cli_parses_all_required_diagnostic_paths(self):
        args = replay.parse_args(self._argv())
        self.assertEqual(args.validation_json, self.validation_json)
        self.assertEqual(args.positions, self.positions_xyz)
        self.assertEqual(args.cpu_abacus, self.cpu_abacus)
        self.assertEqual(args.gpu_abacus, self.gpu_abacus)
        self.assertEqual(args.pp_orb_root, self.pp_orb_root)
        self.assertEqual(args.frames, "0,1,5,8,42,50")
        self.assertEqual(args.workdir, self.workdir)
        self.assertEqual(args.output, self.output)

    def test_run_diagnostic_records_twelve_fresh_paired_cases(self):
        stale = self.workdir / "frame-000-cpu"
        stale.mkdir(parents=True)
        (stale / "stale").write_text("remove only this case\n")
        calls = []

        result = replay.run_diagnostic(
            replay.parse_args(self._argv()),
            fresh_runner=self._fresh_runner(calls))

        expected_order = [
            (index, device)
            for index in replay.DEFAULT_FRAME_INDICES
            for device in ("cpu", "gpu")
        ]
        self.assertEqual(
            [(call["index"], call["device"]) for call in calls],
            expected_order)
        self.assertEqual(len({call["directory"] for call in calls}), 12)
        for pair in range(0, len(calls), 2):
            np.testing.assert_array_equal(
                calls[pair]["cell"], calls[pair + 1]["cell"])
            np.testing.assert_array_equal(
                calls[pair]["positions"], calls[pair + 1]["positions"])
            self.assertTrue(calls[pair]["config_workdir"].endswith("/cpu"))
            self.assertTrue(calls[pair + 1]["config_workdir"].endswith("/gpu"))

        self.assertEqual(result["schema_version"], 1)
        self.assertEqual(
            result["kind"], "gpu-pw-socket-replay-diagnostic")
        self.assertEqual(
            result["selected_frames"], list(replay.DEFAULT_FRAME_INDICES))
        self.assertEqual(
            result["classification"],
            "trajectory_geometry_explains_difference")
        self.assertEqual(
            result["thresholds"],
            replay.ase_validation.IDENTICAL_DOUBLE_LIMITS)
        self.assertEqual(
            result["source_validation"]["validation_json_path"],
            str(self.validation_json.resolve()))
        self.assertEqual(
            result["source_validation"]["validation_json_sha256"],
            hashlib.sha256(self.validation_json.read_bytes()).hexdigest())
        self.assertEqual(
            result["source_validation"]["positions_sha256"],
            hashlib.sha256(self.positions_xyz.read_bytes()).hexdigest())
        self.assertEqual(
            result["source_validation"]["source_identity"]["source_commit"],
            "d" * 40)
        self.assertEqual(
            result["cpu_identity"]["executable_version"], "v-test-cpu")
        self.assertEqual(
            result["gpu_identity"]["executable_version"], "v-test-gpu")
        self.assertEqual(
            result["cpu_identity"]["executable_path"],
            str(self.cpu_abacus.resolve()))
        self.assertEqual(
            result["gpu_identity"]["executable_path"],
            str(self.gpu_abacus.resolve()))
        self.assertEqual(
            result["cpu_identity"]["executable_sha256"],
            hashlib.sha256(self.cpu_abacus.read_bytes()).hexdigest())
        self.assertEqual(
            result["pseudopotential"]["sha256"],
            hashlib.sha256(
                (self.pp_orb_root / "Si_ONCV_PBE-1.2.upf").read_bytes()
            ).hexdigest())

        self.assertEqual(len(result["frames"]), 6)
        for frame, index in zip(
                result["frames"], replay.DEFAULT_FRAME_INDICES):
            self.assertEqual(frame["index"], index)
            self.assertEqual(
                frame["source_positions_path"],
                str(self.positions_xyz.resolve()))
            self.assertEqual(
                frame["cpu_case_path"],
                str((self.workdir / "frame-{:03d}-cpu".format(
                    index)).resolve()))
            self.assertEqual(
                frame["gpu_case_path"],
                str((self.workdir / "frame-{:03d}-gpu".format(
                    index)).resolve()))
            self.assertTrue(frame["fresh_cpu_gpu_decision"]["pass"])
            self.assertTrue(frame["stored_gpu_fresh_gpu_decision"]["pass"])
            self.assertEqual(
                frame["fresh_gpu"]["energy_provenance"],
                "running_scf.log #TOTAL ENERGY#")
            self.assertIn("fileio_energy_ev", frame["fresh_cpu"])
            self.assertIn("stored_gpu", frame)

        replay.run_validation.assert_json_ready(result)
        serialized = json.loads(self.output.read_text())
        self.assertEqual(serialized, result)
        self.assertTrue(self.output.read_text().endswith("\n"))

    def test_run_diagnostic_never_writes_partial_or_invalid_json(self):
        for invalid in ("unconverged", "nonfinite", "missing"):
            with self.subTest(invalid=invalid):
                workdir = self.directory / ("work-" + invalid)
                output = self.directory / ("output-" + invalid + ".json")
                calls = []
                with self.assertRaises(AssertionError):
                    replay.run_diagnostic(
                        replay.parse_args(self._argv(workdir, output)),
                        fresh_runner=self._fresh_runner(calls, invalid))
                self.assertFalse(output.exists())
                self.assertEqual(len(calls), 4)

if __name__ == "__main__":
    unittest.main()
