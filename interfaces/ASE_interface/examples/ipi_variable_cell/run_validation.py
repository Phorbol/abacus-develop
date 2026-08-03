#!/usr/bin/env python3
"""Run pinned i-PI 3.2.0 isotropic/flexible NPT validation with ABACUS."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import socketio_variable_cell as ase_validation

IPI_VERSION = "3.2.0"
GPA_PER_EV_ANGSTROM3 = 160.2176634
BOHR_ANGSTROM = 0.529177210903
HARTREE_EV = 27.211386245988
TEMPLATE_KEYS = ("__SOCKET_NAME__", "__TOTAL_STEPS__", "__PRESSURE_GPA__",
                 "__SEED__", "__PREFIX__")


@contextlib.contextmanager
def working_directory(directory: Path):
    previous = Path.cwd()
    os.chdir(directory)
    try:
        yield
    finally:
        os.chdir(previous)


def render_template(template: Path, socket_name: str, total_steps: int,
                    pressure_gpa: float, seed: int, prefix: str) -> str:
    if total_steps <= 0 or not np.isfinite(pressure_gpa):
        raise ValueError("invalid i-PI steps or target pressure")
    text = template.read_text()
    values = {
        "__SOCKET_NAME__": socket_name,
        "__TOTAL_STEPS__": str(int(total_steps)),
        "__PRESSURE_GPA__": format(float(pressure_gpa), ".17g"),
        "__SEED__": str(int(seed)),
        "__PREFIX__": prefix,
    }
    for key, value in values.items():
        text = text.replace(key, value)
    leftovers = [key for key in TEMPLATE_KEYS if key in text]
    if leftovers:
        raise AssertionError("unsubstituted XML keys: " + ",".join(leftovers))
    root = ET.fromstring(text)
    if root.tag != "simulation":
        raise AssertionError("i-PI template root must be simulation")
    return text


def validate_official_xml(xml_text: str, directory: Path):
    """Parse with the official i-PI 3.2.0 engine, not only ElementTree."""
    import ipi
    from ipi.engine.simulation import Simulation
    if ipi.__version__ != IPI_VERSION:
        raise AssertionError("requires official i-PI=={}".format(IPI_VERSION))
    with working_directory(directory):
        simulation = Simulation.load_from_xml(
            io.StringIO(xml_text), read_only=True, request_banner=False)
    if simulation is None:
        raise AssertionError("official i-PI parser returned no simulation")
    return simulation


def wait_until(predicate, timeout: float, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    """Bounded cleanup used from every real/fake runner finally block."""
    for process in reversed(processes):
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 3.0
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2.0)


def parse_properties(path: Path, expected_steps: int) -> dict:
    """Parse official six-component cell_h/virial_md plus two-atom forces."""
    if not path.is_file():
        raise AssertionError("i-PI properties output is absent: {}".format(path))
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError:
            continue
        if len(values) < 22:
            raise AssertionError("i-PI properties row lacks cell/virial columns")
        if not np.all(np.isfinite(values)):
            raise AssertionError("i-PI properties contain non-finite values")
        forces = np.asarray(values[3:9], dtype=np.float64).reshape(2, 3)
        volume = values[9]
        cell6 = np.asarray(values[10:16], dtype=np.float64)
        virial_pressure6 = np.asarray(values[16:22], dtype=np.float64)
        # i-PI tensor2vec order is xx,yy,zz,xy,xz,yz.
        cell = np.array([[cell6[0], cell6[3], cell6[4]],
                         [0.0, cell6[1], cell6[5]],
                         [0.0, 0.0, cell6[2]]], dtype=np.float64)
        virial_pressure = np.array(
            [[virial_pressure6[0], virial_pressure6[3], virial_pressure6[4]],
             [virial_pressure6[3], virial_pressure6[1], virial_pressure6[5]],
             [virial_pressure6[4], virial_pressure6[5], virial_pressure6[2]]],
            dtype=np.float64)
        determinant = float(np.linalg.det(cell))
        condition = float(np.linalg.cond(cell, 2))
        if determinant <= 0.0 or not np.isfinite(condition) or condition >= 1.0e12:
            raise AssertionError("i-PI output cell is invalid")
        rows.append({
            "step": int(round(values[0])), "potential_ev": values[1],
            "conserved_ev": values[2], "forces_ev_per_angstrom": forces.tolist(),
            "volume_bohr3": volume,
            "cell_bohr": cell.tolist(), "condition_number": condition,
            "virial_pressure_hartree_per_bohr3": virial_pressure.tolist(),
            "virial_hartree": (virial_pressure * volume).tolist(),
        })
    if len(rows) < expected_steps:
        raise AssertionError("only {} of {} i-PI steps parsed".format(
            len(rows), expected_steps))
    return {"steps": rows, "completed_steps": len(rows),
            "volume_sequence_bohr3": [row["volume_bohr3"] for row in rows]}


def paired_pressure_decision(low: dict, high: dict) -> dict:
    low_target = float(low["target_pressure_gpa"])
    high_target = float(high["target_pressure_gpa"])
    low_volumes = np.asarray(low["volume_sequence_bohr3"], dtype=np.float64)
    high_volumes = np.asarray(high["volume_sequence_bohr3"], dtype=np.float64)
    if low_volumes.size < 5 or high_volumes.size < 5:
        raise AssertionError("paired pressure probe requires five steps per replica")
    passed = bool(high_target > low_target
                  and high_volumes[-1] < low_volumes[-1])
    result = {
        "low_target_pressure_gpa": low_target,
        "high_target_pressure_gpa": high_target,
        "low_volume_sequence_bohr3": low_volumes.tolist(),
        "high_volume_sequence_bohr3": high_volumes.tolist(),
        "higher_pressure_has_smaller_final_volume": passed,
    }
    if not passed:
        raise AssertionError("barostat pressure direction is wrong or inconclusive")
    return result


def enrich_ipi_frames(parsed: dict, raw_stresses_kbar: list,
                      identity: dict, config: ase_validation.Config) -> None:
    rows = parsed["steps"]
    if len(raw_stresses_kbar) < len(rows):
        raise AssertionError("raw ABACUS stresses cannot be matched to i-PI steps")
    conversion = HARTREE_EV / BOHR_ANGSTROM ** 3
    for row, raw_stress in zip(rows, raw_stresses_kbar[-len(rows):]):
        cell_bohr = np.asarray(row["cell_bohr"], dtype=np.float64)
        pressure = np.asarray(
            row["virial_pressure_hartree_per_bohr3"], dtype=np.float64)
        ase_stress_full = -pressure * conversion
        ase_stress_voigt = [
            ase_stress_full[0, 0], ase_stress_full[1, 1],
            ase_stress_full[2, 2], ase_stress_full[1, 2],
            ase_stress_full[0, 2], ase_stress_full[0, 1],
        ]
        record = dict(identity)
        record.update({
            "source": "official-ipi", "backend": config.basis,
            "device": config.device, "precision": config.precision,
            "precision_settings": {
                "precision": config.precision,
                "gint_precision": "double" if config.basis == "lcao" else None,
                "socket_float": "IEEE-754 binary64",
            },
            "cell_angstrom": (cell_bohr * BOHR_ANGSTROM).tolist(),
            "volume_angstrom3": float(row["volume_bohr3"]
                                      * BOHR_ANGSTROM ** 3),
            "condition_number": float(np.linalg.cond(cell_bohr, 2)),
            "scf_converged": True, "energy_ev": row["potential_ev"],
            "forces_ev_per_angstrom": row["forces_ev_per_angstrom"],
            "raw_abacus_stress_kbar": raw_stress,
            "socket_virial_hartree": row["virial_hartree"],
            "ase_stress_ev_per_angstrom3": [
                float(value) for value in ase_stress_voigt],
            "thresholds": {
                "finite_difference": ase_validation.FD_LIMITS,
                "identical": ase_validation.IDENTICAL_LIMITS,
                "cpu_gpu_double": ase_validation.DOUBLE_LIMITS,
                "single_mixed_smoke_only": ase_validation.SMOKE_LIMITS,
            },
            "virial_provenance": "official i-PI virial_md times volume",
        })
        ase_validation.assert_real_frame_schema(record)
        row.clear()
        row.update(record)


def _ipi_executable() -> Path:
    candidate = Path(sys.executable).with_name("i-pi")
    if not candidate.is_file():
        found = shutil.which("i-pi")
        if found is None:
            raise AssertionError("i-PI 3.2.0 executable is not installed")
        candidate = Path(found)
    return candidate


def _prepare_instance(config: ase_validation.Config, mode: str, run_dir: Path,
                      steps: int, pressure_gpa: float, seed: int,
                      socket_name: str) -> tuple[Path, Path]:
    shutil.rmtree(run_dir, ignore_errors=True)
    run_dir.mkdir(parents=True)
    shutil.copy2(HERE / "init.xyz", run_dir / "init.xyz")
    prefix = "{}-{}".format(mode, socket_name)
    xml_text = render_template(HERE / (mode + ".xml"), socket_name, steps,
                               pressure_gpa, seed, prefix)
    validate_official_xml(xml_text, run_dir)
    xml_path = run_dir / "input.xml"
    xml_path.write_text(xml_text)
    abacus_dir = run_dir / "abacus"
    ase_validation.prepare_case(config, abacus_dir, socket=True)
    return xml_path, abacus_dir


def run_instance(config: ase_validation.Config, mode: str, run_dir: Path,
                 steps: int, pressure_gpa: float, seed: int) -> dict:
    socket_name = "abacus_vc_{}_{}_{}".format(mode, os.getpid(), time.time_ns())
    xml_path, abacus_dir = _prepare_instance(
        config, mode, run_dir, steps, pressure_gpa, seed, socket_name)
    processes = []
    ipi_out = open(run_dir / "ipi.stdout", "w")
    ipi_err = open(run_dir / "ipi.stderr", "w")
    abacus_out = open(run_dir / "abacus.stdout", "w")
    abacus_err = open(run_dir / "abacus.stderr", "w")
    socket_path = Path("/tmp/ipi_" + socket_name)
    try:
        ipi_process = subprocess.Popen(
            [str(_ipi_executable()), str(xml_path.name)], cwd=run_dir,
            stdout=ipi_out, stderr=ipi_err)
        processes.append(ipi_process)
        if not wait_until(lambda: socket_path.exists() or ipi_process.poll() is not None,
                          timeout=30.0):
            raise TimeoutError("i-PI UNIX socket did not appear")
        if ipi_process.poll() is not None:
            raise RuntimeError("i-PI exited before ABACUS connected")
        environment = os.environ.copy()
        environment["OMP_NUM_THREADS"] = "1"
        environment["ABACUS_SOCKET_ADDRESS"] = str(socket_path) + ":UNIX"
        abacus_process = subprocess.Popen(
            shlex.split(config.abacus), cwd=abacus_dir, env=environment,
            stdout=abacus_out, stderr=abacus_err)
        processes.append(abacus_process)
        deadline = time.monotonic() + max(300.0, 90.0 * steps)
        while ipi_process.poll() is None and time.monotonic() < deadline:
            if abacus_process.poll() is not None:
                raise RuntimeError("ABACUS exited before i-PI completed")
            time.sleep(0.1)
        if ipi_process.poll() is None:
            raise TimeoutError("i-PI trajectory exceeded bounded timeout")
        if ipi_process.returncode != 0:
            raise RuntimeError("i-PI exited with {}".format(ipi_process.returncode))
    finally:
        terminate_processes(processes)
        for stream in (ipi_out, ipi_err, abacus_out, abacus_err):
            stream.close()
    prefix = "{}-{}".format(mode, socket_name)
    parsed = parse_properties(run_dir / (prefix + ".properties"), steps)
    raw_stresses, convergence_count = (
        ase_validation.raw_stress_series_and_convergence(abacus_dir, len(parsed["steps"])))
    enrich_ipi_frames(parsed, raw_stresses, ase_validation._identity(config), config)
    parsed.update({
        "mode": mode, "target_pressure_gpa": float(pressure_gpa),
        "seed": int(seed), "requested_steps": int(steps),
        "socket_name": socket_name, "socket_path": str(socket_path),
        "zero_initial_atomic_velocity": True,
        "zero_initial_barostat_momentum": True,
        "scf_convergence_count": convergence_count,
    })
    return parsed


def run_validation(config: ase_validation.Config, mode: str, steps: int) -> dict:
    if steps not in (10, 50):
        raise AssertionError("official acceptance trajectory must be 10 or 50 steps")
    config.workdir.mkdir(parents=True, exist_ok=True)
    reference_atoms, reference = ase_validation.run_fileio_reference(
        config, config.workdir / "fileio_pressure_reference")
    stress = np.asarray(reference["ase_stress_ev_per_angstrom3"], dtype=np.float64)
    pressure_initial = -float(np.mean(stress[:3])) * GPA_PER_EV_ANGSTROM3
    seed = 314159
    low_target = pressure_initial - 2.0
    high_target = pressure_initial + 2.0
    low = run_instance(config, "isotropic", config.workdir / "probe_low",
                       5, low_target, seed)
    high = run_instance(config, "isotropic", config.workdir / "probe_high",
                        5, high_target, seed)
    direction = paired_pressure_decision(low, high)
    trajectory = run_instance(config, mode, config.workdir / "trajectory",
                              steps, pressure_initial, seed)
    payload = {
        "schema_version": 1, "ipi_version": IPI_VERSION,
        "backend": config.basis, "device": config.device,
        "precision": config.precision,
        "initial_cell_angstrom": reference_atoms.cell.array.tolist(),
        "initial_pressure_gpa_from_fileio": pressure_initial,
        "pressure_offset_gpa": 2.0,
        "paired_probe": direction, "low_probe": low, "high_probe": high,
        "trajectory": trajectory, "fileio_reference": reference,
        "comparison": {
            "same_coordinates": True, "same_cell": True,
            "same_seed": low["seed"] == high["seed"] == seed,
            "zero_barostat_momentum": True,
        },
    }
    if not payload["comparison"]["same_seed"]:
        raise AssertionError("paired probes used different random seeds")
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _self_test() -> None:
    import ipi
    assert ipi.__version__ == IPI_VERSION
    with tempfile.TemporaryDirectory(prefix="task7-ipi-selftest-") as temporary:
        directory = Path(temporary)
        shutil.copy2(HERE / "init.xyz", directory / "init.xyz")
        for mode in ("isotropic", "flexible"):
            xml = render_template(HERE / (mode + ".xml"), "unique_socket",
                                  5, 1.25, 1729, "dry-" + mode)
            assert not any(key in xml for key in TEMPLATE_KEYS)
            root = ET.fromstring(xml)
            assert root.find(".//barostat").attrib["mode"] == mode
            assert root.find(".//pressure").attrib["units"] == "gigapascal"
            simulation = validate_official_xml(xml, directory)
            system = simulation.syslist[0]
            assert np.all(np.asarray(system.beads.p) == 0.0)
            assert np.all(np.asarray(system.motion.barostat.p) == 0.0)
            if mode == "flexible":
                assert type(system.motion.barostat).__name__ == "BaroMTK"
        properties = directory / "synthetic.properties"
        forces = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
        cell6 = [5.43, 5.21, 5.57, 0.31, 0.17, 0.37]
        virial6 = [1.0, 1.1, 1.2, 0.1, 0.2, 0.3]
        lines = ["# step potential conserved forces volume cell_h virial_md"]
        for step in range(5):
            values = ([step, -10.0 + step, -9.0] + forces
                      + [157.0 - step] + cell6 + virial6)
            lines.append(" ".join(format(float(value), ".17g") for value in values))
        properties.write_text("\n".join(lines) + "\n")
        parsed = parse_properties(properties, 5)
        assert parsed["completed_steps"] == 5
        assert parsed["volume_sequence_bohr3"] == [157.0, 156.0, 155.0, 154.0, 153.0]
        assert parsed["steps"][0]["cell_bohr"] == [
            [5.43, 0.31, 0.17], [0.0, 5.21, 0.37], [0.0, 0.0, 5.57]]
        assert parsed["steps"][0]["virial_hartree"][0][1] == 15.700000000000001
        dummy_config = ase_validation.Config(
            "unused", "pw", "cpu", "double", directory,
            directory / "unused.json", directory)
        identity = {
            "executable_version": "self-test",
            "executable_sha256": "0" * 64,
            "source_commit": "self-test", "module": "self-test",
        }
        enrich_ipi_frames(parsed, [np.zeros((3, 3)).tolist()] * 5,
                          identity, dummy_config)
        first = parsed["steps"][0]
        conversion = HARTREE_EV / BOHR_ANGSTROM ** 3
        assert np.allclose(
            first["ase_stress_ev_per_angstrom3"],
            -conversion * np.array([1.0, 1.1, 1.2, 0.3, 0.2, 0.1]))
        assert first["precision_settings"]["socket_float"] == "IEEE-754 binary64"
        low = {"target_pressure_gpa": -1.0,
               "volume_sequence_bohr3": [150, 151, 152, 153, 154]}
        high = {"target_pressure_gpa": 3.0,
                "volume_sequence_bohr3": [150, 149, 148, 147, 146]}
        assert paired_pressure_decision(low, high)[
            "higher_pressure_has_smaller_final_volume"]
        wrong = {"target_pressure_gpa": 3.0,
                 "volume_sequence_bohr3": [150, 151, 152, 153, 154]}
        try:
            paired_pressure_decision(low, wrong)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong pressure/volume direction was accepted")
        processes = []
        marker = directory / "marker"
        try:
            process = subprocess.Popen([
                sys.executable, "-c",
                "import pathlib,time; pathlib.Path(r'{}').write_text('ready'); time.sleep(60)"
                .format(marker)])
            processes.append(process)
            assert wait_until(marker.exists, timeout=2.0)
            assert not wait_until(lambda: False, timeout=0.05, interval=0.01)
        finally:
            terminate_processes(processes)
        assert process.poll() is not None
        assert not Path("/tmp/ipi_unique_socket").exists()
    print("run_validation self-test: PASS (official i-PI {})".format(IPI_VERSION))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abacus", default="abacus")
    parser.add_argument("--basis", choices=("pw", "lcao"), default="pw")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--precision", choices=("double", "single"), default="double")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path("variable-cell-ipi"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pp-orb-root", type=Path,
                        default=Path(__file__).resolve().parents[4] / "tests" / "PP_ORB")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--mode", choices=("isotropic", "flexible"),
                        default="isotropic")
    parser.add_argument("--steps", type=int, choices=(10, 50), default=10)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _self_test()
        return
    output = args.output or args.workdir / "ipi-validation.json"
    config = ase_validation.Config(
        args.abacus, args.basis, args.device, args.precision,
        args.workdir.resolve(), output.resolve(), args.pp_orb_root.resolve())
    if not config.pp_orb_root.is_dir():
        raise SystemExit("--pp-orb-root does not exist: {}".format(
            config.pp_orb_root))
    if args.prepare_only:
        _prepare_instance(config, args.mode, config.workdir / "prepared",
                          args.steps, 0.0, 314159, "abacus_vc_prepare")
        print("prepared parser-validated {} i-PI/ABACUS case in {}".format(
            args.mode, config.workdir))
        return
    run_validation(config, args.mode, args.steps)
    print("wrote {}".format(config.output))


if __name__ == "__main__":
    main()
