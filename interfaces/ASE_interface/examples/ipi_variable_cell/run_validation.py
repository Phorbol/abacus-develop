#!/usr/bin/env python3
"""Run pinned i-PI 3.2.0 isotropic/flexible NPT validation with ABACUS."""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import signal
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
TEMPLATE_KEYS = ("__SOCKET_NAME__", "__TOTAL_STEPS__", "__PRESSURE_GPA__",
                 "__SEED__", "__PREFIX__")
EXPECTED_IPI_PROPERTIES = (
    "step", "potential{electronvolt}", "conserved{electronvolt}",
    "atom_f{ev/ang}(0)",
    "atom_f{ev/ang}(1)",
    "volume", "cell_h", "virial_md",
)
IPI_VOLUME_LIMITS = {"rtol": 1.0e-10, "atol_bohr3": 1.0e-8}
PRESSURE_DIRECTION_LIMITS = {"rtol": 1.0e-8, "atol_bohr3": 1.0e-8}
IPI_STABILITY_LIMITS = {
    "stale_rtol": 1.0e-10, "stale_atol": 1.0e-12,
    "max_volume_ratio": 1.25,
    "conserved_rtol": 5.0e-3, "conserved_atol_ev": 5.0e-2,
    "flexible_shear_rtol": 1.0e-10, "flexible_shear_atol_bohr": 1.0e-12,
}


def assert_json_ready(value, path: str = "$") -> None:
    """Require a recursively finite, built-in-only JSON payload."""
    if isinstance(value, (np.ndarray, np.generic)):
        raise AssertionError("{} contains a NumPy value".format(path))
    if value is None or isinstance(value, (bool, int, str)):
        return
    if isinstance(value, float):
        if not np.isfinite(value):
            raise AssertionError("{} contains a non-finite float".format(path))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            assert_json_ready(item, "{}[{}]".format(path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise AssertionError("{} contains a non-string key".format(path))
            assert_json_ready(item, "{}.{}".format(path, key))
        return
    raise AssertionError("{} contains unsupported type {}".format(
        path, type(value).__name__))


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
    from ipi.engine.outputs import PropertyOutput
    from ipi.engine.properties import getkey
    from ipi.engine.simulation import Simulation
    if ipi.__version__ != IPI_VERSION:
        raise AssertionError("requires official i-PI=={}".format(IPI_VERSION))
    with working_directory(directory):
        simulation = Simulation.load_from_xml(
            io.StringIO(xml_text), read_only=True, request_banner=False)
    if simulation is None:
        raise AssertionError("official i-PI parser returned no simulation")
    property_outputs = [
        output for output in simulation.outtemplate
        if isinstance(output, PropertyOutput)]
    if len(simulation.syslist) != 1 or len(property_outputs) != 1:
        raise AssertionError(
            "official i-PI output contract requires one system and one property output")
    actual = tuple(str(item) for item in property_outputs[0].outlist)
    if actual != EXPECTED_IPI_PROPERTIES:
        raise AssertionError("official i-PI property output contract is wrong")
    system = simulation.syslist[0]
    with tempfile.TemporaryDirectory(
            prefix="ipi-property-output-", dir=directory) as temporary:
        disposable = Path(temporary) / "properties"
        output = PropertyOutput(filename=str(disposable), outlist=actual)
        try:
            output.bind(system)
            output.print_header()
            columns = 0
            for item in actual:
                declared = system.properties.property_dict[getkey(item)]
                size = declared.get("size", 1)
                columns += 1 if size == 1 else size
            if columns != 22:
                raise AssertionError(
                    "official i-PI property output must declare 22 columns")
        except (KeyError, ValueError, RuntimeError) as error:
            raise AssertionError(
                "official i-PI property output cannot bind or print its header") from error
        finally:
            try:
                output.close_stream()
            finally:
                try:
                    disposable.unlink()
                except FileNotFoundError:
                    pass
    return simulation


def wait_until(predicate, timeout: float, interval: float = 0.02) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return bool(predicate())


def abacus_exit_state(returncode, zero_exit_at,
                      now: float, shutdown_grace: float):
    if returncode is None:
        return "running", zero_exit_at
    if returncode != 0:
        return "nonzero", zero_exit_at
    if zero_exit_at is None:
        zero_exit_at = now
    if now - zero_exit_at >= shutdown_grace:
        return "grace-timeout", zero_exit_at
    return "grace", zero_exit_at


def wait_for_ipi_shutdown(ipi_process, abacus_process,
                          trajectory_timeout: float = 900.0,
                          shutdown_grace: float = 15.0,
                          poll_interval: float = 0.1,
                          monotonic=time.monotonic,
                          sleeper=time.sleep) -> int:
    deadline = monotonic() + trajectory_timeout
    zero_exit_at = None
    while True:
        ipi_returncode = ipi_process.poll()
        if ipi_returncode is not None:
            return ipi_returncode
        now = monotonic()
        abacus_returncode = abacus_process.poll()
        state, zero_exit_at = abacus_exit_state(
            abacus_returncode, zero_exit_at, now, shutdown_grace)
        if state == "nonzero":
            raise RuntimeError(
                "ABACUS exited with {}".format(abacus_returncode))
        if state == "grace-timeout":
            raise TimeoutError(
                "i-PI did not exit within {} seconds after ABACUS rc=0".format(
                    shutdown_grace))
        if state == "running" and now >= deadline:
            raise TimeoutError(
                "i-PI trajectory exceeded {} seconds".format(
                    trajectory_timeout))
        sleeper(poll_interval)


def start_managed_process(argv, **kwargs) -> subprocess.Popen:
    process = subprocess.Popen(argv, start_new_session=True, **kwargs)
    process.task7_pgid = process.pid
    return process


def terminate_processes(processes: list[subprocess.Popen]) -> None:
    """TERM then KILL complete process groups with bounded waits."""
    for process in reversed(processes):
        pgid = getattr(process, "task7_pgid", None)
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            elif process.poll() is None:
                process.terminate()
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 3.0
    for process in reversed(processes):
        try:
            process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pgid = getattr(process, "task7_pgid", None)
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    process.kill()
            except ProcessLookupError:
                pass
            process.wait(timeout=2.0)


def owned_socket_path(socket_name: str) -> Path:
    if re.fullmatch(r"abacus_vc_[A-Za-z0-9_]+", socket_name) is None:
        raise AssertionError("refusing unsafe/unowned socket name")
    path = Path("/tmp") / ("ipi_" + socket_name)
    if path.parent != Path("/tmp") or path.name != "ipi_" + socket_name:
        raise AssertionError("socket path escaped ownership boundary")
    return path


def unlink_owned_socket(socket_name: str) -> None:
    path = owned_socket_path(socket_name)
    try:
        if path.is_socket() or path.is_file():
            path.unlink()
    except FileNotFoundError:
        pass


def parse_properties(path: Path, completed_steps: int) -> dict:
    """Parse official six-component cell_h/virial_md plus two-atom forces."""
    if (isinstance(completed_steps, bool) or not isinstance(completed_steps, int)
            or completed_steps < 0):
        raise AssertionError("completed i-PI step count must be a nonnegative integer")
    if not path.is_file():
        raise AssertionError("i-PI properties output is absent: {}".format(path))
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        try:
            values = [float(value) for value in line.split()]
        except ValueError as error:
            raise AssertionError(
                "non-numeric i-PI properties row {}".format(line_number)) from error
        if len(values) != 22:
            raise AssertionError("i-PI properties row lacks cell/virial columns")
        if not np.all(np.isfinite(values)):
            raise AssertionError("i-PI properties contain non-finite values")
        if not values[0].is_integer():
            raise AssertionError("i-PI property step must be an integer")
        step = int(values[0])
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
        volume_tolerance = (IPI_VOLUME_LIMITS["atol_bohr3"]
                            + IPI_VOLUME_LIMITS["rtol"] * abs(determinant))
        if volume <= 0.0 or abs(volume - determinant) > volume_tolerance:
            raise AssertionError("i-PI volume disagrees with det(cell_h)")
        rows.append({
            "step": step, "potential_ev": values[1],
            "conserved_ev": values[2], "forces_ev_per_angstrom": forces.tolist(),
            "volume_bohr3": volume,
            "cell_bohr": cell.tolist(), "condition_number": condition,
            "virial_pressure_hartree_per_bohr3": virial_pressure.tolist(),
            "virial_hartree": (virial_pressure * volume).tolist(),
        })
    expected_sequence = list(range(completed_steps + 1))
    actual_sequence = [row["step"] for row in rows]
    if actual_sequence != expected_sequence:
        raise AssertionError(
            "i-PI property steps must be exactly 0..{}; got {}".format(
                completed_steps, actual_sequence))
    return {"steps": rows, "completed_steps": completed_steps,
            "sample_count": len(rows), "includes_initial_frame": True,
            "volume_sequence_bohr3": [row["volume_bohr3"] for row in rows]}


def paired_pressure_decision(low: dict, high: dict) -> dict:
    low_target = float(low["target_pressure_gpa"])
    high_target = float(high["target_pressure_gpa"])
    low_volumes = np.asarray(low["volume_sequence_bohr3"], dtype=np.float64)
    high_volumes = np.asarray(high["volume_sequence_bohr3"], dtype=np.float64)
    for replica in (low, high):
        if (replica.get("completed_steps") != 5
                or replica.get("sample_count") != 6
                or replica.get("includes_initial_frame") is not True):
            raise AssertionError(
                "paired pressure probe requires initial frame plus five MD steps")
    if low_volumes.shape != (6,) or high_volumes.shape != (6,):
        raise AssertionError("paired pressure probe requires exactly six samples")
    if not np.all(np.isfinite(low_volumes)) or not np.all(np.isfinite(high_volumes)):
        raise AssertionError("paired pressure volumes must be finite")
    scale = max(float(np.max(np.abs(low_volumes))),
                float(np.max(np.abs(high_volumes))))
    tolerance = (PRESSURE_DIRECTION_LIMITS["atol_bohr3"]
                 + PRESSURE_DIRECTION_LIMITS["rtol"] * scale)
    differences = high_volumes - low_volumes
    changes_from_initial = differences - differences[0]
    trend = float(np.mean(changes_from_initial[-2:])
                  - np.mean(changes_from_initial[:2]))
    passed = bool(high_target > low_target
                  and abs(differences[0]) <= tolerance
                  and changes_from_initial[-1] < -tolerance
                  and trend < -tolerance)
    result = {
        "low_target_pressure_gpa": low_target,
        "high_target_pressure_gpa": high_target,
        "low_volume_sequence_bohr3": low_volumes.tolist(),
        "high_volume_sequence_bohr3": high_volumes.tolist(),
        "higher_pressure_has_smaller_final_volume": passed,
        "paired_high_minus_low_volume_bohr3": differences.tolist(),
        "paired_high_minus_low_change_from_initial_bohr3":
            changes_from_initial.tolist(),
        "initial_high_minus_low_volume_bohr3": float(differences[0]),
        "final_high_minus_low_change_from_initial_bohr3":
            float(changes_from_initial[-1]),
        "trend_last_two_minus_first_two_change_from_initial_bohr3": trend,
        "comparison_basis": (
            "six samples: shared initial frame plus five completed MD steps; "
            "final and trend are high-minus-low changes relative to sample 0"),
        "atol_plus_rtol_bohr3": tolerance,
        "thresholds": PRESSURE_DIRECTION_LIMITS,
    }
    if not passed:
        raise AssertionError("barostat pressure direction is wrong or inconclusive")
    return result


def pressure_offset_significance(stress_ev_per_angstrom3,
                                 offset_gpa: float,
                                 config: ase_validation.Config) -> dict:
    """Require the pressure split to exceed propagated stress uncertainty."""
    stress = np.asarray(stress_ev_per_angstrom3, dtype=np.float64)
    if stress.shape != (6,) or not np.all(np.isfinite(stress)):
        raise AssertionError("pressure reference stress must have six finite values")
    limits = ase_validation.active_stress_limits(
        config, ase_validation.IDENTICAL_LIMITS)
    component_tolerance = (limits["atol_ev_per_angstrom3"]
                           + limits["rtol"] * np.abs(stress[:3]))
    pressure_uncertainty = (float(np.mean(component_tolerance))
                            * GPA_PER_EV_ANGSTROM3)
    target_separation = 2.0 * float(offset_gpa)
    required_separation = 2.0 * pressure_uncertainty
    passed = bool(np.isfinite(offset_gpa) and offset_gpa > 0.0
                  and target_separation > required_separation)
    result = {
        "pass": passed,
        "precision": config.precision,
        "active_stress_thresholds": limits,
        "hydrostatic_component_atol_plus_rtol_ev_per_angstrom3":
            component_tolerance.tolist(),
        "propagated_pressure_uncertainty_gpa": pressure_uncertainty,
        "target_pressure_separation_gpa": target_separation,
        "required_separation_gpa": required_separation,
    }
    if not passed:
        raise AssertionError("paired pressure offset is below stress uncertainty")
    return result


def enrich_ipi_frames(parsed: dict, raw_frames: list,
                      identity: dict, config: ase_validation.Config) -> None:
    from ase import units
    from ase.stress import full_3x3_to_voigt_6_stress
    rows = parsed["steps"]
    if len(raw_frames) != len(rows):
        raise AssertionError("raw ABACUS stresses cannot be matched to i-PI steps")
    active = ase_validation.active_stress_limits(
        config, ase_validation.IDENTICAL_LIMITS)
    for row, raw_frame in zip(rows, raw_frames):
        cell_bohr = np.asarray(row["cell_bohr"], dtype=np.float64)
        raw_stress = np.asarray(
            raw_frame["raw_abacus_stress_kbar"], dtype=np.float64)
        ase_stress_voigt = (
            -0.1 * units.GPa * full_3x3_to_voigt_6_stress(raw_stress))
        record = dict(identity)
        record.update({
            "source": "official-ipi", "backend": config.basis,
            "device": config.device, "precision": config.precision,
            "precision_settings": {
                "precision": config.precision,
                "gint_precision": (None if config.basis == "pw" else
                                   ("double" if config.precision == "double" else "mix")),
                "socket_float": "IEEE-754 binary64",
            },
            "cell_angstrom": (cell_bohr * BOHR_ANGSTROM).tolist(),
            "volume_angstrom3": float(row["volume_bohr3"]
                                      * BOHR_ANGSTROM ** 3),
            "condition_number": float(np.linalg.cond(cell_bohr, 2)),
            "scf_converged": raw_frame["scf_converged"],
            "energy_ev": row["potential_ev"],
            "ipi_property_step": row["step"],
            "is_initial_frame": row["step"] == 0,
            "atom_count": len(row["forces_ev_per_angstrom"]),
            "forces_ev_per_angstrom": row["forces_ev_per_angstrom"],
            "raw_abacus_stress_kbar": raw_stress.tolist(),
            "socket_virial_hartree": row["virial_hartree"],
            "ase_stress_ev_per_angstrom3": [
                float(value) for value in ase_stress_voigt],
            "thresholds": {
                "finite_difference": ase_validation.FD_LIMITS,
                "identical": ase_validation.IDENTICAL_LIMITS,
                "cpu_gpu_double": ase_validation.DOUBLE_LIMITS,
                "single_mixed_smoke_only": ase_validation.SMOKE_LIMITS,
                "active_stress": active,
                "volume": ase_validation.VOLUME_LIMITS,
            },
            "virial_provenance": "official i-PI virial_md times volume",
        })
        ase_validation.assert_real_frame_schema(record)
        row.clear()
        row.update(record)


def evaluate_ipi_stability(parsed: dict, mode: str, checkpoint: Path) -> dict:
    rows = parsed["steps"]
    cells = np.asarray([row["cell_bohr"] for row in rows], dtype=np.float64)
    virials = np.asarray([row["virial_hartree"] for row in rows], dtype=np.float64)
    volumes = np.asarray([row["volume_bohr3"] for row in rows], dtype=np.float64)
    conserved = np.asarray([row["conserved_ev"] for row in rows], dtype=np.float64)
    limits = IPI_STABILITY_LIMITS
    cell_change = float(np.max(np.abs(cells - cells[0])))
    virial_change = float(np.max(np.abs(virials - virials[0])))
    cell_stale_tolerance = float(
        limits["stale_atol"] + limits["stale_rtol"] * np.max(np.abs(cells)))
    virial_stale_tolerance = float(
        limits["stale_atol"] + limits["stale_rtol"] * np.max(np.abs(virials)))
    ratio = float(np.max(volumes) / np.min(volumes))
    drift = float(np.max(np.abs(conserved - conserved[0])))
    drift_tolerance = (limits["conserved_atol_ev"] + limits["conserved_rtol"]
                       * max(1.0, float(np.max(np.abs(conserved)))))
    shear = cells[:, (0, 0, 1), (1, 2, 2)]
    shear_change = float(np.max(np.abs(shear - shear[0])))
    shear_tolerance = (limits["flexible_shear_atol_bohr"]
                       + limits["flexible_shear_rtol"] * max(1.0, float(np.max(np.abs(shear)))))
    decisions = {
        "cell_not_stale": cell_change > cell_stale_tolerance,
        "virial_not_stale": virial_change > virial_stale_tolerance,
        "volume_bounded": ratio <= limits["max_volume_ratio"],
        "conserved_drift_ok": drift <= drift_tolerance,
        "flexible_shear_changed": (True if mode != "flexible"
                                   else shear_change > shear_tolerance),
        "checkpoint_nonempty": checkpoint.is_file() and checkpoint.stat().st_size > 0,
    }
    decisions = {
        name: bool(passed) for name, passed in decisions.items()}
    result = {
        "thresholds": limits, "decisions": decisions,
        "cell_max_change_bohr": cell_change, "virial_max_change_hartree": virial_change,
        "volume_ratio": ratio, "conserved_max_drift_ev": drift,
        "conserved_atol_plus_rtol_ev": drift_tolerance,
        "flexible_shear_max_change_bohr": shear_change,
        "flexible_shear_atol_plus_rtol_bohr": shear_tolerance,
    }
    failed = [name for name, passed in decisions.items() if not passed]
    if failed:
        raise AssertionError("i-PI short-run stability failed: " + ",".join(failed))
    return result


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


def build_ipi_prepare_manifest(config: ase_validation.Config, mode: str,
                               steps: int, xml_path: Path,
                               abacus_dir: Path) -> dict:
    manifest = ase_validation.build_prepare_manifest(
        config,
        {"rendered_xml": xml_path, "abacus": abacus_dir},
        "ipi-variable-cell",
        extra={
            "official_ipi": {
                "version": IPI_VERSION,
                "parser": "Simulation.load_from_xml(read_only=True)",
                "mode": mode,
                "steps": int(steps),
            },
            "xml_template": str((HERE / (mode + ".xml")).resolve()),
        },
    )
    assert_ipi_prepare_manifest(manifest)
    return manifest


def assert_ipi_prepare_manifest(manifest: dict) -> None:
    ase_validation.assert_prepare_manifest(manifest)
    if set(manifest["cases"]) != {"rendered_xml", "abacus"}:
        raise AssertionError("i-PI prepare manifest has unexpected cases")
    metadata = manifest["official_ipi"]
    if (not isinstance(metadata, dict)
            or metadata.get("version") != IPI_VERSION
            or metadata.get("mode") not in ("isotropic", "flexible")
            or metadata.get("steps") not in (10, 50)
            or metadata.get("parser") != "Simulation.load_from_xml(read_only=True)"):
        raise AssertionError("invalid official i-PI prepare metadata")
    if (not isinstance(manifest.get("xml_template"), str)
            or not Path(manifest["xml_template"]).is_absolute()):
        raise AssertionError("i-PI XML template path must be absolute")


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
    socket_path = owned_socket_path(socket_name)
    try:
        ipi_process = start_managed_process(
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
        abacus_process = start_managed_process(
            shlex.split(config.abacus), cwd=abacus_dir, env=environment,
            stdout=abacus_out, stderr=abacus_err)
        processes.append(abacus_process)
        ipi_returncode = wait_for_ipi_shutdown(
            ipi_process, abacus_process,
            trajectory_timeout=max(300.0, 90.0 * steps))
        if ipi_returncode != 0:
            raise RuntimeError("i-PI exited with {}".format(ipi_returncode))
        try:
            abacus_process.wait(timeout=10.0)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("ABACUS did not exit after normal i-PI completion") from error
        if abacus_process.returncode != 0:
            raise RuntimeError("ABACUS exited with {}".format(abacus_process.returncode))
    finally:
        terminate_processes(processes)
        unlink_owned_socket(socket_name)
        for stream in (ipi_out, ipi_err, abacus_out, abacus_err):
            stream.close()
    prefix = "{}-{}".format(mode, socket_name)
    parsed = parse_properties(run_dir / (prefix + ".properties"), steps)
    stability = evaluate_ipi_stability(
        parsed, mode, run_dir / (prefix + ".checkpoint"))
    raw_frames = ase_validation.raw_frame_series(abacus_dir, parsed["sample_count"])
    enrich_ipi_frames(parsed, raw_frames, ase_validation._identity(config), config)
    parsed.update({
        "mpi_command": config.abacus,
        "mpi_ranks": ase_validation.infer_mpi_ranks(config.abacus),
        "requested_ks_solver": config.ks_solver,
        "effective_ks_solver": ase_validation.effective_ks_solver(
            abacus_dir, config.ks_solver),
        "mode": mode, "target_pressure_gpa": float(pressure_gpa),
        "seed": int(seed), "requested_steps": int(steps),
        "socket_name": socket_name, "socket_path": str(socket_path),
        "zero_initial_atomic_velocity": True,
        "zero_initial_barostat_momentum": True,
        "scf_convergence_count": len(raw_frames), "stability": stability,
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
    offset_significance = pressure_offset_significance(stress, 2.0, config)
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
        "mpi_command": config.abacus,
        "mpi_ranks": ase_validation.infer_mpi_ranks(config.abacus),
        "requested_ks_solver": config.ks_solver,
        "effective_ks_solver": ase_validation.effective_ks_solver(
            config.workdir / "fileio_pressure_reference", config.ks_solver),
        "initial_cell_angstrom": reference_atoms.cell.array.tolist(),
        "initial_pressure_gpa_from_fileio": pressure_initial,
        "pressure_offset_gpa": 2.0,
        "pressure_offset_significance": offset_significance,
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
    assert_json_ready(payload)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _self_test() -> None:
    import ipi
    from ipi.engine.outputs import PropertyOutput
    from ipi.engine.simulation import Simulation
    from ipi.utils.units import unit_to_user
    assert ipi.__version__ == IPI_VERSION
    with tempfile.TemporaryDirectory(prefix="task7-ipi-selftest-") as temporary:
        directory = Path(temporary)
        shutil.copy2(HERE / "init.xyz", directory / "init.xyz")
        expected = (
            "step", "potential{electronvolt}", "conserved{electronvolt}",
            "atom_f{ev/ang}(0)",
            "atom_f{ev/ang}(1)",
            "volume", "cell_h", "virial_md",
        )
        rendered = {}
        for mode in ("isotropic", "flexible"):
            xml = render_template(HERE / (mode + ".xml"), "unique_socket",
                                  5, 1.25, 1729, "dry-" + mode)
            rendered[mode] = xml
            assert not any(key in xml for key in TEMPLATE_KEYS)
            root = ET.fromstring(xml)
            assert root.attrib["floatformat"] == "%24.16e"
            assert root.find(".//barostat").attrib["mode"] == mode
            assert root.find(".//pressure").attrib["units"] == "gigapascal"
            simulation = validate_official_xml(xml, directory)
            property_outputs = [
                output for output in simulation.outtemplate
                if isinstance(output, PropertyOutput)]
            assert len(property_outputs) == 1
            assert tuple(str(item) for item in property_outputs[0].outlist) == expected
            system = simulation.syslist[0]
            assert np.all(np.asarray(system.beads.p) == 0.0)
            assert np.all(np.asarray(system.motion.barostat.p) == 0.0)
            if mode == "flexible":
                assert type(system.motion.barostat).__name__ == "BaroMTK"
        unrecognized = rendered["isotropic"].replace(
            "atom_f{ev/ang}(0)", "not_a_property")
        try:
            validate_official_xml(unrecognized, directory)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                "unrecognized i-PI output property was accepted")
        assert np.all(np.isfinite(unit_to_user(
            "force", "ev/ang", np.asarray([1.0], dtype=np.float64))))
        assert np.all(np.isfinite(unit_to_user(
            "energy", "electronvolt", np.asarray([1.0], dtype=np.float64))))
        try:
            unit_to_user("force", "electronvolt/angstrom",
                         np.asarray([1.0], dtype=np.float64))
        except TypeError:
            pass
        else:
            raise AssertionError("invalid long-form i-PI force unit was accepted")
        official_samples = []
        official_md_steps = []
        class DummyCheckpoint:
            def store(self):
                pass
        class DummyOutput:
            def active(self):
                return True
            def write(self):
                official_samples.append(dummy_simulation.step + 1)
        class DummySimulation:
            step = 0
            tsteps = 5
            threading = False
            safe_stride = 1000
            ttime = 0
            rollback = True
            chk = DummyCheckpoint()
            outputs = [DummyOutput()]
            def run_step(self, step):
                official_md_steps.append(step)
        dummy_simulation = DummySimulation()
        Simulation.run(dummy_simulation)
        assert official_samples == list(range(6))
        assert official_md_steps == list(range(5))
        print("official i-PI output semantics probe: "
              "5 completed MD steps -> property samples 0..5")
        properties = directory / "synthetic.properties"
        forces = [0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
        base_cell6 = np.array([5.43, 5.21, 5.57, 0.31, 0.17, 0.37])
        base_virial6 = np.array([1.0, 1.1, 1.2, 0.1, 0.2, 0.3])
        lines = ["# step potential conserved forces volume cell_h virial_md"]
        for step in range(6):
            cell6 = base_cell6.copy()
            cell6[:3] *= 1.0 - 1.0e-4 * step
            cell6[3:] += step * np.array([2.0e-4, -1.0e-4, 3.0e-4])
            volume = float(np.prod(cell6[:3]))
            virial6 = base_virial6 + step * 1.0e-3
            values = ([step, -10.0 + step, -9.0] + forces
                      + [volume] + cell6.tolist() + virial6.tolist())
            lines.append(" ".join(format(float(value), ".17g") for value in values))
        properties.write_text("\n".join(lines) + "\n")
        parsed = parse_properties(properties, 5)
        assert parsed["completed_steps"] == 5
        assert parsed["sample_count"] == 6
        assert parsed["includes_initial_frame"] is True
        assert [row["step"] for row in parsed["steps"]] == list(range(6))
        def reject_properties(label, mutated_lines):
            mutated = directory / (label + ".properties")
            mutated.write_text("\n".join(mutated_lines) + "\n")
            try:
                parse_properties(mutated, 5)
            except AssertionError:
                pass
            else:
                raise AssertionError(label + " properties mutation was accepted")
        reject_properties("n-lines", lines[:-1])
        extra_lines = list(lines)
        extra_values = extra_lines[-1].split()
        extra_values[0] = "6"
        extra_lines.append(" ".join(extra_values))
        reject_properties("n-plus-two-lines", extra_lines)
        reject_properties("garbage-line", lines + ["not numeric property data"])
        duplicate_lines = list(lines)
        duplicate_values = duplicate_lines[3].split()
        duplicate_values[0] = "1"
        duplicate_lines[3] = " ".join(duplicate_values)
        reject_properties("duplicate-step", duplicate_lines)
        noninteger_lines = list(lines)
        noninteger_values = noninteger_lines[3].split()
        noninteger_values[0] = "1.5"
        noninteger_lines[3] = " ".join(noninteger_values)
        reject_properties("noninteger-step", noninteger_lines)
        out_of_order_lines = list(lines)
        out_of_order_lines[2], out_of_order_lines[3] = (
            out_of_order_lines[3], out_of_order_lines[2])
        reject_properties("out-of-order-step", out_of_order_lines)
        assert parsed["steps"][0]["cell_bohr"] == [
            [5.43, 0.31, 0.17], [0.0, 5.21, 0.37], [0.0, 0.0, 5.57]]
        checkpoint = directory / "synthetic.checkpoint"
        checkpoint.write_text("checkpoint")
        stability_results = {
            mode: evaluate_ipi_stability(parsed, mode, checkpoint)
            for mode in ("isotropic", "flexible")
        }
        for mode, stability in stability_results.items():
            assert_json_ready(stability, "$.{}_stability".format(mode))
            assert all(type(value) is bool
                       for value in stability["decisions"].values())
        assert stability_results["flexible"]["decisions"][
            "flexible_shear_changed"]
        parsed["stability"] = stability_results["flexible"]
        dummy_config = ase_validation.Config(
            "unused", "pw", "cpu", "double", directory,
            directory / "unused.json", directory)
        single_config = ase_validation.Config(
            "unused", "pw", "gpu", "single", directory,
            directory / "unused-single.json", directory)
        identity = {
            "executable_version": "self-test",
            "executable_sha256": "0" * 64,
            "source_commit": "a" * 40, "module": "self-test",
        }
        from ase import units
        raw_frames = []
        for row in parsed["steps"]:
            volume_ang3 = row["volume_bohr3"] * BOHR_ANGSTROM ** 3
            virial = np.asarray(row["virial_hartree"], dtype=np.float64)
            stress_full = -virial * units.Ha / volume_ang3
            raw = -stress_full / (0.1 * units.GPa)
            raw_frames.append({"scf_converged": True,
                               "raw_abacus_stress_kbar": raw.tolist()})
        enrich_ipi_frames(parsed, raw_frames, identity, dummy_config)
        serialization_payload = {
            "schema_version": 1,
            "ipi_version": IPI_VERSION,
            "backend": dummy_config.basis,
            "device": dummy_config.device,
            "precision": dummy_config.precision,
            "initial_cell_angstrom": parsed["steps"][0]["cell_angstrom"],
            "initial_pressure_gpa_from_fileio": 0.0,
            "pressure_offset_gpa": 2.0,
            "pressure_offset_significance": {
                "pass": True, "active_stress_thresholds":
                    ase_validation.IDENTICAL_LIMITS,
            },
            "paired_probe": {"pass": True},
            "low_probe": parsed,
            "high_probe": parsed,
            "trajectory": parsed,
            "fileio_reference": parsed["steps"][0],
            "comparison": {
                "same_coordinates": True, "same_cell": True,
                "same_seed": True, "zero_barostat_momentum": True,
            },
        }
        assert_json_ready(serialization_payload)
        json.dumps(serialization_payload, allow_nan=False)
        json_ready_target = parsed["steps"][0]
        for label, invalid_value in (
                ("ndarray", np.zeros(1)),
                ("numpy-scalar", np.float64(0.0)),
                ("nan", float("nan")),
                ("positive-inf", float("inf")),
                ("negative-inf", float("-inf"))):
            json_ready_target["json_ready_mutation"] = invalid_value
            try:
                assert_json_ready(serialization_payload)
            except AssertionError:
                pass
            else:
                raise AssertionError(label + " JSON mutation was accepted")
            finally:
                del json_ready_target["json_ready_mutation"]
        print("official i-PI JSON serialization probe: enriched payload PASS; "
              "ndarray/NumPy-scalar/NaN/Inf mutations rejected")
        first = parsed["steps"][0]
        assert np.asarray(first["ase_stress_ev_per_angstrom3"]).shape == (6,)
        assert first["precision_settings"]["socket_float"] == "IEEE-754 binary64"
        probe_metadata = {"completed_steps": 5, "sample_count": 6,
                          "includes_initial_frame": True}
        low = dict(probe_metadata, target_pressure_gpa=-1.0,
                   volume_sequence_bohr3=[150, 151, 152, 153, 154, 155])
        high = dict(probe_metadata, target_pressure_gpa=3.0,
                    volume_sequence_bohr3=[150, 149, 148, 147, 146, 145])
        assert paired_pressure_decision(low, high)[
            "higher_pressure_has_smaller_final_volume"]
        smoke_significance = pressure_offset_significance(
            np.zeros(6), 2.0, single_config)
        assert smoke_significance["active_stress_thresholds"] == (
            ase_validation.SMOKE_LIMITS)
        try:
            pressure_offset_significance(np.zeros(6), 0.01, single_config)
        except AssertionError:
            pass
        else:
            raise AssertionError("pressure offset below single smoke noise was accepted")
        wrong = dict(probe_metadata, target_pressure_gpa=3.0,
                     volume_sequence_bohr3=[150, 151, 152, 153, 154, 155])
        try:
            paired_pressure_decision(low, wrong)
        except AssertionError:
            pass
        else:
            raise AssertionError("wrong pressure/volume direction was accepted")
        insignificant = dict(
            probe_metadata, target_pressure_gpa=3.0,
            volume_sequence_bohr3=[150, 150, 150, 150, 150, 150 - 1.0e-13])
        low_flat = dict(
            probe_metadata, target_pressure_gpa=-1.0,
            volume_sequence_bohr3=[150, 150, 150, 150, 150, 150])
        try:
            paired_pressure_decision(low_flat, insignificant)
        except AssertionError:
            pass
        else:
            raise AssertionError("insignificant pressure response was accepted")
        # Exact convergence/stress pairing rejects missing and extra frames.
        logdir = directory / "log" / "OUT.TEST"
        logdir.mkdir(parents=True)
        log = logdir / "running_scf.log"
        block = "#SCF IS CONVERGED#\nTOTAL-STRESS (KBAR)\n1 0 0\n0 1 0\n0 0 1\n"
        log.write_text(block * 6)
        assert len(ase_validation.raw_frame_series(logdir.parent, 6)) == 6
        for count in (5, 7):
            try:
                ase_validation.raw_frame_series(logdir.parent, count)
            except AssertionError:
                pass
            else:
                raise AssertionError("nonexact raw-log frame count was accepted")
        # Stability mutations: stale, explosion, drift, no flexible shear,
        # missing checkpoint, and volume/cell mismatch must fail.
        def raw_parsed():
            return parse_properties(properties, 5)
        mutations = []
        stale = raw_parsed()
        for row in stale["steps"][1:]:
            row["cell_bohr"] = stale["steps"][0]["cell_bohr"]
            row["virial_hartree"] = stale["steps"][0]["virial_hartree"]
            row["volume_bohr3"] = stale["steps"][0]["volume_bohr3"]
        mutations.append(("stale", stale, "isotropic", checkpoint))
        explosion = raw_parsed()
        explosion["steps"][-1]["volume_bohr3"] *= 2
        mutations.append(("explosion", explosion, "isotropic", checkpoint))
        drifted = raw_parsed()
        drifted["steps"][-1]["conserved_ev"] += 10
        mutations.append(("drift", drifted, "isotropic", checkpoint))
        no_shear = raw_parsed()
        for row in no_shear["steps"]:
            cell = np.asarray(row["cell_bohr"])
            cell[0, 1], cell[0, 2], cell[1, 2] = 0.31, 0.17, 0.37
            row["cell_bohr"] = cell.tolist()
        mutations.append(("no-shear", no_shear, "flexible", checkpoint))
        mutations.append(("missing-checkpoint", raw_parsed(), "isotropic",
                          directory / "absent.checkpoint"))
        for label, data, mode, check in mutations:
            try:
                evaluate_ipi_stability(data, mode, check)
            except AssertionError:
                pass
            else:
                raise AssertionError(label + " stability mutation was accepted")
        mismatched = directory / "mismatched.properties"
        mismatch_lines = properties.read_text().splitlines()
        mismatch_values = mismatch_lines[1].split()
        mismatch_values[9] = "999"
        mismatch_lines[1] = " ".join(mismatch_values)
        mismatched.write_text("\n".join(mismatch_lines) + "\n")
        try:
            parse_properties(mismatched, 5)
        except AssertionError:
            pass
        else:
            raise AssertionError("volume/determinant mismatch was accepted")
        prepare_manifest = {
            "schema_version": 1, "kind": "ipi-variable-cell",
            "backend": "pw", "device": "cpu", "precision": "double",
            "gint_precision": None, "is_reference": True,
            "cases": {"rendered_xml": str(properties.resolve()),
                      "abacus": str(directory.resolve())},
            "resolved_files": {
                "pseudopotential": str(properties.resolve()), "orbital": None},
            "socket_variable_cell": True, "cal_stress": True,
            "identity": {"executable_version": "self-test",
                         "executable_sha256": "0" * 64,
                         "source_commit": "a" * 40, "module": "self-test"},
            "official_ipi": {
                "version": IPI_VERSION,
                "parser": "Simulation.load_from_xml(read_only=True)",
                "mode": "isotropic", "steps": 10},
            "xml_template": str((HERE / "isotropic.xml").resolve()),
        }
        assert_ipi_prepare_manifest(prepare_manifest)
        invalid_prepare = json.loads(json.dumps(prepare_manifest))
        del invalid_prepare["official_ipi"]["parser"]
        try:
            assert_ipi_prepare_manifest(invalid_prepare)
        except AssertionError:
            pass
        else:
            raise AssertionError("incomplete i-PI prepare metadata was accepted")
        processes = []
        marker = directory / "marker"
        try:
            child_pid_file = directory / "child.pid"
            process = start_managed_process([
                sys.executable, "-c",
                ("import pathlib,subprocess,sys,time; "
                 "c=subprocess.Popen([sys.executable,'-c','import time;time.sleep(60)']); "
                 "pathlib.Path(r'{}').write_text(str(c.pid)); "
                 "pathlib.Path(r'{}').write_text('ready'); time.sleep(60)")
                .format(child_pid_file, marker)])
            processes.append(process)
            assert wait_until(marker.exists, timeout=2.0)
            assert not wait_until(lambda: False, timeout=0.05, interval=0.01)
        finally:
            terminate_processes(processes)
        assert process.poll() is not None
        child_pid = int(child_pid_file.read_text())
        def child_gone():
            stat = Path("/proc/{}/stat".format(child_pid))
            return not stat.exists() or stat.read_text().split()[2] == "Z"
        assert wait_until(child_gone, 2.0)
        graceful = start_managed_process([sys.executable, "-c", "raise SystemExit(0)"])
        assert graceful.wait(timeout=2.0) == 0
        class FakeProcess:
            def __init__(self, returncodes):
                self.returncodes = list(returncodes)
            def poll(self):
                if len(self.returncodes) > 1:
                    return self.returncodes.pop(0)
                return self.returncodes[0]
        class FakeClock:
            def __init__(self, values):
                self.values = iter(values)
            def __call__(self):
                return next(self.values)
        assert abacus_exit_state(None, None, 3.0, 15.0) == ("running", None)
        assert abacus_exit_state(0, None, 3.0, 15.0) == ("grace", 3.0)
        assert abacus_exit_state(0, 3.0, 18.0, 15.0) == (
            "grace-timeout", 3.0)
        assert wait_for_ipi_shutdown(
            FakeProcess([None, None, 0]), FakeProcess([None, 0]),
            monotonic=FakeClock([0.0, 0.0, 1.0]),
            sleeper=lambda _: None) == 0
        try:
            wait_for_ipi_shutdown(
                FakeProcess([None]), FakeProcess([9, 0]),
                monotonic=FakeClock([0.0, 0.0]), sleeper=lambda _: None)
        except RuntimeError as error:
            assert str(error) == "ABACUS exited with 9"
        else:
            raise AssertionError("nonzero ABACUS exit was accepted")
        try:
            wait_for_ipi_shutdown(
                FakeProcess([None, None]), FakeProcess([0, 0]),
                monotonic=FakeClock([0.0, 0.0, 15.0]),
                sleeper=lambda _: None)
        except TimeoutError as error:
            assert "within 15" in str(error)
        else:
            raise AssertionError("stalled i-PI shutdown was accepted")
        try:
            wait_for_ipi_shutdown(
                FakeProcess([None]), FakeProcess([None]),
                trajectory_timeout=900.0,
                monotonic=FakeClock([0.0, 900.0]),
                sleeper=lambda _: None)
        except TimeoutError as error:
            assert "exceeded 900" in str(error)
        else:
            raise AssertionError("trajectory timeout was accepted")
        assert wait_for_ipi_shutdown(
            FakeProcess([None, None, None, 0]), FakeProcess([0, 0, 0]),
            monotonic=FakeClock([0.0, 899.0, 900.0, 913.999]),
            sleeper=lambda _: None) == 0
        socket_name = "abacus_vc_selftest"
        owned = owned_socket_path(socket_name)
        owned.write_text("owned")
        unlink_owned_socket(socket_name)
        assert not owned.exists()
        try:
            owned_socket_path("../unsafe")
        except AssertionError:
            pass
        else:
            raise AssertionError("unsafe socket path was accepted")
    print("run_validation self-test: PASS (official i-PI {})".format(IPI_VERSION))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abacus", default="abacus")
    parser.add_argument("--basis", choices=("pw", "lcao"), default="pw")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--precision", choices=("double", "single"), default="double")
    parser.add_argument("--ks-solver", default=None,
                        help="explicit ABACUS ks_solver (default preserves backend defaults)")
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
    output = args.output or args.workdir / (
        "prepare.json" if args.prepare_only else "ipi-validation.json")
    config = ase_validation.Config(
        args.abacus, args.basis, args.device, args.precision,
        args.workdir.resolve(), output.resolve(), args.pp_orb_root.resolve(),
        args.ks_solver)
    if not config.pp_orb_root.is_dir():
        raise SystemExit("--pp-orb-root does not exist: {}".format(
            config.pp_orb_root))
    if args.prepare_only:
        xml_path, abacus_dir = _prepare_instance(
            config, args.mode, config.workdir / "prepared",
            args.steps, 0.0, 314159, "abacus_vc_prepare")
        manifest = build_ipi_prepare_manifest(
            config, args.mode, args.steps, xml_path, abacus_dir)
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(json.dumps(
            manifest, indent=2, allow_nan=False) + "\n")
        print("prepared parser-validated {} i-PI/ABACUS case in {}".format(
            args.mode, config.workdir))
        return
    run_validation(config, args.mode, args.steps)
    print("wrote {}".format(config.output))


if __name__ == "__main__":
    main()
