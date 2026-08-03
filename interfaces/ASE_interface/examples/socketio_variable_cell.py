#!/usr/bin/env python3
"""Validate ABACUS variable-cell socket I/O against ASE and FileIO.

ASE Voigt order is xx, yy, zz, yz, xz, xy. Socket floating-point values are
binary64; validation uses eV/Angstrom units only as an independent outer check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

VOIGT = ("xx", "yy", "zz", "yz", "xz", "xy")
DELTAS = (1.0e-4, 3.0e-4, 1.0e-3)
FD_LIMITS = {"rtol": 2.0e-2, "atol_ev_per_angstrom3": 5.0e-4}
IDENTICAL_LIMITS = {"rtol": 2.0e-3, "atol_ev_per_angstrom3": 5.0e-5}
DOUBLE_LIMITS = {"rtol": 1.0e-2, "atol_ev_per_angstrom3": 1.0e-4}
SMOKE_LIMITS = {"rtol": 5.0e-2, "atol_ev_per_angstrom3": 2.0e-3}
VOLUME_LIMITS = {"rtol": 1.0e-10, "atol_angstrom3": 1.0e-8}
FILTER_LIMITS = {
    "energy_rtol": 1.0e-8, "energy_atol_ev": 1.0e-7,
    "stress_rtol": 1.0e-4, "stress_atol_ev_per_angstrom3": 1.0e-7,
    "shear_rtol": 1.0e-4, "shear_atol": 1.0e-8,
    "gradient_rtol": 1.0e-4, "gradient_atol_ev_per_angstrom3": 1.0e-7,
}
HARTREE_EV = 27.211386245988
REQUIRED_FRAME_FIELDS = (
    "executable_version", "executable_sha256", "source_commit", "module",
    "backend", "device", "precision", "cell_angstrom",
    "precision_settings",
    "volume_angstrom3", "condition_number", "scf_converged", "energy_ev",
    "atom_count", "forces_ev_per_angstrom", "raw_abacus_stress_kbar",
    "socket_virial_hartree", "ase_stress_ev_per_angstrom3", "thresholds",
)


@dataclass(frozen=True)
class Config:
    abacus: str
    basis: str
    device: str
    precision: str
    workdir: Path
    output: Path
    pp_orb_root: Path


def displaced_triclinic_si2():
    """The canonical PW/LCAO, ASE/i-PI frame."""
    from ase import Atoms
    return Atoms(
        "Si2",
        positions=np.array([[0.12, 0.08, 0.05], [2.77, 2.58, 2.91]],
                           dtype=np.float64),
        cell=np.array([[5.43, 0.31, 0.17],
                       [0.00, 5.21, 0.37],
                       [0.00, 0.00, 5.57]], dtype=np.float64),
        pbc=True,
    )


def deformation_matrix(component: int, amount: float) -> np.ndarray:
    """Return F; Voigt shear uses gamma/2 in both symmetric entries."""
    if component not in range(6) or not np.isfinite(amount):
        raise ValueError("invalid finite strain")
    f = np.eye(3, dtype=np.float64)
    if component < 3:
        f[component, component] += amount
    else:
        i, j = ((1, 2), (0, 2), (0, 1))[component - 3]
        f[i, j] += 0.5 * amount
        f[j, i] += 0.5 * amount
    return f


def deform_atoms(atoms, component: int, amount: float):
    varied = atoms.copy()
    f = deformation_matrix(component, amount)
    varied.set_cell(np.asarray(atoms.cell, dtype=np.float64) @ f.T,
                    scale_atoms=True)
    assert_valid_frame(varied.cell.array, varied.positions)
    return varied


def assert_valid_frame(cell: np.ndarray, positions: np.ndarray) -> None:
    cell = np.asarray(cell, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    if cell.shape != (3, 3) or positions.ndim != 2 or positions.shape[1] != 3:
        raise AssertionError("invalid cell/position shape")
    if not np.all(np.isfinite(cell)) or not np.all(np.isfinite(positions)):
        raise AssertionError("non-finite cell or positions")
    determinant = float(np.linalg.det(cell))
    condition = float(np.linalg.cond(cell, 2))
    if not determinant > 0.0:
        raise AssertionError("cell determinant must be positive")
    if not np.isfinite(condition) or condition >= 1.0e12:
        raise AssertionError("cell is singular or ill-conditioned")


def finite_difference_scan(
    atoms,
    energy: Callable[[object], float],
    analytic_stress: Iterable[float],
    deltas: Iterable[float] = DELTAS,
    limits: dict = FD_LIMITS,
) -> list[dict]:
    """Central six-strain derivative divided by the unstrained volume."""
    reference = np.asarray(tuple(analytic_stress), dtype=np.float64)
    if reference.shape != (6,) or not np.all(np.isfinite(reference)):
        raise AssertionError("analytic ASE stress must contain six finite values")
    volume = float(atoms.get_volume())
    records = []
    for component, name in enumerate(VOIGT):
        points = []
        for delta in deltas:
            plus_result = energy(deform_atoms(atoms, component, +delta))
            minus_result = energy(deform_atoms(atoms, component, -delta))
            plus_frame = minus_frame = None
            if isinstance(plus_result, tuple):
                plus_result, plus_frame = plus_result
            if isinstance(minus_result, tuple):
                minus_result, minus_frame = minus_result
            plus, minus = float(plus_result), float(minus_result)
            if not np.isfinite(plus) or not np.isfinite(minus):
                raise AssertionError("finite-difference energy is non-finite")
            fd = (plus - minus) / (2.0 * delta * volume)
            tolerance = (limits["atol_ev_per_angstrom3"]
                         + limits["rtol"] * abs(reference[component]))
            points.append({
                "delta": float(delta), "energy_plus_ev": plus,
                "energy_minus_ev": minus, "fd_stress_ev_per_angstrom3": fd,
                "analytic_stress_ev_per_angstrom3": float(reference[component]),
                "absolute_error": abs(fd - reference[component]),
                "atol_plus_rtol": tolerance,
                "pass": bool(abs(fd - reference[component]) <= tolerance),
            })
            if plus_frame is not None or minus_frame is not None:
                if plus_frame is None or minus_frame is None:
                    raise AssertionError("both finite-difference frames are required")
                assert_real_frame_schema(plus_frame)
                assert_real_frame_schema(minus_frame)
                points[-1]["plus_frame"] = plus_frame
                points[-1]["minus_frame"] = minus_frame
        passed = [point for point in points if point["pass"]]
        adjacent_plateau = any(
            points[i]["pass"] and points[i + 1]["pass"]
            and abs(points[i]["fd_stress_ev_per_angstrom3"]
                    - points[i + 1]["fd_stress_ev_per_angstrom3"])
            <= max(points[i]["atol_plus_rtol"], points[i + 1]["atol_plus_rtol"])
            for i in range(len(points) - 1)
        )
        records.append({"component": name, "points": points,
                        "passing_points": len(passed),
                        "plateau_pass": bool(adjacent_plateau)})
    return records


def require_fd_plateau(records: list[dict]) -> None:
    if [record.get("component") for record in records] != list(VOIGT):
        raise AssertionError("finite-difference records use the wrong Voigt order")
    failed = [record["component"] for record in records
              if record.get("passing_points", 0) < 2
              or not record.get("plateau_pass", False)]
    if failed:
        raise AssertionError("no multi-delta convergence plateau: " + ",".join(failed))


def virial_from_ase_stress(stress: Iterable[float], volume: float) -> np.ndarray:
    """Invert ASE sigma=-W/V and return a full virial in Hartree."""
    from ase.stress import voigt_6_to_full_3x3_stress
    tensor = voigt_6_to_full_3x3_stress(
        np.asarray(tuple(stress), dtype=np.float64))
    return -float(volume) * tensor / HARTREE_EV


def validate_virial_sign(stress: Iterable[float], virial_hartree: np.ndarray,
                         volume: float) -> None:
    expected = virial_from_ase_stress(stress, volume)
    actual = np.asarray(virial_hartree, dtype=np.float64)
    if actual.shape != (3, 3) or not np.all(np.isfinite(actual)):
        raise AssertionError("socket virial is absent or non-finite")
    if not np.allclose(actual, expected, rtol=2.0e-12, atol=2.0e-12):
        raise AssertionError("socket virial sign/unit/layout violates sigma_ASE=-W/V")


def assert_real_frame_schema(record: dict) -> None:
    from ase import units
    from ase.stress import full_3x3_to_voigt_6_stress
    missing = [field for field in REQUIRED_FRAME_FIELDS if field not in record]
    if missing:
        raise AssertionError("real frame is missing fields: " + ",".join(missing))
    for key in ("executable_version", "executable_sha256", "source_commit",
                "module", "backend", "device", "precision"):
        if not isinstance(record[key], str):
            raise AssertionError(key + " must be a string")
    if not record["executable_version"] or not record["source_commit"]:
        raise AssertionError("identity strings must be nonempty")
    if (len(record["executable_sha256"]) != 64
            or re.fullmatch(r"[0-9a-fA-F]{64}", record["executable_sha256"]) is None):
        raise AssertionError("executable_sha256 must be 64 hexadecimal characters")
    if record["backend"] not in ("pw", "lcao") or record["device"] not in ("cpu", "gpu"):
        raise AssertionError("invalid backend/device")
    if record["precision"] not in ("double", "single"):
        raise AssertionError("invalid precision")
    precision = record["precision_settings"]
    if not isinstance(precision, dict) or precision.get("socket_float") != "IEEE-754 binary64":
        raise AssertionError("invalid precision settings")
    expected_gint = (None if record["backend"] == "pw" else
                     ("double" if record["precision"] == "double" else "mix"))
    if (precision.get("precision") != record["precision"]
            or precision.get("gint_precision") != expected_gint):
        raise AssertionError("recorded precision settings are inconsistent")
    try:
        cell = np.asarray(record["cell_angstrom"], dtype=np.float64)
        forces = np.asarray(record["forces_ev_per_angstrom"], dtype=np.float64)
        raw_stress = np.asarray(record["raw_abacus_stress_kbar"], dtype=np.float64)
        virial = np.asarray(record["socket_virial_hartree"], dtype=np.float64)
        stress = np.asarray(record["ase_stress_ev_per_angstrom3"], dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AssertionError("real frame arrays must be numeric") from error
    assert_valid_frame(cell, forces)
    atom_count = record["atom_count"]
    if (isinstance(atom_count, bool) or not isinstance(atom_count, (int, np.integer))
            or atom_count <= 0 or forces.shape[0] != atom_count):
        raise AssertionError("atom_count disagrees with force rows")
    if raw_stress.shape != (3, 3):
        raise AssertionError("forces/raw-stress shape is invalid")
    if virial.shape != (3, 3) or stress.shape != (6,):
        raise AssertionError("virial/ASE-stress shape is invalid")
    arrays = (forces, raw_stress, virial, stress)
    if not all(np.all(np.isfinite(value)) for value in arrays):
        raise AssertionError("real frame arrays must be finite")
    energy = record["energy_ev"]
    volume = record["volume_angstrom3"]
    condition = record["condition_number"]
    if isinstance(energy, bool) or not np.isscalar(energy) or not np.isfinite(energy):
        raise AssertionError("energy must be a finite scalar")
    if (isinstance(volume, bool) or not np.isscalar(volume)
            or not np.isfinite(volume) or volume <= 0.0):
        raise AssertionError("volume must be finite and positive")
    determinant = float(np.linalg.det(cell))
    volume_tolerance = (VOLUME_LIMITS["atol_angstrom3"]
                        + VOLUME_LIMITS["rtol"] * abs(determinant))
    if abs(float(volume) - determinant) > volume_tolerance:
        raise AssertionError("recorded volume disagrees with cell determinant")
    computed_condition = float(np.linalg.cond(cell, 2))
    if (isinstance(condition, bool) or not np.isscalar(condition)
            or not np.isfinite(condition)
            or condition >= 1.0e12
            or not np.isclose(condition, computed_condition, rtol=1.0e-10, atol=1.0e-12)):
        raise AssertionError("recorded condition number is invalid")
    thresholds = record["thresholds"]
    if not isinstance(thresholds, dict) or not thresholds:
        raise AssertionError("thresholds must be a nonempty mapping")
    def check_thresholds(value):
        if isinstance(value, dict):
            if not value:
                raise AssertionError("empty threshold group")
            for nested in value.values():
                check_thresholds(nested)
        elif isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) or value < 0:
            raise AssertionError("threshold values must be finite and nonnegative")
    check_thresholds(thresholds)
    active = thresholds["active_stress"]
    tolerance = active["atol_ev_per_angstrom3"] + active["rtol"] * np.abs(stress)
    raw_ase = -0.1 * units.GPa * full_3x3_to_voigt_6_stress(raw_stress)
    if not np.all(np.abs(stress - raw_ase) <= tolerance):
        raise AssertionError("raw ABACUS stress sign/unit/order mismatch")
    virial_stress = -full_3x3_to_voigt_6_stress(virial) * units.Ha / float(volume)
    if not np.all(np.abs(stress - virial_stress) <= tolerance):
        raise AssertionError("virial/ASE-stress closure failed")
    if type(record["scf_converged"]) is not bool or record["scf_converged"] is not True:
        raise AssertionError("SCF convergence may not be fabricated or omitted")


def assert_json_schema(payload: dict) -> None:
    if payload.get("schema_version") != 1:
        raise AssertionError("unsupported or absent JSON schema version")
    for record in payload.get("frames", []):
        assert_real_frame_schema(record)
    require_fd_plateau(payload["finite_difference"])
    for name in ("unit_cell_filter", "frechet_cell_filter"):
        result = payload["filters"][name]
        if result.get("accepted_steps", 0) < 3:
            raise AssertionError(name + " did not accept three stable steps")
        if len(result.get("frames", [])) < result["accepted_steps"] + 1:
            raise AssertionError(name + " accepted-step/frame count is inconsistent")
        if not result.get("energy_decreased") or not result.get("stress_decreased"):
            raise AssertionError(name + " did not improve both energy and stress")
        for record in result.get("frames", []):
            assert_real_frame_schema(record)


def evaluate_filter_stability(frames: list[dict], initial_shear_gradient=None,
                              minimum_steps: int = 3, limits=None) -> dict:
    limits = dict(FILTER_LIMITS if limits is None else limits)
    if len(frames) - 1 < minimum_steps:
        raise AssertionError("filter did not accept the minimum number of steps")
    for frame in frames:
        assert_valid_frame(frame["cell_angstrom"],
                           np.zeros((1, 3), dtype=np.float64))
        values = (frame["energy_ev"],
                  frame["max_abs_stress_ev_per_angstrom3"])
        if not np.all(np.isfinite(values)):
            raise AssertionError("filter energy/stress is non-finite")
    energy_drop = frames[0]["energy_ev"] - frames[-1]["energy_ev"]
    stress_drop = (frames[0]["max_abs_stress_ev_per_angstrom3"]
                   - frames[-1]["max_abs_stress_ev_per_angstrom3"])
    energy_tolerance = (limits["energy_atol_ev"] + limits["energy_rtol"]
                        * max(abs(frames[0]["energy_ev"]), abs(frames[-1]["energy_ev"])))
    stress_scale = max(frames[0]["max_abs_stress_ev_per_angstrom3"],
                       frames[-1]["max_abs_stress_ev_per_angstrom3"])
    stress_tolerance = limits["stress_atol_ev_per_angstrom3"] + limits["stress_rtol"] * stress_scale
    shear_decision = {"pass": False}
    if initial_shear_gradient is not None:
        gradient = np.asarray(initial_shear_gradient, dtype=np.float64)
        if gradient.shape != (3,) or not np.all(np.isfinite(gradient)):
            raise AssertionError("initial shear gradient must have three finite values")
        gradient_magnitude = float(np.linalg.norm(gradient))
        gradient_tolerance = (
            limits["gradient_atol_ev_per_angstrom3"]
            + limits["gradient_rtol"] * gradient_magnitude)
        candidates = []
        for frame in frames[1:]:
            strain = np.asarray(frame["cell_strain_voigt"], dtype=np.float64)
            displacement = strain[3:]
            magnitude = float(np.linalg.norm(displacement))
            displacement_tolerance = (limits["shear_atol"]
                                      + limits["shear_rtol"] * magnitude)
            projection = float(np.dot(gradient, displacement))
            projection_tolerance = gradient_tolerance * magnitude
            candidates.append({
                "magnitude": magnitude,
                "negative_gradient_projection_ev_per_angstrom3": projection,
                "displacement_atol_plus_rtol": displacement_tolerance,
                "gradient_atol_plus_rtol_ev_per_angstrom3": gradient_tolerance,
                "projection_atol_plus_rtol_ev_per_angstrom3": projection_tolerance,
                "pass": bool(magnitude > displacement_tolerance
                             and gradient_magnitude > gradient_tolerance
                             and projection < -projection_tolerance),
            })
        shear_decision = {"initial_gradient_ev_per_angstrom3": gradient.tolist(),
                          "gradient_magnitude_ev_per_angstrom3": gradient_magnitude,
                          "steps": candidates,
                          "pass": any(item["pass"] for item in candidates)}
    result = {
        "accepted_steps": len(frames) - 1,
        "energy_decrease_ev": float(energy_drop),
        "stress_decrease_ev_per_angstrom3": float(stress_drop),
        "energy_atol_plus_rtol_ev": float(energy_tolerance),
        "stress_atol_plus_rtol_ev_per_angstrom3": float(stress_tolerance),
        "energy_decreased": bool(energy_drop > energy_tolerance),
        "stress_decreased": bool(stress_drop > stress_tolerance),
        "shear_direction": shear_decision, "thresholds": limits,
        "positive_determinant_and_finite": True,
    }
    if initial_shear_gradient is not None and not shear_decision["pass"]:
        raise AssertionError("filter has no nonzero shear displacement along energy descent")
    return result


def _common_kwargs(config: Config) -> dict:
    inp = {
        "calculation": "scf", "basis_type": config.basis,
        "device": config.device, "precision": config.precision,
        "ecutwfc": 50, "symmetry": 0, "kspacing": 0.45,
        "scf_thr": 1.0e-9 if config.precision == "double" else 1.0e-6,
        "scf_nmax": 100, "chg_extrap": "atomic", "cal_force": 1,
        "cal_stress": 1,
    }
    if config.basis == "lcao":
        inp["gint_precision"] = "double" if config.precision == "double" else "mix"
    kwargs = {
        "pseudopotentials": {"Si": "Si_ONCV_PBE-1.2.upf"},
        "inp": inp,
    }
    if config.basis == "lcao":
        kwargs["basissets"] = {"Si": "Si_gga_8au_100Ry_2s2p1d.orb"}
    return kwargs


def active_stress_limits(config: Config, reference_limits: dict) -> dict:
    return dict(reference_limits if config.precision == "double" else SMOKE_LIMITS)


def _load_abacus_api():
    interface_root = Path(__file__).resolve().parents[1]
    if str(interface_root) not in sys.path:
        sys.path.insert(0, str(interface_root))
    from abacuslite import Abacus, AbacusProfile, AbacusSocketIO
    return Abacus, AbacusProfile, AbacusSocketIO


def _profile(config: Config):
    _, AbacusProfile, _ = _load_abacus_api()
    return AbacusProfile(command=config.abacus,
                         pseudo_dir=config.pp_orb_root,
                         orbital_dir=config.pp_orb_root,
                         omp_num_threads=1)


def _recording_socket_class():
    """Capture the exact ASE SocketServer virial before core.py consumes it."""
    _, _, AbacusSocketIO = _load_abacus_api()

    class RecordingAbacusSocketIO(AbacusSocketIO):
        def launch_server(self):
            server = super().launch_server()
            calculate = server.calculate

            def record(atoms):
                results = calculate(atoms)
                if "virial" not in results:
                    raise AssertionError("socket result omitted the raw virial")
                from ase import units
                self.last_socket_virial_ev = np.asarray(
                    results["virial"], dtype=np.float64).copy()
                self.last_socket_virial_hartree = self.last_socket_virial_ev / units.Ha
                return results

            server.calculate = record
            return server

    return RecordingAbacusSocketIO


def prepare_case(config: Config, directory: Path, socket: bool = True) -> Path:
    """Write a runnable PW/LCAO case without launching a calculation."""
    Abacus, _, AbacusSocketIO = _load_abacus_api()
    atoms = displaced_triclinic_si2()
    shutil.rmtree(directory, ignore_errors=True)
    directory.mkdir(parents=True)
    kwargs = _common_kwargs(config)
    if socket:
        calc = AbacusSocketIO(profile=_profile(config), directory=directory,
                              unixsocket="task7_prepare", variable_cell=True,
                              **kwargs)
        try:
            calc.abacus.write_input(atoms, properties=["energy", "forces", "stress"])
        finally:
            calc.close()
    else:
        calc = Abacus(profile=_profile(config), directory=directory, **kwargs)
        calc.write_input(atoms, properties=["energy", "forces", "stress"])
    text = (directory / "INPUT").read_text()
    required = ("socket_variable_cell", "cal_stress") if socket else ("cal_stress",)
    for keyword in required:
        if not re.search(r"^\s*{}\s+(1|true)\s*$".format(keyword), text,
                         flags=re.MULTILINE | re.IGNORECASE):
            raise AssertionError("prepared INPUT lacks real " + keyword)
    expected_gint = "double" if config.precision == "double" else "mix"
    if config.basis == "lcao" and not re.search(
            r"^\s*gint_precision\s+{}\s*$".format(expected_gint), text,
            flags=re.MULTILINE | re.IGNORECASE):
        raise AssertionError("LCAO gint_precision does not match precision mode")
    return directory


def _find_log(directory: Path) -> Path:
    matches = sorted(directory.glob("OUT.*/running_scf.log"))
    if not matches:
        raise AssertionError("ABACUS running_scf.log is absent")
    return matches[-1]


def raw_frame_series(directory: Path, expected_frames: int | None = 1) -> list:
    text = _find_log(directory).read_text(errors="replace")
    if "convergence has not been achieved" in text.lower():
        raise AssertionError("ABACUS reported an unconverged SCF")
    lines = text.splitlines()
    frames = []
    pending_convergence = 0
    for index, line in enumerate(lines):
        if ("#SCF IS CONVERGED#" in line
                or "charge density convergence is achieved" in line):
            pending_convergence += 1
        if "TOTAL-STRESS" not in line.upper():
            continue
        if pending_convergence != 1:
            raise AssertionError("stress block lacks a unique preceding SCF convergence")
        rows = []
        for candidate in lines[index + 1:index + 12]:
            numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?",
                                 candidate)
            if len(numbers) >= 3:
                rows.append([float(value) for value in numbers[-3:]])
                if len(rows) == 3:
                    break
        if len(rows) == 3:
            frames.append({"scf_converged": True,
                           "raw_abacus_stress_kbar": np.asarray(
                               rows, dtype=np.float64).tolist()})
            pending_convergence = 0
    if ((expected_frames is not None and len(frames) != expected_frames)
            or not frames or pending_convergence != 0):
        raise AssertionError("ABACUS log frames do not exactly match expected frames")
    return frames


def raw_stress_series_and_convergence(directory: Path,
                                      expected_frames: int = 1) -> tuple[list, int]:
    frames = raw_frame_series(directory, expected_frames)
    return [frame["raw_abacus_stress_kbar"] for frame in frames], len(frames)


def _raw_stress_and_convergence(directory: Path) -> tuple[list, bool]:
    frame = raw_frame_series(directory, None)[-1]
    return frame["raw_abacus_stress_kbar"], frame["scf_converged"]


def _identity(config: Config) -> dict:
    command = shlex.split(config.abacus)
    completed = subprocess.run(command + ["--version"], text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               timeout=30, check=True)
    match = re.search(r"ABACUS version\s+(\S+)", completed.stdout)
    if match is None:
        raise AssertionError("ABACUS version output is unparseable")
    binary = None
    for token in reversed(command):
        candidate = shutil.which(token)
        if candidate and "abacus" in Path(candidate).name.lower():
            binary = Path(candidate)
            break
    if binary is None:
        raise AssertionError("cannot resolve ABACUS executable for hashing")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    try:
        source_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3],
            text=True, stdout=subprocess.PIPE, check=True).stdout.strip()
    except subprocess.SubprocessError:
        source_commit = "unknown"
    return {"executable_version": match.group(1),
            "executable_sha256": digest, "source_commit": source_commit,
            "module": os.environ.get("LOADEDMODULES", "")}


def build_prepare_manifest(config: Config, cases: dict, kind: str,
                           extra: dict | None = None) -> dict:
    identity = _identity(config)
    pseudo = (config.pp_orb_root / "Si_ONCV_PBE-1.2.upf").resolve()
    orbital = (config.pp_orb_root / "Si_gga_8au_100Ry_2s2p1d.orb").resolve()
    if not pseudo.is_file() or (config.basis == "lcao" and not orbital.is_file()):
        raise AssertionError("resolved PP/ORB files are absent")
    manifest = {
        "schema_version": 1, "kind": kind, "backend": config.basis,
        "device": config.device, "precision": config.precision,
        "gint_precision": (None if config.basis == "pw" else
                           ("double" if config.precision == "double" else "mix")),
        "is_reference": config.precision == "double",
        "cases": {name: str(Path(path).resolve()) for name, path in cases.items()},
        "resolved_files": {
            "pseudopotential": str(pseudo),
            "orbital": str(orbital) if config.basis == "lcao" else None,
        },
        "socket_variable_cell": True, "cal_stress": True,
        "identity": identity,
    }
    if extra:
        manifest.update(extra)
    assert_prepare_manifest(manifest)
    return manifest


def assert_prepare_manifest(manifest: dict) -> None:
    required = ("schema_version", "kind", "backend", "device", "precision",
                "gint_precision", "is_reference", "cases", "resolved_files",
                "socket_variable_cell", "cal_stress", "identity")
    missing = [key for key in required if key not in manifest]
    if missing:
        raise AssertionError("prepare manifest missing: " + ",".join(missing))
    if (manifest["schema_version"] != 1
            or not isinstance(manifest["kind"], str) or not manifest["kind"]
            or manifest["backend"] not in ("pw", "lcao")
            or manifest["device"] not in ("cpu", "gpu")
            or manifest["precision"] not in ("double", "single")
            or type(manifest["is_reference"]) is not bool
            or manifest["is_reference"] != (manifest["precision"] == "double")
            or not isinstance(manifest["cases"], dict)):
        raise AssertionError("invalid prepare manifest schema")
    if (not manifest["cases"]
            or not all(isinstance(path, str) and Path(path).is_absolute()
                       for path in manifest["cases"].values())):
        raise AssertionError("prepare case paths must be absolute")
    if manifest["socket_variable_cell"] is not True or manifest["cal_stress"] is not True:
        raise AssertionError("prepare manifest lacks socket/stress flags")
    expected = (None if manifest["backend"] == "pw" else
                ("double" if manifest["precision"] == "double" else "mix"))
    if manifest["gint_precision"] != expected:
        raise AssertionError("prepare manifest gint_precision mismatch")
    resolved = manifest["resolved_files"]
    if (not isinstance(resolved, dict)
            or not isinstance(resolved.get("pseudopotential"), str)
            or not Path(resolved["pseudopotential"]).is_absolute()):
        raise AssertionError("prepare pseudopotential path must be absolute")
    orbital = resolved.get("orbital")
    if ((manifest["backend"] == "pw" and orbital is not None)
            or (manifest["backend"] == "lcao"
                and (not isinstance(orbital, str)
                     or not Path(orbital).is_absolute()))):
        raise AssertionError("prepare orbital path is inconsistent")
    identity = manifest["identity"]
    if not isinstance(identity, dict) or not all(
            key in identity for key in ("executable_version", "executable_sha256",
                                        "source_commit", "module")):
        raise AssertionError("prepare manifest identity is incomplete")
    if (not all(isinstance(identity[key], str) for key in
                ("executable_version", "executable_sha256", "source_commit", "module"))
            or not identity["executable_version"] or not identity["source_commit"]
            or re.fullmatch(r"[0-9a-fA-F]{64}",
                            identity["executable_sha256"]) is None):
        raise AssertionError("prepare manifest identity is invalid")


def _frame_record(config: Config, identity: dict, atoms, directory: Path,
                  source: str, raw_socket_virial=None) -> dict:
    stress = np.asarray(atoms.get_stress(), dtype=np.float64)
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    energy = float(atoms.get_potential_energy())
    raw_stress, converged = _raw_stress_and_convergence(directory)
    volume = float(atoms.get_volume())
    if source.startswith("socket"):
        if raw_socket_virial is None:
            raise AssertionError("actual socket virial was not captured")
        virial = np.asarray(raw_socket_virial, dtype=np.float64)
    else:
        virial = virial_from_ase_stress(stress, volume)
    validate_virial_sign(stress, virial, volume)
    result = dict(identity)
    result.update({
        "source": source, "backend": config.basis, "device": config.device,
        "precision": config.precision,
        "precision_settings": {
            "precision": config.precision,
            "gint_precision": (None if config.basis == "pw" else
                               ("double" if config.precision == "double" else "mix")),
            "socket_float": "IEEE-754 binary64",
        },
        "cell_angstrom": np.asarray(atoms.cell, dtype=np.float64).tolist(),
        "volume_angstrom3": volume,
        "condition_number": float(np.linalg.cond(atoms.cell.array, 2)),
        "scf_converged": converged, "energy_ev": energy,
        "atom_count": int(len(forces)),
        "forces_ev_per_angstrom": forces.tolist(),
        "raw_abacus_stress_kbar": raw_stress,
        "socket_virial_hartree": virial.tolist(),
        "ase_stress_ev_per_angstrom3": stress.tolist(),
        "thresholds": {"identical": IDENTICAL_LIMITS,
                       "finite_difference": FD_LIMITS,
                       "cpu_gpu_double": DOUBLE_LIMITS,
                       "single_mixed_smoke_only": SMOKE_LIMITS,
                       "active_stress": active_stress_limits(config, IDENTICAL_LIMITS),
                       "volume": VOLUME_LIMITS},
        "virial_provenance": ("captured from ASE SocketServer before conversion"
                              if source.startswith("socket") else
                              "derived from FileIO ASE stress for schema parity"),
    })
    return result


def run_fileio_reference(config: Config, directory: Path) -> tuple[object, dict]:
    Abacus, _, _ = _load_abacus_api()
    shutil.rmtree(directory, ignore_errors=True)
    atoms = displaced_triclinic_si2()
    atoms.calc = Abacus(profile=_profile(config), directory=directory,
                        **_common_kwargs(config))
    atoms.get_potential_energy()
    atoms.get_forces()
    atoms.get_stress()
    return atoms, _frame_record(config, _identity(config), atoms, directory, "fileio")


def _filter_run(config: Config, filter_name: str, directory: Path) -> dict:
    from ase.filters import FrechetCellFilter, UnitCellFilter
    from ase.optimize import BFGS
    AbacusSocketIO = _recording_socket_class()
    shutil.rmtree(directory, ignore_errors=True)
    atoms = displaced_triclinic_si2()
    calc = AbacusSocketIO(profile=_profile(config), directory=directory,
                          unixsocket="task7_{}_{}".format(filter_name, os.getpid()),
                          timeout=300, variable_cell=True, **_common_kwargs(config))
    accepted = []
    identity = _identity(config)
    initial_cell = atoms.cell.array.copy()
    with calc:
        atoms.calc = calc
        filter_class = (UnitCellFilter if filter_name == "unit_cell_filter"
                        else FrechetCellFilter)
        filtered = filter_class(atoms)
        optimizer = BFGS(filtered, logfile=str(directory / "optimizer.log"),
                         maxstep=0.01)

        def capture():
            if (accepted and np.array_equal(
                    atoms.cell.array,
                    np.asarray(accepted[-1]["cell_angstrom"], dtype=np.float64))):
                return
            assert_valid_frame(atoms.cell.array, atoms.positions)
            energy = float(atoms.get_potential_energy())
            max_stress = float(np.max(np.abs(atoms.get_stress())))
            if not np.isfinite(energy + max_stress):
                raise AssertionError("filter produced non-finite energy/stress")
            frame = _frame_record(
                config, identity, atoms, directory, "socket",
                calc.last_socket_virial_hartree)
            frame["max_abs_stress_ev_per_angstrom3"] = max_stress
            f = np.linalg.solve(initial_cell, atoms.cell.array).T
            frame["cell_strain_voigt"] = [
                float(f[0, 0] - 1.0), float(f[1, 1] - 1.0),
                float(f[2, 2] - 1.0), float(f[1, 2] + f[2, 1]),
                float(f[0, 2] + f[2, 0]), float(f[0, 1] + f[1, 0])]
            accepted.append(frame)
        capture()
        optimizer.attach(capture, interval=1)
        optimizer.run(fmax=0.0, steps=3)
    if len(accepted) < 4:
        raise AssertionError(filter_name + " did not complete three accepted steps")
    gradient = np.asarray(
        accepted[0]["ase_stress_ev_per_angstrom3"], dtype=np.float64)[3:]
    limits = dict(FILTER_LIMITS)
    if config.precision == "single":
        limits["stress_atol_ev_per_angstrom3"] = SMOKE_LIMITS["atol_ev_per_angstrom3"]
        limits["stress_rtol"] = SMOKE_LIMITS["rtol"]
        limits["gradient_atol_ev_per_angstrom3"] = SMOKE_LIMITS[
            "atol_ev_per_angstrom3"]
        limits["gradient_rtol"] = SMOKE_LIMITS["rtol"]
    result = evaluate_filter_stability(accepted, gradient, limits=limits)
    result["frames"] = accepted
    if not result["energy_decreased"] or not result["stress_decreased"]:
        raise AssertionError(filter_name + " failed energy/stress stability criteria")
    return result


def run_validation(config: Config) -> dict:
    AbacusSocketIO = _recording_socket_class()
    config.workdir.mkdir(parents=True, exist_ok=True)
    identity = _identity(config)
    _, file_record = run_fileio_reference(config, config.workdir / "fileio_reference")
    socket_dir = config.workdir / "socket_reference"
    shutil.rmtree(socket_dir, ignore_errors=True)
    socket_atoms = displaced_triclinic_si2()
    socket_calc = AbacusSocketIO(
        profile=_profile(config), directory=socket_dir,
        unixsocket="task7_ase_{}".format(os.getpid()), timeout=300,
        variable_cell=True, **_common_kwargs(config))
    with socket_calc:
        socket_atoms.calc = socket_calc
        socket_atoms.get_potential_energy()
        socket_atoms.get_forces()
        socket_stress = socket_atoms.get_stress()
        socket_record = _frame_record(
            config, identity, socket_atoms, socket_dir, "socket",
            socket_calc.last_socket_virial_hartree)

        def socket_energy(varied):
            varied.calc = socket_calc
            energy = varied.get_potential_energy()
            frame = _frame_record(
                config, identity, varied, socket_dir, "socket-fd",
                socket_calc.last_socket_virial_hartree)
            return energy, frame
        finite_difference = finite_difference_scan(
            socket_atoms, socket_energy, socket_stress,
            limits=active_stress_limits(config, FD_LIMITS))
    require_fd_plateau(finite_difference)
    energy_error = abs(socket_record["energy_ev"] - file_record["energy_ev"])
    force_error = float(np.max(np.abs(
        np.asarray(socket_record["forces_ev_per_angstrom"])
        - np.asarray(file_record["forces_ev_per_angstrom"]))))
    stress_error = np.abs(np.asarray(socket_record["ase_stress_ev_per_angstrom3"])
                          - np.asarray(file_record["ase_stress_ev_per_angstrom3"]))
    identical_limits = active_stress_limits(config, IDENTICAL_LIMITS)
    stress_tolerance = (identical_limits["atol_ev_per_angstrom3"]
                        + identical_limits["rtol"] * np.abs(
                            np.asarray(file_record["ase_stress_ev_per_angstrom3"])))
    identical = {
        "energy_absolute_error_ev": energy_error,
        "force_max_absolute_error_ev_per_angstrom": force_error,
        "stress_absolute_errors_ev_per_angstrom3": stress_error.tolist(),
        "stress_atol_plus_rtol": stress_tolerance.tolist(),
        "pass": bool(energy_error <= 1.0e-4 and force_error <= 1.0e-5
                     and np.all(stress_error <= stress_tolerance)),
        "thresholds": identical_limits,
        "is_reference": config.precision == "double",
    }
    if not identical["pass"]:
        raise AssertionError("identical-frame FileIO/socket comparison failed")
    shear = [record for record in finite_difference
             if record["component"] in ("yz", "xz", "xy")]
    if not any(abs(record["points"][1]["fd_stress_ev_per_angstrom3"])
               > active_stress_limits(config, FD_LIMITS)["atol_ev_per_angstrom3"]
               for record in shear):
        raise AssertionError("no nonzero finite-difference-consistent shear response")
    payload = {
        "schema_version": 1, "voigt_order": list(VOIGT),
        "shear_deformation": "symmetric gamma/2",
        "finite_difference_deltas": list(DELTAS),
        "frames": [file_record, socket_record],
        "identical_frame": identical, "finite_difference": finite_difference,
        "filters": {
            "unit_cell_filter": _filter_run(
                config, "unit_cell_filter", config.workdir / "unit_cell_filter"),
            "frechet_cell_filter": _filter_run(
                config, "frechet_cell_filter", config.workdir / "frechet_cell_filter"),
        },
    }
    assert_json_schema(payload)
    config.output.parent.mkdir(parents=True, exist_ok=True)
    config.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return payload


def _analytic_self_test() -> None:
    from unittest.mock import patch
    from ase import units
    atoms = displaced_triclinic_si2()
    base_cell = atoms.cell.array.copy()
    volume = atoms.get_volume()
    reference_strain = np.array([0.02, -0.015, 0.01, 0.012, -0.009, 0.017])
    stiffness = np.array([1.2, 1.5, 1.8, 0.9, 1.1, 1.4])
    expected = stiffness * reference_strain / volume

    def analytic_energy(varied):
        f = np.linalg.solve(base_cell, varied.cell.array).T
        strain = np.array([f[0, 0] - 1.0, f[1, 1] - 1.0,
                           f[2, 2] - 1.0, f[1, 2] + f[2, 1],
                           f[0, 2] + f[2, 0], f[0, 1] + f[1, 0]])
        shifted = reference_strain + strain
        return 0.5 * float(np.sum(stiffness * shifted * shifted))
    shear = deformation_matrix(3, 0.2)
    assert shear[1, 2] == shear[2, 1] == 0.1
    records = finite_difference_scan(atoms, analytic_energy, expected)
    require_fd_plateau(records)
    recovered = np.array([record["points"][1]["fd_stress_ev_per_angstrom3"]
                          for record in records])
    assert np.allclose(recovered, expected, rtol=1.0e-10, atol=1.0e-12)
    virial = virial_from_ase_stress(expected, volume)
    validate_virial_sign(expected, virial, volume)
    try:
        validate_virial_sign(expected, -virial, volume)
    except AssertionError:
        pass
    else:
        raise AssertionError("deliberately negated virial was accepted")
    broken = json.loads(json.dumps(records))
    for point in broken[0]["points"]:
        point["pass"] = False
    broken[0]["points"][1]["pass"] = True
    broken[0]["passing_points"] = 1
    broken[0]["plateau_pass"] = False
    try:
        require_fd_plateau(broken)
    except AssertionError:
        pass
    else:
        raise AssertionError("one lucky delta incorrectly established a plateau")
    try:
        assert_valid_frame(np.diag([1.0, 1.0, -1.0]), np.zeros((2, 3)))
    except AssertionError:
        pass
    else:
        raise AssertionError("left-handed cell was accepted")
    frame = {
        "executable_version": "self-test", "executable_sha256": "0" * 64,
        "source_commit": "self-test", "module": "self-test",
        "backend": "pw", "device": "cpu", "precision": "double",
        "precision_settings": {"precision": "double", "gint_precision": None,
                               "socket_float": "IEEE-754 binary64"},
        "cell_angstrom": np.eye(3).tolist(), "volume_angstrom3": 1.0,
        "condition_number": 1.0, "scf_converged": True, "energy_ev": 0.0,
        "atom_count": 2,
        "forces_ev_per_angstrom": np.zeros((2, 3)).tolist(),
        "raw_abacus_stress_kbar": np.zeros((3, 3)).tolist(),
        "socket_virial_hartree": np.zeros((3, 3)).tolist(),
        "ase_stress_ev_per_angstrom3": np.zeros(6).tolist(),
        "thresholds": {"active_stress": IDENTICAL_LIMITS, "volume": VOLUME_LIMITS},
    }
    accepted_mutations = []
    Recording = _recording_socket_class()
    _, _, BaseSocket = _load_abacus_api()
    fake_server = type("FakeServer", (), {})()
    fake_server.calculate = lambda atoms: {
        "virial": units.Ha * virial.copy(), "energy": 0.0,
        "forces": np.zeros((2, 3))}
    recorder = object.__new__(Recording)
    with patch.object(BaseSocket, "launch_server", return_value=fake_server):
        server = recorder.launch_server()
    server.calculate(atoms)
    if np.allclose(recorder.last_socket_virial_hartree, virial):
        pass
    else:
        accepted_mutations.append("ASE-server-eV-recorded-as-Hartree")
    mutations = {
        "negative-volume": {"volume_angstrom3": -1.0},
        "infinite-energy": {"energy_ev": float("inf")},
        "nan-force": {"forces_ev_per_angstrom": [[float("nan"), 0, 0], [0, 0, 0]]},
        "garbage-force": {"forces_ev_per_angstrom": [["garbage", 0, 0], [0, 0, 0]]},
        "one-by-one-raw-stress": {"raw_abacus_stress_kbar": [[0.0]]},
        "garbage-sha256": {"executable_sha256": "not-a-sha256"},
        "volume-cell-mismatch": {"volume_angstrom3": 2.0},
        "condition-number-mismatch": {"condition_number": 2.0},
        "boolean-condition-number": {"condition_number": True},
        "integer-scf-flag": {"scf_converged": 1},
        "atom-count-mismatch": {"atom_count": 1},
        "wrong-virial-sign": {"socket_virial_hartree": np.ones((3, 3)).tolist()},
    }
    for label, changes in mutations.items():
        mutated = dict(frame)
        mutated.update(changes)
        try:
            assert_real_frame_schema(mutated)
        except AssertionError:
            pass
        else:
            accepted_mutations.append(label)
    if accepted_mutations:
        raise AssertionError("review mutations accepted: "
                             + ",".join(accepted_mutations))
    manifest = {
        "schema_version": 1, "kind": "self-test", "backend": "pw",
        "device": "cpu", "precision": "double", "gint_precision": None,
        "is_reference": True, "cases": {"socket": "/tmp/socket"},
        "resolved_files": {"pseudopotential": "/tmp/Si.upf", "orbital": None},
        "socket_variable_cell": True, "cal_stress": True,
        "identity": {"executable_version": "self-test",
                     "executable_sha256": "0" * 64,
                     "source_commit": "self-test", "module": "self-test"},
    }
    assert_prepare_manifest(manifest)
    incomplete_manifest = dict(manifest)
    del incomplete_manifest["resolved_files"]
    try:
        assert_prepare_manifest(incomplete_manifest)
    except AssertionError:
        pass
    else:
        raise AssertionError("incomplete prepare manifest was accepted")
    for label, mutate in (
            ("invalid-identity-sha",
             lambda value: value["identity"].update(
                 {"executable_sha256": "invalid"})),
            ("relative-pseudopotential",
             lambda value: value["resolved_files"].update(
                 {"pseudopotential": "Si.upf"}))):
        invalid_manifest = json.loads(json.dumps(manifest))
        mutate(invalid_manifest)
        try:
            assert_prepare_manifest(invalid_manifest)
        except AssertionError:
            pass
        else:
            raise AssertionError(label + " prepare manifest was accepted")
    payload = {"schema_version": 1, "frames": [frame],
               "finite_difference": records,
               "filters": {
                   "unit_cell_filter": {"accepted_steps": 3,
                                        "frames": [dict(frame) for _ in range(4)],
                                        "energy_decreased": True,
                                        "stress_decreased": True},
                   "frechet_cell_filter": {"accepted_steps": 3,
                                           "frames": [dict(frame) for _ in range(4)],
                                           "energy_decreased": True,
                                           "stress_decreased": True}}}
    assert_json_schema(payload)
    del frame["raw_abacus_stress_kbar"]
    try:
        assert_json_schema(payload)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing raw stress was accepted")
    stable_frames = [
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 2.0,
         "max_abs_stress_ev_per_angstrom3": 0.4,
         "cell_strain_voigt": [0, 0, 0, 0, 0, 0]},
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 1.8,
         "max_abs_stress_ev_per_angstrom3": 0.3,
         "cell_strain_voigt": [0, 0, 0, -0.002, 0, 0]},
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 1.6,
         "max_abs_stress_ev_per_angstrom3": 0.2,
         "cell_strain_voigt": [0, 0, 0, -0.003, 0, 0]},
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 1.4,
         "max_abs_stress_ev_per_angstrom3": 0.1,
         "cell_strain_voigt": [0, 0, 0, -0.004, 0, 0]},
    ]
    assert evaluate_filter_stability(stable_frames, [1, 0, 0])["stress_decreased"]
    smoke_filter_limits = dict(FILTER_LIMITS)
    smoke_filter_limits["stress_atol_ev_per_angstrom3"] = SMOKE_LIMITS[
        "atol_ev_per_angstrom3"]
    smoke_filter_limits["stress_rtol"] = SMOKE_LIMITS["rtol"]
    smoke_filter_limits["gradient_atol_ev_per_angstrom3"] = SMOKE_LIMITS[
        "atol_ev_per_angstrom3"]
    smoke_filter_limits["gradient_rtol"] = SMOKE_LIMITS["rtol"]
    assert evaluate_filter_stability(
        stable_frames, [1, 0, 0], limits=smoke_filter_limits)[
            "shear_direction"]["pass"]
    wrong_shear = json.loads(json.dumps(stable_frames))
    for item in wrong_shear[1:]:
        item["cell_strain_voigt"][3] *= -1
    try:
        evaluate_filter_stability(wrong_shear, [1, 0, 0])
    except AssertionError:
        pass
    else:
        raise AssertionError("wrong-direction filter shear was accepted")
    try:
        evaluate_filter_stability(stable_frames[:3])
    except AssertionError:
        pass
    else:
        raise AssertionError("filter with fewer than three steps was accepted")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--abacus", default="abacus",
                        help="ABACUS command, optionally including MPI launcher")
    parser.add_argument("--basis", choices=("pw", "lcao"), default="pw")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--precision", choices=("double", "single"), default="double")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--workdir", type=Path, default=Path("variable-cell-ase"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--pp-orb-root", type=Path,
                        default=Path(__file__).resolve().parents[3] / "tests" / "PP_ORB")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.self_test:
        _analytic_self_test()
        print("socketio_variable_cell self-test: PASS")
        return
    output = args.output or args.workdir / (
        "prepare.json" if args.prepare_only else "ase-validation.json")
    config = Config(args.abacus, args.basis, args.device, args.precision,
                    args.workdir.resolve(), output.resolve(),
                    args.pp_orb_root.resolve())
    if not config.pp_orb_root.is_dir():
        raise SystemExit("--pp-orb-root does not exist: {}".format(config.pp_orb_root))
    if args.prepare_only:
        fileio = prepare_case(config, config.workdir / "fileio", socket=False)
        socket = prepare_case(config, config.workdir / "socket", socket=True)
        manifest = build_prepare_manifest(
            config, {"fileio": fileio, "socket": socket}, "ase-variable-cell")
        config.output.parent.mkdir(parents=True, exist_ok=True)
        config.output.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
        print("prepared {} {} {} cases in {}".format(
            config.basis, config.device, config.precision, config.workdir))
        return
    run_validation(config)
    print("wrote {}".format(config.output))


if __name__ == "__main__":
    main()
