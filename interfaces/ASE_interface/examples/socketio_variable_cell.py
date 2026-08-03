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
HARTREE_EV = 27.211386245988
REQUIRED_FRAME_FIELDS = (
    "executable_version", "executable_sha256", "source_commit", "module",
    "backend", "device", "precision", "cell_angstrom",
    "precision_settings",
    "volume_angstrom3", "condition_number", "scf_converged", "energy_ev",
    "forces_ev_per_angstrom", "raw_abacus_stress_kbar",
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
            plus = float(energy(deform_atoms(atoms, component, +delta)))
            minus = float(energy(deform_atoms(atoms, component, -delta)))
            if not np.isfinite(plus) or not np.isfinite(minus):
                raise AssertionError("finite-difference energy is non-finite")
            fd = (plus - minus) / (2.0 * delta * volume)
            tolerance = (FD_LIMITS["atol_ev_per_angstrom3"]
                         + FD_LIMITS["rtol"] * abs(reference[component]))
            points.append({
                "delta": float(delta), "energy_plus_ev": plus,
                "energy_minus_ev": minus, "fd_stress_ev_per_angstrom3": fd,
                "analytic_stress_ev_per_angstrom3": float(reference[component]),
                "absolute_error": abs(fd - reference[component]),
                "atol_plus_rtol": tolerance,
                "pass": bool(abs(fd - reference[component]) <= tolerance),
            })
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
    missing = [field for field in REQUIRED_FRAME_FIELDS if field not in record]
    if missing:
        raise AssertionError("real frame is missing fields: " + ",".join(missing))
    assert_valid_frame(record["cell_angstrom"],
                       np.asarray(record["forces_ev_per_angstrom"], dtype=float))
    if record["scf_converged"] is not True:
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


def evaluate_filter_stability(frames: list[dict], minimum_steps: int = 3,
                              energy_atol_ev: float = 1.0e-10,
                              stress_atol_ev_per_angstrom3: float = 1.0e-10) -> dict:
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
    return {
        "accepted_steps": len(frames) - 1,
        "energy_decrease_ev": float(energy_drop),
        "stress_decrease_ev_per_angstrom3": float(stress_drop),
        "energy_atol_ev": float(energy_atol_ev),
        "stress_atol_ev_per_angstrom3": float(stress_atol_ev_per_angstrom3),
        "energy_decreased": bool(energy_drop > energy_atol_ev),
        "stress_decreased": bool(stress_drop > stress_atol_ev_per_angstrom3),
        "positive_determinant_and_finite": True,
    }


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
        inp["gint_precision"] = "double"
    kwargs = {
        "pseudopotentials": {"Si": "Si_ONCV_PBE-1.2.upf"},
        "inp": inp,
    }
    if config.basis == "lcao":
        kwargs["basissets"] = {"Si": "Si_gga_8au_100Ry_2s2p1d.orb"}
    return kwargs


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
                self.last_socket_virial_hartree = np.asarray(
                    results["virial"], dtype=np.float64).copy()
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
    if config.basis == "lcao" and not re.search(
            r"^\s*gint_precision\s+double\s*$", text,
            flags=re.MULTILINE | re.IGNORECASE):
        raise AssertionError("LCAO reference must write gint_precision double")
    return directory


def _find_log(directory: Path) -> Path:
    matches = sorted(directory.glob("OUT.*/running_scf.log"))
    if not matches:
        raise AssertionError("ABACUS running_scf.log is absent")
    return matches[-1]


def raw_stress_series_and_convergence(directory: Path,
                                      expected_frames: int = 1) -> tuple[list, int]:
    text = _find_log(directory).read_text(errors="replace")
    convergence_count = (text.count("#SCF IS CONVERGED#")
                         + text.count("charge density convergence is achieved"))
    if "convergence has not been achieved" in text.lower():
        raise AssertionError("ABACUS reported an unconverged SCF")
    lines = text.splitlines()
    blocks = []
    for index, line in enumerate(lines):
        if "TOTAL-STRESS" not in line.upper():
            continue
        rows = []
        for candidate in lines[index + 1:index + 12]:
            numbers = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?",
                                 candidate)
            if len(numbers) >= 3:
                rows.append([float(value) for value in numbers[-3:]])
                if len(rows) == 3:
                    break
        if len(rows) == 3:
            blocks.append(rows)
    if len(blocks) < expected_frames:
        raise AssertionError("raw ABACUS stress frames are incomplete")
    if convergence_count < expected_frames:
        raise AssertionError("ABACUS SCF convergence records are incomplete")
    return np.asarray(blocks, dtype=np.float64).tolist(), convergence_count


def _raw_stress_and_convergence(directory: Path) -> tuple[list, bool]:
    blocks, _ = raw_stress_series_and_convergence(directory, 1)
    return blocks[-1], True


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


def _frame_record(config: Config, identity: dict, atoms, directory: Path,
                  source: str, raw_socket_virial=None) -> dict:
    stress = np.asarray(atoms.get_stress(), dtype=np.float64)
    forces = np.asarray(atoms.get_forces(), dtype=np.float64)
    energy = float(atoms.get_potential_energy())
    raw_stress, converged = _raw_stress_and_convergence(directory)
    volume = float(atoms.get_volume())
    if source == "socket":
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
            "gint_precision": "double" if config.basis == "lcao" else None,
            "socket_float": "IEEE-754 binary64",
        },
        "cell_angstrom": np.asarray(atoms.cell, dtype=np.float64).tolist(),
        "volume_angstrom3": volume,
        "condition_number": float(np.linalg.cond(atoms.cell.array, 2)),
        "scf_converged": converged, "energy_ev": energy,
        "forces_ev_per_angstrom": forces.tolist(),
        "raw_abacus_stress_kbar": raw_stress,
        "socket_virial_hartree": virial.tolist(),
        "ase_stress_ev_per_angstrom3": stress.tolist(),
        "thresholds": {"identical": IDENTICAL_LIMITS,
                       "finite_difference": FD_LIMITS,
                       "cpu_gpu_double": DOUBLE_LIMITS,
                       "single_mixed_smoke_only": SMOKE_LIMITS},
        "virial_provenance": ("captured from ASE SocketServer before conversion"
                              if source == "socket" else
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
            accepted.append(frame)
        capture()
        optimizer.attach(capture, interval=1)
        optimizer.run(fmax=0.0, steps=3)
    if len(accepted) < 4:
        raise AssertionError(filter_name + " did not complete three accepted steps")
    result = evaluate_filter_stability(accepted)
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
            return varied.get_potential_energy()
        finite_difference = finite_difference_scan(
            socket_atoms, socket_energy, socket_stress)
    require_fd_plateau(finite_difference)
    energy_error = abs(socket_record["energy_ev"] - file_record["energy_ev"])
    force_error = float(np.max(np.abs(
        np.asarray(socket_record["forces_ev_per_angstrom"])
        - np.asarray(file_record["forces_ev_per_angstrom"]))))
    stress_error = np.abs(np.asarray(socket_record["ase_stress_ev_per_angstrom3"])
                          - np.asarray(file_record["ase_stress_ev_per_angstrom3"]))
    stress_tolerance = (IDENTICAL_LIMITS["atol_ev_per_angstrom3"]
                        + IDENTICAL_LIMITS["rtol"] * np.abs(
                            np.asarray(file_record["ase_stress_ev_per_angstrom3"])))
    identical = {
        "energy_absolute_error_ev": energy_error,
        "force_max_absolute_error_ev_per_angstrom": force_error,
        "stress_absolute_errors_ev_per_angstrom3": stress_error.tolist(),
        "stress_atol_plus_rtol": stress_tolerance.tolist(),
        "pass": bool(energy_error <= 1.0e-4 and force_error <= 1.0e-5
                     and np.all(stress_error <= stress_tolerance)),
    }
    if not identical["pass"]:
        raise AssertionError("identical-frame FileIO/socket comparison failed")
    shear = [record for record in finite_difference
             if record["component"] in ("yz", "xz", "xy")]
    if not any(abs(record["points"][1]["fd_stress_ev_per_angstrom3"])
               > FD_LIMITS["atol_ev_per_angstrom3"] for record in shear):
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
    frame = {field: 0.0 for field in REQUIRED_FRAME_FIELDS}
    frame.update({"cell_angstrom": np.eye(3).tolist(),
                  "forces_ev_per_angstrom": np.zeros((2, 3)).tolist(),
                  "scf_converged": True})
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
         "max_abs_stress_ev_per_angstrom3": 0.4},
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 1.8,
         "max_abs_stress_ev_per_angstrom3": 0.3},
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 1.6,
         "max_abs_stress_ev_per_angstrom3": 0.2},
        {"cell_angstrom": np.eye(3).tolist(), "energy_ev": 1.4,
         "max_abs_stress_ev_per_angstrom3": 0.1},
    ]
    assert evaluate_filter_stability(stable_frames)["stress_decreased"]
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
    output = args.output or args.workdir / "ase-validation.json"
    config = Config(args.abacus, args.basis, args.device, args.precision,
                    args.workdir.resolve(), output.resolve(),
                    args.pp_orb_root.resolve())
    if not config.pp_orb_root.is_dir():
        raise SystemExit("--pp-orb-root does not exist: {}".format(config.pp_orb_root))
    if args.prepare_only:
        prepare_case(config, config.workdir / "fileio", socket=False)
        prepare_case(config, config.workdir / "socket", socket=True)
        print("prepared {} {} {} cases in {}".format(
            config.basis, config.device, config.precision, config.workdir))
        return
    run_validation(config)
    print("wrote {}".format(config.output))


if __name__ == "__main__":
    main()
