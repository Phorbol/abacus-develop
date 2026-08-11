#!/usr/bin/env python3
"""Validate ABACUS variable-cell socket I/O against ASE and FileIO.

ASE Voigt order is xx, yy, zz, yz, xz, xy. Socket floating-point values are
binary64; validation uses eV/Angstrom units only as an independent outer check.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
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
IDENTICAL_DOUBLE_LIMITS = {
    "energy": {"rtol": 1.0e-8, "atol_ev": 1.0e-4},
    "force_max": {"rtol": 1.0e-5, "atol_ev_per_angstrom": 1.0e-5},
    "stress": IDENTICAL_LIMITS,
}
IDENTICAL_SMOKE_LIMITS = {
    "energy": {"rtol": 5.0e-5, "atol_ev": 2.0e-3},
    "force_max": {"rtol": 5.0e-2, "atol_ev_per_angstrom": 2.0e-3},
    "stress": SMOKE_LIMITS,
}
VOLUME_LIMITS = {"rtol": 1.0e-10, "atol_angstrom3": 1.0e-8}
FILTER_LIMITS = {
    "energy_rtol": 1.0e-8, "energy_atol_ev": 1.0e-7,
    "stress_rtol": 1.0e-4, "stress_atol_ev_per_angstrom3": 1.0e-7,
    "shear_rtol": 1.0e-4, "shear_atol": 1.0e-8,
    "gradient_rtol": 1.0e-4, "gradient_atol_ev_per_angstrom3": 1.0e-7,
}
REQUIRED_FRAME_FIELDS = (
    "executable_version", "executable_sha256", "source_commit", "module",
    "backend", "device", "precision", "cell_angstrom",
    "precision_settings",
    "volume_angstrom3", "condition_number", "scf_converged", "energy_ev",
    "atom_count", "forces_ev_per_angstrom", "raw_abacus_stress_kbar",
    "socket_virial_hartree", "ase_stress_ev_per_angstrom3", "thresholds",
)
SOURCE_COMMIT_PATTERN = re.compile(r"[0-9a-fA-F]{40}\n?")


@dataclass(frozen=True)
class Config:
    abacus: str
    basis: str
    device: str
    precision: str
    workdir: Path
    output: Path
    pp_orb_root: Path
    # Optional explicit Kohn-Sham solver. None preserves the historical
    # validation defaults while allowing the MPI matrix to exercise each one.
    ks_solver: str | None = None


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


def stable_filter_diamond_si8():
    """The stable full-cell ASE-filter fixture proven by the Task 9f run."""
    from ase import Atoms
    cell = np.array([
        [5.46258, 0.01629, -0.01086],
        [0.01629, 5.40285, 0.013575],
        [-0.01086, 0.013575, 5.42457],
    ], dtype=np.float64)
    fractional_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ], dtype=np.float64)
    return Atoms(
        ["Si"] * 8,
        scaled_positions=fractional_positions,
        cell=cell,
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
    frames_required: bool = False,
) -> list[dict]:
    """Central six-strain derivative divided by the unstrained volume."""
    reference = np.asarray(tuple(analytic_stress), dtype=np.float64)
    if reference.shape != (6,) or not np.all(np.isfinite(reference)):
        raise AssertionError("analytic ASE stress must contain six finite values")
    if type(frames_required) is not bool:
        raise AssertionError("frames_required must be boolean")
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
            if frames_required and (plus_frame is None or minus_frame is None):
                raise AssertionError(
                    "real finite differences require complete plus/minus frames")
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
                        "frames_required": frames_required,
                        "passing_points": len(passed),
                        "plateau_pass": bool(adjacent_plateau)})
    return records


def require_fd_plateau(records: list[dict], frames_required=None) -> None:
    if [record.get("component") for record in records] != list(VOIGT):
        raise AssertionError("finite-difference records use the wrong Voigt order")
    if frames_required is not None:
        if type(frames_required) is not bool:
            raise AssertionError("finite-difference frame contract must be boolean")
        for record in records:
            if record.get("frames_required") is not frames_required:
                raise AssertionError("finite-difference frame contract is inconsistent")
            if frames_required:
                for point in record.get("points", []):
                    if "plus_frame" not in point or "minus_frame" not in point:
                        raise AssertionError(
                            "real finite-difference point lacks plus/minus frames")
                    assert_real_frame_schema(point["plus_frame"])
                    assert_real_frame_schema(point["minus_frame"])
    failed = [record["component"] for record in records
              if record.get("passing_points", 0) < 2
              or not record.get("plateau_pass", False)]
    if failed:
        raise AssertionError("no multi-delta convergence plateau: " + ",".join(failed))


def virial_from_ase_stress(stress: Iterable[float], volume: float) -> np.ndarray:
    """Invert ASE sigma=-W/V and return a full virial in Hartree."""
    from ase import units
    from ase.stress import voigt_6_to_full_3x3_stress
    tensor = voigt_6_to_full_3x3_stress(
        np.asarray(tuple(stress), dtype=np.float64))
    return -float(volume) * tensor / units.Ha


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
    if not record["executable_version"]:
        raise AssertionError("identity strings must be nonempty")
    if re.fullmatch(r"[0-9a-f]{40}", record["source_commit"]) is None:
        raise AssertionError("source_commit must be normalized 40-hex")
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
    result_kind = payload.get("result_kind")
    frames_required = payload.get("finite_difference_frames_required")
    if result_kind not in ("real-validation", "analytic-self-test"):
        raise AssertionError("result_kind must distinguish real and analytic output")
    if (type(frames_required) is not bool
            or frames_required != (result_kind == "real-validation")):
        raise AssertionError("finite-difference frame requirement is inconsistent")
    for record in payload.get("frames", []):
        assert_real_frame_schema(record)
    assert_identical_frame_schema(payload["identical_frame"])
    require_fd_plateau(payload["finite_difference"], frames_required)
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
        if config.ks_solver is not None:
            inp["ks_solver"] = config.ks_solver
        elif config.device == "cpu":
            # The parallel-capable validation path avoids the CUDA-aware ELPA
            # host-array defect by using ABACUS's ScaLAPACK solver.
            inp["ks_solver"] = "scalapack_gvx"
    elif config.ks_solver is not None:
        inp["ks_solver"] = config.ks_solver
    kwargs = {
        "pseudopotentials": {"Si": "Si_ONCV_PBE-1.2.upf"},
        "inp": inp,
    }
    if config.basis == "lcao":
        kwargs["basissets"] = {"Si": "Si_gga_8au_100Ry_2s2p1d.orb"}
    return kwargs


def active_stress_limits(config: Config, reference_limits: dict) -> dict:
    return dict(reference_limits if config.precision == "double" else SMOKE_LIMITS)


def active_identical_limits(config: Config) -> dict:
    limits = (IDENTICAL_DOUBLE_LIMITS if config.precision == "double"
              else IDENTICAL_SMOKE_LIMITS)
    return copy.deepcopy(limits)


def identical_frame_decision(
        energy_error_ev: float, energy_reference_abs_ev: float,
        force_error_ev_per_angstrom: float,
        force_reference_max_abs_ev_per_angstrom: float,
        stress_errors_ev_per_angstrom3,
        stress_reference_abs_ev_per_angstrom3,
        thresholds: dict) -> dict:
    try:
        energy_limits = thresholds["energy"]
        force_limits = thresholds["force_max"]
        stress_limits = thresholds["stress"]
        threshold_values = (
            energy_limits["atol_ev"], energy_limits["rtol"],
            force_limits["atol_ev_per_angstrom"], force_limits["rtol"],
            stress_limits["atol_ev_per_angstrom3"], stress_limits["rtol"])
    except (KeyError, TypeError) as error:
        raise AssertionError("identical-frame thresholds are incomplete") from error
    scalars = (energy_error_ev, energy_reference_abs_ev,
               force_error_ev_per_angstrom,
               force_reference_max_abs_ev_per_angstrom) + threshold_values
    if (any(isinstance(value, bool) or not np.isscalar(value)
            or not np.isfinite(value) or value < 0.0 for value in scalars)):
        raise AssertionError("identical-frame values must be finite and nonnegative")
    stress_errors = np.asarray(stress_errors_ev_per_angstrom3, dtype=np.float64)
    stress_reference = np.asarray(
        stress_reference_abs_ev_per_angstrom3, dtype=np.float64)
    if (stress_errors.shape != (6,) or stress_reference.shape != (6,)
            or not np.all(np.isfinite(stress_errors))
            or not np.all(np.isfinite(stress_reference))
            or np.any(stress_errors < 0.0) or np.any(stress_reference < 0.0)):
        raise AssertionError("identical-frame stress values must be six nonnegative values")
    energy_tolerance = (energy_limits["atol_ev"]
                        + energy_limits["rtol"] * energy_reference_abs_ev)
    force_tolerance = (force_limits["atol_ev_per_angstrom"]
                       + force_limits["rtol"]
                       * force_reference_max_abs_ev_per_angstrom)
    stress_tolerance = (stress_limits["atol_ev_per_angstrom3"]
                        + stress_limits["rtol"] * stress_reference)
    energy_pass = bool(energy_error_ev <= energy_tolerance)
    force_pass = bool(force_error_ev_per_angstrom <= force_tolerance)
    stress_pass = bool(np.all(stress_errors <= stress_tolerance))
    return {
        "energy_absolute_error_ev": float(energy_error_ev),
        "energy_reference_abs_ev": float(energy_reference_abs_ev),
        "force_max_absolute_error_ev_per_angstrom":
            float(force_error_ev_per_angstrom),
        "force_reference_max_abs_ev_per_angstrom":
            float(force_reference_max_abs_ev_per_angstrom),
        "stress_absolute_errors_ev_per_angstrom3": stress_errors.tolist(),
        "stress_reference_abs_ev_per_angstrom3": stress_reference.tolist(),
        "energy_atol_plus_rtol_ev": float(energy_tolerance),
        "force_atol_plus_rtol_ev_per_angstrom": float(force_tolerance),
        "stress_atol_plus_rtol_ev_per_angstrom3": stress_tolerance.tolist(),
        "energy_pass": energy_pass, "force_pass": force_pass,
        "stress_pass": stress_pass,
        "pass": bool(energy_pass and force_pass and stress_pass),
        "thresholds": copy.deepcopy(thresholds),
    }


def assert_identical_frame_schema(record: dict) -> None:
    required = (
        "energy_absolute_error_ev", "energy_reference_abs_ev",
        "force_max_absolute_error_ev_per_angstrom",
        "force_reference_max_abs_ev_per_angstrom",
        "stress_absolute_errors_ev_per_angstrom3",
        "stress_reference_abs_ev_per_angstrom3",
        "energy_atol_plus_rtol_ev",
        "force_atol_plus_rtol_ev_per_angstrom",
        "stress_atol_plus_rtol_ev_per_angstrom3",
        "energy_pass", "force_pass", "stress_pass", "pass", "thresholds",
        "is_reference")
    missing = [key for key in required if key not in record]
    if missing:
        raise AssertionError("identical-frame JSON is missing: " + ",".join(missing))
    replay = identical_frame_decision(
        record["energy_absolute_error_ev"], record["energy_reference_abs_ev"],
        record["force_max_absolute_error_ev_per_angstrom"],
        record["force_reference_max_abs_ev_per_angstrom"],
        record["stress_absolute_errors_ev_per_angstrom3"],
        record["stress_reference_abs_ev_per_angstrom3"], record["thresholds"])
    for key in ("energy_atol_plus_rtol_ev",
                "force_atol_plus_rtol_ev_per_angstrom"):
        if not np.isclose(record[key], replay[key], rtol=0.0, atol=1.0e-15):
            raise AssertionError("identical-frame recorded tolerance is stale")
    if not np.allclose(
            record["stress_atol_plus_rtol_ev_per_angstrom3"],
            replay["stress_atol_plus_rtol_ev_per_angstrom3"],
            rtol=0.0, atol=1.0e-15):
        raise AssertionError("identical-frame stress tolerances are stale")
    for key in ("energy_pass", "force_pass", "stress_pass", "pass"):
        if type(record[key]) is not bool or record[key] != replay[key]:
            raise AssertionError("identical-frame decision cannot be replayed")
    if type(record["is_reference"]) is not bool:
        raise AssertionError("identical-frame is_reference must be boolean")


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
    solvers = [value.lower() for value in re.findall(
        r"^\s*ks_solver\s+(\S+)\s*$", text,
        flags=re.MULTILINE | re.IGNORECASE)]
    expected_solver = config.ks_solver
    if expected_solver is None and config.basis == "lcao" and config.device == "cpu":
        expected_solver = "scalapack_gvx"
    if expected_solver is not None:
        if solvers != [expected_solver.lower()]:
            raise AssertionError(
                "validation requires exact ks_solver {}".format(expected_solver))
    elif solvers:
        raise AssertionError("ks_solver is not expected for this validation case")
    return directory


def infer_mpi_ranks(command: str) -> int:
    """Return rank count from a conventional mpirun/srun command.

    This is provenance only; the launcher remains user-controlled. Unknown
    launchers intentionally report one rather than guessing from environment.
    """
    tokens = shlex.split(command)
    for index, token in enumerate(tokens[:-1]):
        if token in ("-np", "-n", "--np", "--ntasks"):
            try:
                ranks = int(tokens[index + 1])
            except (TypeError, ValueError):
                break
            if ranks > 0:
                return ranks
    return 1


def effective_ks_solver(directory: Path, requested: str | None = None) -> str:
    """Read the solver ABACUS resolved in INPUT.info when available."""
    candidates = sorted(directory.glob("OUT.*/INPUT.info"))
    for path in reversed(candidates):
        text = path.read_text(errors="replace")
        match = re.search(r"^\s*ks_solver\s+(\S+)\s*$", text,
                          flags=re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).lower()
    if requested is not None:
        return requested.lower()
    return "abacus-default"

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


def _normalize_source_commit(value: str, source: str) -> str:
    if SOURCE_COMMIT_PATTERN.fullmatch(value) is None:
        raise AssertionError(
            "{} must contain exactly one 40-hex source commit".format(source))
    return (value[:-1] if value.endswith("\n") else value).lower()


def resolve_source_commit(script_path: Path | None = None) -> str:
    """Resolve provenance from one explicit source or staged-runtime layout."""
    script = Path(os.path.abspath(os.fspath(script_path or __file__)))
    examples = script.parent
    ase_interface = examples.parent
    layout_root = ase_interface.parent
    if examples.name != "examples" or ase_interface.name != "ASE_interface":
        raise AssertionError("source commit resolver received an unknown layout")
    for component in (script, examples, ase_interface, layout_root):
        if component.is_symlink():
            raise AssertionError(
                "source commit resolver rejects symlink component {}".format(
                    component))
    if not script.is_file():
        raise AssertionError("source commit resolver script is not a file")
    if layout_root.name == "interfaces":
        repository = layout_root.parent
        if repository.is_symlink():
            raise AssertionError("source checkout repository root is a symlink")
        top_level = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"], cwd=repository,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False)
        if top_level.returncode != 0:
            raise AssertionError("cannot resolve source checkout Git root")
        reported_root = top_level.stdout[:-1] if top_level.stdout.endswith("\n") \
            else top_level.stdout
        if (not reported_root
                or "\n" in reported_root
                or Path(reported_root).resolve() != repository.resolve()):
            raise AssertionError("source checkout Git root does not match layout")
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"], cwd=repository,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=False)
        if head.returncode != 0:
            raise AssertionError("cannot resolve source checkout commit")
        return _normalize_source_commit(head.stdout, "git rev-parse HEAD")
    marker = layout_root / "SOURCE_COMMIT"
    if marker.is_symlink():
        raise AssertionError("staged source commit marker is a symlink")
    if not marker.is_file():
        raise AssertionError("staged runtime is missing {}".format(marker))
    try:
        value = marker.read_bytes().decode("ascii")
    except (OSError, UnicodeDecodeError) as error:
        raise AssertionError("cannot read staged source commit marker") from error
    return _normalize_source_commit(value, str(marker))


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
    source_commit = resolve_source_commit()
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
            or not identity["executable_version"]
            or re.fullmatch(r"[0-9a-fA-F]{64}",
                            identity["executable_sha256"]) is None
            or re.fullmatch(r"[0-9a-f]{40}",
                            identity["source_commit"]) is None):
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
    atoms = stable_filter_diamond_si8()
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
            limits=active_stress_limits(config, FD_LIMITS),
            frames_required=True)
    require_fd_plateau(finite_difference)
    energy_error = abs(socket_record["energy_ev"] - file_record["energy_ev"])
    force_error = float(np.max(np.abs(
        np.asarray(socket_record["forces_ev_per_angstrom"])
        - np.asarray(file_record["forces_ev_per_angstrom"]))))
    stress_error = np.abs(np.asarray(socket_record["ase_stress_ev_per_angstrom3"])
                          - np.asarray(file_record["ase_stress_ev_per_angstrom3"]))
    file_forces = np.asarray(file_record["forces_ev_per_angstrom"])
    file_stress = np.asarray(file_record["ase_stress_ev_per_angstrom3"])
    identical = identical_frame_decision(
        energy_error, abs(file_record["energy_ev"]), force_error,
        float(np.max(np.abs(file_forces))), stress_error,
        np.abs(file_stress), active_identical_limits(config))
    identical["is_reference"] = config.precision == "double"
    if not identical["pass"]:
        raise AssertionError("identical-frame FileIO/socket comparison failed")
    shear = [record for record in finite_difference
             if record["component"] in ("yz", "xz", "xy")]
    if not any(abs(record["points"][1]["fd_stress_ev_per_angstrom3"])
               > active_stress_limits(config, FD_LIMITS)["atol_ev_per_angstrom3"]
               for record in shear):
        raise AssertionError("no nonzero finite-difference-consistent shear response")
    payload = {
        "schema_version": 1, "result_kind": "real-validation",
        "finite_difference_frames_required": True,
        "voigt_order": list(VOIGT),
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
    from ase.stress import full_3x3_to_voigt_6_stress
    self_test_script = Path(__file__).resolve()
    if self_test_script.parents[2].name == "interfaces":
        expected_checkout_commit = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD"],
            cwd=self_test_script.parents[3], text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            check=True).stdout.strip().lower()
        assert resolve_source_commit() == expected_checkout_commit
    else:
        assert resolve_source_commit() == _normalize_source_commit(
            (self_test_script.parents[2] / "SOURCE_COMMIT").read_text(),
            "self-test staged marker")
    valid_marker_commit = "a" * 40
    with tempfile.TemporaryDirectory(
            prefix="task7-source-commit-selftest-") as temporary:
        temporary_root = Path(temporary)
        runtime = temporary_root / "runtime"
        staged_script = runtime / "ASE_interface" / "examples" / Path(__file__).name
        staged_script.parent.mkdir(parents=True)
        staged_script.write_text("# staged layout probe\n")
        marker = runtime / "SOURCE_COMMIT"
        marker.write_text(valid_marker_commit.upper() + "\n")
        assert resolve_source_commit(staged_script) == valid_marker_commit
        invalid_markers = {
            "empty": "",
            "nonhex": "g" * 40,
            "39-hex": "a" * 39,
            "41-hex": "a" * 41,
            "leading-space": " " + "a" * 40 + "\n",
            "trailing-space": "a" * 40 + " \n",
            "extra-blank-line": "a" * 40 + "\n\n",
            "crlf": "a" * 40 + "\r\n",
            "multiple-conflicting": "a" * 40 + "\n" + "b" * 40,
        }
        accepted_markers = []
        for label, value in invalid_markers.items():
            marker.write_text(value)
            try:
                resolve_source_commit(staged_script)
            except AssertionError:
                pass
            else:
                accepted_markers.append(label)
        if accepted_markers:
            raise AssertionError("non-exact source commit markers accepted: {}".format(
                ",".join(accepted_markers)))
        marker.unlink()
        try:
            resolve_source_commit(staged_script)
        except AssertionError:
            pass
        else:
            raise AssertionError("missing source commit marker was accepted")

        def initialize_repository(repository: Path) -> str:
            repository.mkdir(parents=True)
            subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
            (repository / "tracked").write_text("provenance self-test\n")
            subprocess.run(["git", "add", "tracked"], cwd=repository, check=True)
            subprocess.run([
                "git", "-c", "user.name=Task 7 self-test",
                "-c", "user.email=task7@example.invalid",
                "commit", "-qm", "provenance self-test",
            ], cwd=repository, check=True)
            return subprocess.run(
                ["git", "rev-parse", "--verify", "HEAD"], cwd=repository,
                text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=True).stdout.strip().lower()

        exact_repository = temporary_root / "exact-repository"
        exact_commit = initialize_repository(exact_repository)
        exact_script = (exact_repository / "interfaces" / "ASE_interface"
                        / "examples" / "probe.py")
        exact_script.parent.mkdir(parents=True)
        exact_script.write_text("# exact source checkout probe\n")
        assert resolve_source_commit(exact_script) == exact_commit

        parent_repository = temporary_root / "parent-repository"
        initialize_repository(parent_repository)
        nested_script = (parent_repository / "nested" / "interfaces"
                         / "ASE_interface" / "examples" / "probe.py")
        nested_script.parent.mkdir(parents=True)
        nested_script.write_text("# parent Git collision probe\n")

        symlink_runtime = temporary_root / "symlink-runtime"
        symlink_script = (symlink_runtime / "ASE_interface" / "examples"
                          / "probe.py")
        symlink_script.parent.mkdir(parents=True)
        symlink_script.symlink_to(self_test_script)
        (symlink_runtime / "SOURCE_COMMIT").write_text("b" * 40 + "\n")

        component_runtime = temporary_root / "component-runtime"
        component_runtime.mkdir()
        (component_runtime / "SOURCE_COMMIT").write_text("c" * 40 + "\n")
        (component_runtime / "ASE_interface").symlink_to(
            self_test_script.parents[1], target_is_directory=True)
        component_script = (component_runtime / "ASE_interface" / "examples"
                            / self_test_script.name)

        accepted_unsafe_layouts = []
        for label, unsafe_script in (
                ("parent-git-collision", nested_script),
                ("staged-script-symlink", symlink_script),
                ("staged-component-symlink", component_script)):
            try:
                resolve_source_commit(unsafe_script)
            except AssertionError:
                pass
            else:
                accepted_unsafe_layouts.append(label)
        if accepted_unsafe_layouts:
            raise AssertionError("unsafe provenance layouts accepted: {}".format(
                ",".join(accepted_unsafe_layouts)))
    print("source commit provenance probe: checkout/staged exact 40-hex PASS; "
          "strict marker and lexical Git/symlink collisions rejected")
    atoms = displaced_triclinic_si2()
    filter_fixture_factory = globals().get("stable_filter_diamond_si8")
    if not callable(filter_fixture_factory):
        raise AssertionError("dedicated stable diamond-Si8 filter fixture is missing")
    filter_atoms = filter_fixture_factory()
    expected_filter_cell = np.array([
        [5.46258, 0.01629, -0.01086],
        [0.01629, 5.40285, 0.013575],
        [-0.01086, 0.013575, 5.42457],
    ], dtype=np.float64)
    expected_filter_fractions = np.array([
        [0.0, 0.0, 0.0],
        [0.25, 0.25, 0.25],
        [0.0, 0.5, 0.5],
        [0.25, 0.75, 0.75],
        [0.5, 0.0, 0.5],
        [0.75, 0.25, 0.75],
        [0.5, 0.5, 0.0],
        [0.75, 0.75, 0.25],
    ], dtype=np.float64)
    if (filter_atoms.get_chemical_symbols() != ["Si"] * 8
            or not np.all(filter_atoms.pbc)
            or filter_atoms.cell.array.dtype != np.float64
            or filter_atoms.positions.dtype != np.float64
            or not np.array_equal(filter_atoms.cell.array, expected_filter_cell)
            or not np.array_equal(
                filter_atoms.positions,
                expected_filter_fractions @ expected_filter_cell)
            or not np.allclose(
                filter_atoms.get_scaled_positions(wrap=False),
                expected_filter_fractions, rtol=0.0, atol=4.0e-16)):
        raise AssertionError("stable filter fixture identity/cell/positions changed")
    filter_volume = float(filter_atoms.get_volume())
    filter_condition = float(np.linalg.cond(filter_atoms.cell.array, 2))
    filter_density = (float(np.sum(filter_atoms.get_masses()))
                      * units._amu * 1.0e27 / filter_volume)
    filter_distances = filter_atoms.get_all_distances(mic=True)
    np.fill_diagonal(filter_distances, np.inf)
    filter_mic_distance = float(np.min(filter_distances))
    filter_shears = filter_atoms.cell.array[[0, 0, 1], [1, 2, 2]]
    if (not 160.09 < filter_volume < 160.10
            or not 1.0 < filter_condition < 2.0
            or not 2.32 < filter_density < 2.34
            or not 2.33 < filter_mic_distance < 2.35
            or not np.all(filter_shears != 0.0)
            or len(set(filter_shears.tolist())) != 3):
        raise AssertionError("stable filter fixture physical invariants changed")

    solver_probe_root = Path("unused")
    assert infer_mpi_ranks("mpirun -np 2 abacus") == 2
    assert infer_mpi_ranks("srun --ntasks 4 abacus") == 4
    assert infer_mpi_ranks("abacus") == 1
    assert effective_ks_solver(Path("absent"), "genelpa") == "genelpa"

    solver_cases = (
        ("lcao-cpu-double", "lcao", "cpu", "double", 1.0e-9,
         "double", "scalapack_gvx"),
        ("lcao-cpu-single", "lcao", "cpu", "single", 1.0e-6,
         "mix", "scalapack_gvx"),
        ("lcao-gpu-double", "lcao", "gpu", "double", 1.0e-9,
         "double", None),
        ("lcao-gpu-single", "lcao", "gpu", "single", 1.0e-6,
         "mix", None),
        ("pw-cpu-double", "pw", "cpu", "double", 1.0e-9, None, None),
        ("pw-cpu-single", "pw", "cpu", "single", 1.0e-6, None, None),
        ("pw-gpu-double", "pw", "gpu", "double", 1.0e-9, None, None),
        ("pw-gpu-single", "pw", "gpu", "single", 1.0e-6, None, None),
    )
    for (label, basis, device, precision, scf_thr,
         gint_precision, solver) in solver_cases:
        solver_config = Config(
            abacus="unused", basis=basis, device=device, precision=precision,
            workdir=solver_probe_root, output=solver_probe_root,
            pp_orb_root=solver_probe_root, ks_solver=solver)
        actual_kwargs = _common_kwargs(solver_config)
        actual_inp = dict(actual_kwargs["inp"])
        actual_solver = actual_inp.pop("ks_solver", None)
        expected_inp = {
            "calculation": "scf", "basis_type": basis,
            "device": device, "precision": precision,
            "ecutwfc": 50, "symmetry": 0, "kspacing": 0.45,
            "scf_thr": scf_thr,
            "scf_nmax": 100, "chg_extrap": "atomic", "cal_force": 1,
            "cal_stress": 1,
        }
        if gint_precision is not None:
            expected_inp["gint_precision"] = gint_precision
        expected_kwargs = {
            "pseudopotentials": {"Si": "Si_ONCV_PBE-1.2.upf"},
            "inp": expected_inp,
        }
        if basis == "lcao":
            expected_kwargs["basissets"] = {
                "Si": "Si_gga_8au_100Ry_2s2p1d.orb"}
        actual_without_solver = dict(actual_kwargs)
        actual_without_solver["inp"] = actual_inp
        if actual_without_solver != expected_kwargs:
            raise AssertionError(
                label + " validation kwargs changed beyond ks_solver")
        if actual_solver != solver:
            raise AssertionError(
                label + " must use exact ks_solver scalapack_gvx"
                if solver == "scalapack_gvx" else
                label + " must not set ks_solver")

    class InputWriterProbe:
        def __init__(self, directory, inp):
            self.directory = Path(directory)
            self.inp = copy.deepcopy(inp)

        def write_input(self, _atoms, properties):
            if properties != ["energy", "forces", "stress"]:
                raise AssertionError("prepare_case requested wrong properties")
            lines = ["{} {}".format(key, value)
                     for key, value in self.inp.items()]
            (self.directory / "INPUT").write_text("\n".join(lines) + "\n")

    class FileIOProbe(InputWriterProbe):
        def __init__(self, profile, directory, **kwargs):
            if profile is None:
                raise AssertionError("prepare_case omitted profile")
            super().__init__(directory, kwargs["inp"])

    class SocketIOProbe:
        def __init__(self, profile, directory, unixsocket, variable_cell,
                     **kwargs):
            if profile is None or not unixsocket or variable_cell is not True:
                raise AssertionError("prepare_case socket setup changed")
            socket_inp = copy.deepcopy(kwargs["inp"])
            socket_inp["socket_variable_cell"] = 1
            self.abacus = InputWriterProbe(directory, socket_inp)

        def close(self):
            pass

    current_module = sys.modules[__name__]
    with tempfile.TemporaryDirectory(
            prefix="task9p-solver-input-selftest-") as temporary:
        input_root = Path(temporary)
        with patch.object(
                current_module, "_load_abacus_api",
                return_value=(FileIOProbe, object, SocketIOProbe)), \
                patch.object(current_module, "_profile", return_value=object()):
            for (label, basis, device, precision, _scf_thr,
                 _gint, solver) in solver_cases:
                solver_config = Config(
                    abacus="unused", basis=basis, device=device,
                    precision=precision, workdir=input_root,
                    output=input_root / "unused.json",
                    pp_orb_root=input_root, ks_solver=solver)
                for socket in (False, True):
                    case_label = label + ("-socket" if socket else "-fileio")
                    prepared = prepare_case(
                        solver_config, input_root / case_label, socket=socket)
                    input_text = (prepared / "INPUT").read_text()
                    input_solvers = re.findall(
                        r"^\s*ks_solver\s+(\S+)\s*$", input_text,
                        flags=re.MULTILINE | re.IGNORECASE)
                    expected_solvers = [] if solver is None else [solver]
                    if ([value.lower() for value in input_solvers]
                            != expected_solvers):
                        raise AssertionError(
                            case_label + " generated INPUT has wrong ks_solver")

            cpu_lcao_config = Config(
                abacus="unused", basis="lcao", device="cpu",
                precision="double", workdir=input_root,
                output=input_root / "unused.json", pp_orb_root=input_root)
            cpu_lcao_kwargs = _common_kwargs(cpu_lcao_config)
            for fallback_solver in ("lapack", "genelpa", None):
                fallback_kwargs = copy.deepcopy(cpu_lcao_kwargs)
                if fallback_solver is None:
                    fallback_kwargs["inp"].pop("ks_solver")
                    fallback_label = "implicit genelpa"
                else:
                    fallback_kwargs["inp"]["ks_solver"] = fallback_solver
                    fallback_label = fallback_solver
                try:
                    with patch.object(
                            current_module, "_common_kwargs",
                            return_value=fallback_kwargs):
                        prepare_case(
                            cpu_lcao_config,
                            input_root / ("fallback-" + fallback_label.replace(" ", "-")),
                            socket=False)
                except AssertionError as error:
                    if str(error) != "validation requires exact ks_solver scalapack_gvx":
                        raise
                else:
                    raise AssertionError(
                        "prepare_case accepted CPU LCAO INPUT with ks_solver "
                        + fallback_label)

    class FilterFixtureObserved(Exception):
        pass

    class ProbeCalculator:
        def __init__(self, **_kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, _exception_type, _exception, _traceback):
            return False

    def observe_filter_fixture(observed_atoms):
        if (observed_atoms.get_chemical_symbols() != ["Si"] * 8
                or not np.array_equal(
                    observed_atoms.cell.array, expected_filter_cell)
                or not np.array_equal(
                    observed_atoms.positions, filter_atoms.positions)):
            raise AssertionError(
                "filter runner did not use stable diamond-Si8 fixture")
        raise FilterFixtureObserved

    import ase.filters
    probe_config = Config(
        abacus="unused", basis="pw", device="cpu", precision="double",
        workdir=Path("unused"), output=Path("unused"),
        pp_orb_root=Path("unused"))
    with tempfile.TemporaryDirectory(
            prefix="task9g-filter-fixture-selftest-") as temporary:
        try:
            with patch.object(
                    current_module, "_recording_socket_class",
                    return_value=ProbeCalculator), patch.object(
                        current_module, "_profile", return_value=object()), \
                    patch.object(
                        current_module, "_common_kwargs", return_value={}), \
                    patch.object(
                        current_module, "_identity", return_value={}), \
                    patch.object(
                        ase.filters, "UnitCellFilter",
                        new=observe_filter_fixture):
                _filter_run(
                    probe_config, "unit_cell_filter",
                    Path(temporary) / "filter-run")
        except FilterFixtureObserved:
            pass
        else:
            raise AssertionError(
                "filter runner did not reach stable diamond-Si8 fixture")
    base_cell = atoms.cell.array.copy()
    volume = atoms.get_volume()
    reference_strain = np.array([0.02, -0.015, 0.01, 0.012, -0.009, 0.017])
    stiffness = np.array([1.2, 1.5, 1.8, 0.9, 1.1, 1.4])
    expected = stiffness * reference_strain / volume

    capture_volume = 73.25
    capture_virial_ev = np.array([
        [4.0, -0.75, 1.125],
        [-0.75, -2.5, 0.375],
        [1.125, 0.375, 3.25],
    ], dtype=np.float64)
    capture_stress = (-full_3x3_to_voigt_6_stress(capture_virial_ev)
                      / capture_volume)
    captured_virial_hartree = capture_virial_ev / units.Ha
    try:
        validate_virial_sign(
            capture_stress, captured_virial_hartree, capture_volume)
    except AssertionError as error:
        mismatch = np.max(np.abs(
            captured_virial_hartree
            - virial_from_ase_stress(capture_stress, capture_volume)))
        raise AssertionError(
            "capture-path virial closure mismatch: {:.17g} Ha".format(
                mismatch)) from error

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
        "source_commit": "a" * 40, "module": "self-test",
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
        "invalid-source-commit": {"source_commit": "unknown"},
        "unnormalized-source-commit": {"source_commit": "A" * 40},
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
                     "source_commit": "a" * 40, "module": "self-test"},
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
            ("invalid-source-commit",
             lambda value: value["identity"].update(
                 {"source_commit": "unknown"})),
            ("unnormalized-source-commit",
             lambda value: value["identity"].update(
                 {"source_commit": "A" * 40})),
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
    analytic_identical = identical_frame_decision(
        0.0, 0.0, 0.0, 0.0, np.zeros(6), np.zeros(6),
        copy.deepcopy(IDENTICAL_DOUBLE_LIMITS))
    analytic_identical["is_reference"] = True
    payload = {"schema_version": 1,
               "result_kind": "analytic-self-test",
               "finite_difference_frames_required": False,
               "frames": [frame],
               "identical_frame": analytic_identical,
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
    missing_identical_thresholds = json.loads(json.dumps(payload))
    missing_identical_thresholds["identical_frame"] = {"pass": True}
    try:
        assert_json_schema(missing_identical_thresholds)
    except AssertionError:
        pass
    else:
        raise AssertionError("missing identical-frame thresholds were accepted")
    replay_errors = np.full(6, 5.0e-4)
    double_decision = identical_frame_decision(
        5.0e-4, 0.0, 5.0e-4, 0.0, replay_errors, np.zeros(6),
        copy.deepcopy(IDENTICAL_DOUBLE_LIMITS))
    smoke_decision = identical_frame_decision(
        5.0e-4, 0.0, 5.0e-4, 0.0, replay_errors, np.zeros(6),
        copy.deepcopy(IDENTICAL_SMOKE_LIMITS))
    assert not double_decision["pass"] and smoke_decision["pass"]
    tight_limits = copy.deepcopy(IDENTICAL_SMOKE_LIMITS)
    tight_limits["energy"].update({"atol_ev": 1.0e-6, "rtol": 0.0})
    assert not identical_frame_decision(
        5.0e-4, 0.0, 5.0e-4, 0.0, replay_errors, np.zeros(6),
        tight_limits)["pass"]
    assert not identical_frame_decision(
        1.0e-2, 0.0, 1.0e-2, 0.0, np.full(6, 1.0e-2), np.zeros(6),
        copy.deepcopy(IDENTICAL_SMOKE_LIMITS))["pass"]
    stale_threshold_decision = copy.deepcopy(smoke_decision)
    stale_threshold_decision["is_reference"] = False
    stale_threshold_decision["thresholds"]["stress"][
        "atol_ev_per_angstrom3"] = 1.0e-6
    try:
        assert_identical_frame_schema(stale_threshold_decision)
    except AssertionError:
        pass
    else:
        raise AssertionError("stale identical-frame threshold replay was accepted")
    real_fd_without_frames = copy.deepcopy(payload)
    real_fd_without_frames["result_kind"] = "real-validation"
    real_fd_without_frames["finite_difference_frames_required"] = True
    for record in real_fd_without_frames["finite_difference"]:
        record["frames_required"] = True
    try:
        assert_json_schema(real_fd_without_frames)
    except AssertionError:
        pass
    else:
        raise AssertionError("real finite differences without frames were accepted")
    complete_real_fd = copy.deepcopy(payload)
    complete_real_fd["result_kind"] = "real-validation"
    complete_real_fd["finite_difference_frames_required"] = True
    for record in complete_real_fd["finite_difference"]:
        record["frames_required"] = True
        for point in record["points"]:
            point["plus_frame"] = copy.deepcopy(frame)
            point["minus_frame"] = copy.deepcopy(frame)
    assert_json_schema(complete_real_fd)
    single_missing_fd = copy.deepcopy(complete_real_fd)
    del single_missing_fd["finite_difference"][0]["points"][0]["minus_frame"]
    try:
        assert_json_schema(single_missing_fd)
    except AssertionError:
        pass
    else:
        raise AssertionError("single missing finite-difference frame was accepted")
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
