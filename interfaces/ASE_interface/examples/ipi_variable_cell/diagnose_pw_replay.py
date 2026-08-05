#!/usr/bin/env python3
"""Replay saved GPU PW socket frames through fresh ABACUS calculations."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from ase.io import read

HERE = Path(__file__).resolve().parent
EXAMPLES = HERE.parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
if str(EXAMPLES) not in sys.path:
    sys.path.insert(0, str(EXAMPLES))
import run_validation
import socketio_variable_cell as ase_validation

DEFAULT_FRAME_INDICES: tuple[int, ...] = (0, 1, 5, 8, 42, 50)


def _validate_indices(indices: tuple[int, ...], frame_count: int) -> None:
    if (type(indices) is not tuple or not indices
            or isinstance(frame_count, bool) or not isinstance(frame_count, int)
            or frame_count <= 0):
        raise AssertionError("frame indices and count are invalid")
    if any(isinstance(index, bool) or not isinstance(index, int)
           for index in indices):
        raise AssertionError("frame indices must be integers")
    if tuple(sorted(set(indices))) != indices:
        raise AssertionError("frame indices must be unique and sorted")
    if indices[0] < 0 or indices[-1] >= frame_count:
        raise AssertionError("frame index is out of range")


def parse_frame_indices(value: str, frame_count: int) -> tuple[int, ...]:
    """Parse a nonempty, sorted, unique comma-separated frame selection."""
    if not isinstance(value, str) or not value:
        raise ValueError("frame selection must be nonempty")
    try:
        indices = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise ValueError("frame selection must contain integers") from error
    _validate_indices(indices, frame_count)
    return indices


def load_replay_frames(result_path: Path, positions_path: Path,
                       indices: tuple[int, ...]) -> tuple[dict, list[dict]]:
    """Load selected positions and replace rounded XYZ cells with JSON cells."""
    payload = json.loads(Path(result_path).read_text())
    run_validation.assert_json_ready(payload)
    if not isinstance(payload, dict):
        raise AssertionError("validation result must be a JSON object")
    required_identity = {
        "ipi_version": "3.2.0",
        "backend": "pw",
        "device": "gpu",
        "precision": "double",
    }
    if any(payload.get(key) != expected
           for key, expected in required_identity.items()):
        raise AssertionError("source must be an i-PI 3.2.0 GPU PW/double result")
    trajectory = payload.get("trajectory")
    if not isinstance(trajectory, dict) or not isinstance(
            trajectory.get("steps"), list):
        raise AssertionError("source trajectory steps are missing")
    stored_steps = trajectory["steps"]
    xyz_frames = read(str(positions_path), index=":")
    if not isinstance(xyz_frames, list):
        xyz_frames = [xyz_frames]
    if len(stored_steps) != len(xyz_frames):
        raise AssertionError("JSON and XYZ frame counts differ")
    _validate_indices(indices, len(stored_steps))
    if indices == DEFAULT_FRAME_INDICES and len(stored_steps) != 51:
        raise AssertionError("the default Job 762695 replay requires 51 frames")
    symbols = xyz_frames[0].get_chemical_symbols()
    if any(frame.get_chemical_symbols() != symbols for frame in xyz_frames):
        raise AssertionError("XYZ atom symbols differ between frames")

    frames = []
    for index in indices:
        stored_step = stored_steps[index]
        if not isinstance(stored_step, dict) or "cell_angstrom" not in stored_step:
            raise AssertionError("stored replay frame lacks its cell")
        try:
            json_cell = np.asarray(
                stored_step["cell_angstrom"], dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise AssertionError("stored replay cell must be numeric") from error
        xyz_atoms = xyz_frames[index]
        atoms = xyz_atoms.copy()
        atoms.set_cell(json_cell, scale_atoms=False)
        ase_validation.assert_valid_frame(json_cell, atoms.positions)
        frames.append({
            "index": index,
            "atoms": atoms,
            "stored_gpu": stored_step,
            "xyz_cell_max_abs_delta_angstrom": float(
                np.max(np.abs(xyz_atoms.cell.array - json_cell))),
        })
    return payload, frames


def compare_records(reference: dict, candidate: dict) -> dict:
    """Compare two energy/force/stress records with existing PW thresholds."""
    if not isinstance(reference, dict) or not isinstance(candidate, dict):
        raise AssertionError("comparison records must be dictionaries")
    try:
        reference_energy = reference["energy_ev"]
        candidate_energy = candidate["energy_ev"]
        reference_forces = np.asarray(
            reference["forces_ev_per_angstrom"], dtype=np.float64)
        candidate_forces = np.asarray(
            candidate["forces_ev_per_angstrom"], dtype=np.float64)
        reference_stress = np.asarray(
            reference["ase_stress_ev_per_angstrom3"], dtype=np.float64)
        candidate_stress = np.asarray(
            candidate["ase_stress_ev_per_angstrom3"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise AssertionError("comparison records are incomplete or nonnumeric") from error
    if (isinstance(reference_energy, bool) or not np.isscalar(reference_energy)
            or not np.isfinite(reference_energy)
            or isinstance(candidate_energy, bool)
            or not np.isscalar(candidate_energy)
            or not np.isfinite(candidate_energy)):
        raise AssertionError("comparison energies must be finite scalars")
    if (reference_forces.ndim != 2 or reference_forces.shape[1:] != (3,)
            or not reference_forces.size
            or candidate_forces.shape != reference_forces.shape
            or not np.all(np.isfinite(reference_forces))
            or not np.all(np.isfinite(candidate_forces))):
        raise AssertionError("comparison forces must have matching finite rows")
    if (reference_stress.shape != (6,) or candidate_stress.shape != (6,)
            or not np.all(np.isfinite(reference_stress))
            or not np.all(np.isfinite(candidate_stress))):
        raise AssertionError("comparison stresses must contain six finite values")
    energy_error = abs(float(candidate_energy) - float(reference_energy))
    force_error = float(np.max(np.abs(candidate_forces - reference_forces)))
    stress_errors = np.abs(candidate_stress - reference_stress)
    return ase_validation.identical_frame_decision(
        energy_error, abs(float(reference_energy)),
        force_error, float(np.max(np.abs(reference_forces))),
        stress_errors, np.abs(reference_stress),
        ase_validation.IDENTICAL_DOUBLE_LIMITS)


def classify_replay(cpu_gpu: list[dict], stored_gpu: list[dict],
                    indices: tuple[int, ...]) -> str:
    """Classify replay evidence using the design's ordered decision rules."""
    if type(cpu_gpu) is not list or type(stored_gpu) is not list:
        raise AssertionError("replay decisions must be lists")
    if type(indices) is not tuple or not indices or indices[0] != 0:
        raise AssertionError("replay classification requires a step-0 anchor")
    _validate_indices(indices, indices[-1] + 1)
    if len(cpu_gpu) != len(indices) or len(stored_gpu) != len(indices):
        raise AssertionError("replay decision counts must match selected frames")
    decisions = cpu_gpu + stored_gpu
    if any(type(decision) is not dict or type(decision.get("pass")) is not bool
           for decision in decisions):
        raise AssertionError("replay decisions must contain boolean pass values")
    if not all(decision["pass"] for decision in cpu_gpu):
        return "general_gpu_backend_difference"
    if stored_gpu[0]["pass"] and any(
            not decision["pass"] for decision in stored_gpu[1:]):
        return "continuous_socket_state_suspected"
    if all(decision["pass"] for decision in stored_gpu):
        return "trajectory_geometry_explains_difference"
    return "inconclusive"
