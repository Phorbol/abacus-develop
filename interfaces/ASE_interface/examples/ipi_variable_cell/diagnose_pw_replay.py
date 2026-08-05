#!/usr/bin/env python3
"""Replay saved GPU PW socket frames through fresh ABACUS calculations."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shlex
import shutil
import sys
from pathlib import Path

import numpy as np
from ase.cell import Cell
from ase.geometry import cellpar_to_cell
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
IPI_CELL_MARKER = "CELL(abcABC):"
# i-PI writes abcABC values with five digits after the decimal point.  The
# half-quantization interval is 5.0e-6 in the native length (angstrom) and
# angle (degree) units.  Allow another 0.1e-6 for the upstream
# bohr-to-angstrom conversion and cellpar reconstruction; the immutable Job
# 762695 source reaches 5.058800658e-6 while a full last-digit disagreement is
# still rejected.
IPI_CELL_PARAMETER_TOLERANCE = 5.1e-6
_SOURCE_FRAME_IDENTITY_KEYS = (
    "executable_version", "executable_sha256", "source_commit", "module",
    "source", "backend", "device", "precision")


def _read_ipi_cells(positions_path: Path, frame_count: int) -> list[dict]:
    """Parse rounded i-PI cell, step, bead, and unit metadata."""
    lines = Path(positions_path).read_text().splitlines()
    if sum(line.count(IPI_CELL_MARKER) for line in lines) != frame_count:
        raise AssertionError("XYZ must contain one i-PI CELL header per frame")
    headers = [line for line in lines if IPI_CELL_MARKER in line]
    records = []
    for expected_step, header in enumerate(headers):
        if header.count(IPI_CELL_MARKER) != 1:
            raise AssertionError("XYZ frame must contain one i-PI CELL marker")
        values_text, step_marker, metadata = header.split(
            IPI_CELL_MARKER, 1)[1].partition("Step:")
        values = values_text.split()
        if not step_marker or len(values) != 6:
            raise AssertionError("i-PI CELL header must contain six values")
        metadata_fields = metadata.split()
        if (len(metadata_fields) != 5
                or metadata_fields[1] != "Bead:"
                or metadata_fields[3:] != [
                    "positions{angstrom}", "cell{angstrom}"]):
            raise AssertionError(
                "i-PI Step, Bead, and angstrom units are required")
        try:
            cellpar = np.asarray([float(value) for value in values],
                                 dtype=np.float64)
            step = int(metadata_fields[0])
            bead = int(metadata_fields[2])
        except ValueError as error:
            raise AssertionError(
                "i-PI CELL, Step, and Bead values must be numeric") from error
        if step != expected_step:
            raise AssertionError(
                "i-PI XYZ steps must be ordered exactly 0..{}".format(
                    frame_count - 1))
        if bead != 0:
            raise AssertionError("i-PI replay requires Bead 0")
        if (not np.all(np.isfinite(cellpar))
                or np.any(cellpar[:3] <= 0.0)
                or np.any(cellpar[3:] <= 0.0)
                or np.any(cellpar[3:] >= 180.0)):
            raise AssertionError("i-PI CELL lengths or angles are invalid")
        try:
            cell = cellpar_to_cell(cellpar).T
        except (AssertionError, ValueError) as error:
            raise AssertionError("i-PI CELL geometry is invalid") from error
        ase_validation.assert_valid_frame(cell, np.empty((0, 3)))
        records.append({
            "cell": cell,
            "cellpar": cellpar,
            "step": step,
            "bead": bead,
        })
    return records


def _validate_trajectory_metadata(
        trajectory: dict, frame_count: int, positions_path: Path) -> None:
    """Bind an official completed trajectory to its positions artifact."""
    expected_steps = frame_count - 1
    for key in ("requested_steps", "completed_steps", "sample_count"):
        if type(trajectory.get(key)) is not int:
            raise AssertionError("trajectory {} must be an integer".format(key))
    if (trajectory["requested_steps"] != expected_steps
            or trajectory["completed_steps"] != expected_steps
            or trajectory["sample_count"] != frame_count
            or trajectory.get("includes_initial_frame") is not True):
        raise AssertionError(
            "trajectory completion metadata disagrees with stored frames")
    mode = trajectory.get("mode")
    socket_name = trajectory.get("socket_name")
    if mode not in ("isotropic", "flexible"):
        raise AssertionError("trajectory mode is not an official i-PI mode")
    if (not isinstance(socket_name, str)
            or re.fullmatch(r"abacus_vc_[A-Za-z0-9_]+", socket_name) is None):
        raise AssertionError("trajectory socket name is invalid")
    expected_name = "{}-{}.positions_0.xyz".format(mode, socket_name)
    if Path(positions_path).name != expected_name:
        raise AssertionError(
            "positions filename disagrees with trajectory mode/socket name")


def _validate_stored_frame(
        stored_step: dict, xyz_atoms, xyz_metadata: dict,
        index: int, expected_identity) -> tuple[np.ndarray, dict]:
    """Validate one official stored frame and its matching XYZ record."""
    if not isinstance(stored_step, dict):
        raise AssertionError("stored replay frame must be an object")
    ase_validation.assert_real_frame_schema(stored_step)
    if (type(stored_step.get("ipi_property_step")) is not int
            or type(stored_step.get("is_initial_frame")) is not bool):
        raise AssertionError("stored i-PI step/initial metadata has wrong types")
    required_values = {
        "source": "official-ipi",
        "backend": "pw",
        "device": "gpu",
        "precision": "double",
        "ipi_property_step": index,
        "is_initial_frame": index == 0,
    }
    if any(stored_step.get(key) != value
           for key, value in required_values.items()):
        raise AssertionError("stored replay frame identity or sequence is wrong")
    if stored_step["atom_count"] != len(xyz_atoms):
        raise AssertionError("stored atom count disagrees with XYZ")
    identity = {
        key: stored_step.get(key) for key in _SOURCE_FRAME_IDENTITY_KEYS}
    if expected_identity is not None and identity != expected_identity:
        raise AssertionError("stored replay frame identities are inconsistent")
    try:
        json_cell = np.asarray(stored_step["cell_angstrom"],
                               dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise AssertionError("stored replay cell must be numeric") from error
    ase_validation.assert_valid_frame(json_cell, xyz_atoms.positions)
    json_cellpar = Cell(json_cell.T).cellpar()
    cellpar_error = np.abs(xyz_metadata["cellpar"] - json_cellpar)
    if np.any(cellpar_error[:3] > IPI_CELL_PARAMETER_TOLERANCE):
        raise AssertionError(
            "XYZ and JSON cell lengths exceed abcABC quantization")
    if np.any(cellpar_error[3:] > IPI_CELL_PARAMETER_TOLERANCE):
        raise AssertionError(
            "XYZ and JSON cell angles exceed abcABC quantization")
    return json_cell, identity


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
    _validate_trajectory_metadata(trajectory, len(stored_steps), positions_path)
    xyz_metadata = _read_ipi_cells(positions_path, len(xyz_frames))
    _validate_indices(indices, len(stored_steps))
    if indices == DEFAULT_FRAME_INDICES and len(stored_steps) != 51:
        raise AssertionError("the default Job 762695 replay requires 51 frames")
    symbols = xyz_frames[0].get_chemical_symbols()
    if any(frame.get_chemical_symbols() != symbols for frame in xyz_frames):
        raise AssertionError("XYZ atom symbols differ between frames")
    if not symbols or any(symbol != "Si" for symbol in symbols):
        raise AssertionError("the fixed Si replay does not support other symbols")

    validated_frames = []
    expected_identity = None
    for index, (stored_step, xyz_atoms, xyz_record) in enumerate(zip(
            stored_steps, xyz_frames, xyz_metadata)):
        json_cell, identity = _validate_stored_frame(
            stored_step, xyz_atoms, xyz_record, index, expected_identity)
        if expected_identity is None:
            expected_identity = identity
        atoms = xyz_atoms.copy()
        atoms.set_cell(json_cell, scale_atoms=False)
        atoms.set_pbc((True, True, True))
        validated_frames.append({
            "index": index,
            "atoms": atoms,
            "stored_gpu": stored_step,
            "xyz_cell_max_abs_delta_angstrom": float(
                np.max(np.abs(xyz_record["cell"] - json_cell))),
        })

    frames = [validated_frames[index] for index in indices]
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

_RAW_TOTAL_ENERGY = re.compile(
    r"^\s*#TOTAL ENERGY#\s+"
    r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[Ee][-+]?\d+)?)\s+eV\s*$",
    flags=re.MULTILINE)
_REQUIRED_INPUT_VALUES = {
    "calculation": "scf",
    "basis_type": "pw",
    "precision": "double",
    "chg_extrap": "atomic",
}
_REQUIRED_INPUT_NUMBERS = {
    "ecutwfc": 50.0,
    "kspacing": 0.45,
    "scf_thr": 1.0e-9,
    "scf_nmax": 100.0,
    "cal_force": 1.0,
    "cal_stress": 1.0,
}
_SOCKET_INPUT_KEYWORDS = ("socket_driver", "socket_variable_cell")


def parse_raw_total_energy(log_path: Path) -> float:
    """Return the unique finite socket-equivalent raw ABACUS energy."""
    path = Path(log_path)
    if not path.is_file():
        raise AssertionError("ABACUS running_scf.log is absent")
    matches = _RAW_TOTAL_ENERGY.findall(path.read_text(errors="replace"))
    if len(matches) != 1:
        raise AssertionError(
            "running_scf.log must contain exactly one raw total energy")
    energy = float(matches[0])
    if not np.isfinite(energy):
        raise AssertionError("raw total energy must be finite")
    return energy


def _paired_input_fields(text: str) -> dict[str, str]:
    if not isinstance(text, str):
        raise AssertionError("ABACUS INPUT must be text")
    if any(re.search(r"\b{}\b".format(keyword), text, re.IGNORECASE)
           for keyword in _SOCKET_INPUT_KEYWORDS):
        raise AssertionError("fresh replay INPUT must not contain socket keywords")
    fields: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if (not stripped or stripped.upper() == "INPUT_PARAMETERS"
                or stripped.startswith("#")):
            continue
        tokens = stripped.split()
        if len(tokens) < 2:
            continue
        key = tokens[0].lower()
        value = " ".join(tokens[1:])
        if key in fields:
            raise AssertionError("ABACUS INPUT repeats {}".format(key))
        fields[key] = value
    for key, expected in _REQUIRED_INPUT_VALUES.items():
        if fields.get(key, "").lower() != expected:
            raise AssertionError(
                "fresh replay INPUT requires {} {}".format(key, expected))
    for key, expected in _REQUIRED_INPUT_NUMBERS.items():
        try:
            actual = float(fields[key])
        except (KeyError, ValueError) as error:
            raise AssertionError(
                "fresh replay INPUT requires numeric {}".format(key)) from error
        if not np.isfinite(actual) or actual != expected:
            raise AssertionError(
                "fresh replay INPUT has wrong {}".format(key))
    device = fields.get("device", "").lower()
    if device not in ("cpu", "gpu"):
        raise AssertionError("fresh replay INPUT requires cpu or gpu device")
    return fields


def normalize_paired_input(text: str) -> str:
    """Validate a fresh PW/double INPUT and mask its one device value."""
    fields = _paired_input_fields(text)
    device_pattern = re.compile(
        r"^(\s*device\s+){}(\s*)$".format(re.escape(fields["device"])),
        flags=re.MULTILINE | re.IGNORECASE)
    normalized, count = device_pattern.subn(r"\1<device>\2", text)
    if count != 1:
        raise AssertionError("fresh replay INPUT must declare device once")
    return normalized


def assert_paired_inputs(cpu_input: Path, gpu_input: Path) -> None:
    """Require fresh CPU/GPU INPUT files to differ only by device."""
    cpu_text = Path(cpu_input).read_text()
    gpu_text = Path(gpu_input).read_text()
    if _paired_input_fields(cpu_text)["device"].lower() != "cpu":
        raise AssertionError("CPU replay INPUT does not request device cpu")
    if _paired_input_fields(gpu_text)["device"].lower() != "gpu":
        raise AssertionError("GPU replay INPUT does not request device gpu")
    if normalize_paired_input(cpu_text) != normalize_paired_input(gpu_text):
        raise AssertionError("CPU/GPU replay INPUT files differ beyond device")


def _assert_requested_device(log_path: Path, device: str) -> None:
    text = Path(log_path).read_text(errors="replace")
    banner = re.compile(
        r"^\s*(?:RUNNING WITH DEVICE\s*:\s*)?{}\s*/".format(
            re.escape(device)),
        flags=re.MULTILINE | re.IGNORECASE)
    if banner.search(text) is None:
        raise AssertionError(
            "ABACUS log lacks requested {} device banner".format(device))


def run_fresh_frame(config: ase_validation.Config, atoms,
                    directory: Path) -> dict:
    """Run one isolated non-socket FileIO SCF and record raw replay energy."""
    directory = Path(directory)
    if directory.exists():
        raise AssertionError("fresh replay directory already exists")
    if (config.basis != "pw" or config.precision != "double"
            or config.device not in ("cpu", "gpu")):
        raise AssertionError("fresh replay requires CPU/GPU PW/double config")
    ase_validation.assert_valid_frame(
        np.asarray(atoms.cell, dtype=np.float64),
        np.asarray(atoms.positions, dtype=np.float64))
    Abacus, _, _ = ase_validation._load_abacus_api()
    fresh = atoms.copy()
    fresh.calc = Abacus(
        profile=ase_validation._profile(config), directory=directory,
        **ase_validation._common_kwargs(config))
    fileio_energy = float(fresh.get_potential_energy())
    fresh.get_forces()
    fresh.get_stress()
    logs = sorted(directory.glob("OUT.*/running_scf.log"))
    if len(logs) != 1:
        raise AssertionError(
            "fresh replay must produce exactly one running_scf.log")
    log_path = logs[0]
    ase_validation.raw_frame_series(directory, expected_frames=1)
    _assert_requested_device(log_path, config.device)
    record = ase_validation._frame_record(
        config, ase_validation._identity(config), fresh, directory, "fileio")
    record["fileio_energy_ev"] = fileio_energy
    record["energy_ev"] = parse_raw_total_energy(log_path)
    record["energy_provenance"] = "running_scf.log #TOTAL ENERGY#"
    if record.get("scf_converged") is not True:
        raise AssertionError("fresh replay SCF did not converge")
    run_validation.assert_json_ready(record)
    return record


_SOURCE_IDENTITY_KEYS = (
    "executable_version", "executable_sha256", "source_commit", "module")


def _sha256_file(path: Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise AssertionError("required input file is absent: {}".format(path))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _executable_path(command: str) -> Path:
    try:
        tokens = shlex.split(command)
    except ValueError as error:
        raise AssertionError("ABACUS command is invalid") from error
    for token in reversed(tokens):
        candidate = shutil.which(token)
        if candidate is not None and "abacus" in Path(candidate).name.lower():
            path = Path(candidate).resolve()
            if not path.is_file():
                break
            return path
    raise AssertionError("cannot resolve ABACUS executable path")


def _diagnostic_identity(config: ase_validation.Config) -> dict:
    identity = copy.deepcopy(ase_validation._identity(config))
    executable = _executable_path(config.abacus)
    digest = _sha256_file(executable)
    if identity.get("executable_sha256") != digest:
        raise AssertionError("ABACUS executable identity hash is inconsistent")
    identity.update({
        "command": config.abacus,
        "executable_path": str(executable),
    })
    run_validation.assert_json_ready(identity)
    return identity


def _source_identity(frames: list[dict]) -> dict:
    if not frames:
        raise AssertionError("source replay frames are absent")
    identity = {
        key: frames[0]["stored_gpu"].get(key)
        for key in _SOURCE_IDENTITY_KEYS
    }
    if (not all(isinstance(identity[key], str)
                for key in _SOURCE_IDENTITY_KEYS)
            or not identity["executable_version"]
            or re.fullmatch(r"[0-9a-fA-F]{64}",
                            identity["executable_sha256"]) is None
            or re.fullmatch(r"[0-9a-fA-F]{40}",
                            identity["source_commit"]) is None):
        raise AssertionError("source validation identity is incomplete")
    for frame in frames[1:]:
        if any(frame["stored_gpu"].get(key) != identity[key]
               for key in _SOURCE_IDENTITY_KEYS):
            raise AssertionError("source frame identities are inconsistent")
    identity["executable_sha256"] = identity["executable_sha256"].lower()
    identity["source_commit"] = identity["source_commit"].lower()
    return identity


def _validate_fresh_record(
        record: dict, config: ase_validation.Config,
        expected_identity: dict) -> None:
    run_validation.assert_json_ready(record)
    required = (
        "source", "backend", "device", "precision", "scf_converged",
        "energy_ev", "fileio_energy_ev", "energy_provenance",
        "cell_angstrom", "forces_ev_per_angstrom",
        "ase_stress_ev_per_angstrom3") + _SOURCE_IDENTITY_KEYS
    if not isinstance(record, dict) or any(
            key not in record for key in required):
        raise AssertionError("fresh replay record is incomplete")
    if (record["source"] != "fileio"
            or record["backend"] != "pw"
            or record["device"] != config.device
            or record["precision"] != "double"
            or record["scf_converged"] is not True
            or record["energy_provenance"]
            != "running_scf.log #TOTAL ENERGY#"
            or any(record[key] != expected_identity.get(key)
                   for key in _SOURCE_IDENTITY_KEYS)):
        raise AssertionError("fresh replay record identity or convergence is invalid")
    fileio_energy = record["fileio_energy_ev"]
    if (isinstance(fileio_energy, bool)
            or not np.isscalar(fileio_energy)
            or not np.isfinite(fileio_energy)):
        raise AssertionError("fresh FileIO energy must be a finite scalar")
    cell = np.asarray(record["cell_angstrom"], dtype=np.float64)
    forces = np.asarray(record["forces_ev_per_angstrom"], dtype=np.float64)
    ase_validation.assert_valid_frame(cell, np.empty((0, 3)))
    if forces.ndim != 2 or forces.shape[1:] != (3,) or not forces.size:
        raise AssertionError("fresh replay forces are invalid")
    compare_records(record, record)


def _reset_case_directory(
        workdir: Path, case: Path, used: set[Path]) -> None:
    root = workdir.resolve()
    resolved = case.resolve()
    if resolved.parent != root or resolved in used:
        raise AssertionError("fresh replay case path is unsafe or reused")
    if case.is_symlink():
        raise AssertionError("fresh replay case path must not be a symlink")
    if case.exists():
        shutil.rmtree(case)
    used.add(resolved)


def _precompute_case_paths(
        workdir: Path, indices: tuple[int, ...], output: Path,
        protected_paths: tuple[Path, ...]) -> dict[tuple[int, str], Path]:
    root = workdir.resolve()
    if output == root or root in output.parents:
        raise AssertionError("diagnostic output must be outside workdir")

    cases: dict[tuple[int, str], Path] = {}
    for index in indices:
        for device in ("cpu", "gpu"):
            case = root / "frame-{:03d}-{}".format(index, device)
            if case.is_symlink():
                raise AssertionError("fresh replay case path must not be a symlink")
            resolved = case.resolve()
            if resolved.parent != root or resolved in cases.values():
                raise AssertionError("fresh replay case path is unsafe or reused")
            cases[(index, device)] = resolved

    for protected in protected_paths:
        resolved_protected = protected.resolve()
        if any(resolved_protected == case
               or case in resolved_protected.parents
               for case in cases.values()):
            raise AssertionError("protected input overlaps fresh replay case")
    return cases


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay stored GPU PW socket frames as fresh CPU/GPU SCFs")
    parser.add_argument("--validation-json", type=Path, required=True)
    parser.add_argument("--positions", type=Path, required=True)
    parser.add_argument("--cpu-abacus", type=Path, required=True)
    parser.add_argument("--gpu-abacus", type=Path, required=True)
    parser.add_argument("--pp-orb-root", type=Path, required=True)
    parser.add_argument(
        "--frames",
        default=",".join(str(index) for index in DEFAULT_FRAME_INDICES))
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run_diagnostic(
        args: argparse.Namespace, fresh_runner=run_fresh_frame) -> dict:
    """Run all selected fresh replays and write one transactional manifest."""
    validation_json = Path(args.validation_json).resolve()
    positions = Path(args.positions).resolve()
    cpu_abacus = Path(args.cpu_abacus).resolve()
    gpu_abacus = Path(args.gpu_abacus).resolve()
    pp_orb_root = Path(args.pp_orb_root).resolve()
    workdir = Path(args.workdir).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise AssertionError("refusing to overwrite diagnostic output")
    source_hash = _sha256_file(validation_json)
    positions_hash = _sha256_file(positions)
    pseudopotential = (pp_orb_root / "Si_ONCV_PBE-1.2.upf").resolve()
    pseudopotential_hash = _sha256_file(pseudopotential)

    source_preview = json.loads(validation_json.read_text())
    run_validation.assert_json_ready(source_preview)
    try:
        frame_count = len(source_preview["trajectory"]["steps"])
    except (KeyError, TypeError) as error:
        raise AssertionError("source validation trajectory is incomplete") from error
    indices = parse_frame_indices(args.frames, frame_count)
    payload, replay_frames = load_replay_frames(
        validation_json, positions, indices)
    case_paths = _precompute_case_paths(
        workdir, indices, output,
        (validation_json, positions, cpu_abacus, gpu_abacus,
         pseudopotential))

    cpu_config = ase_validation.Config(
        str(cpu_abacus), "pw", "cpu", "double", workdir / "cpu",
        output, pp_orb_root)
    gpu_config = ase_validation.Config(
        str(gpu_abacus), "pw", "gpu", "double", workdir / "gpu",
        output, pp_orb_root)
    cpu_identity = _diagnostic_identity(cpu_config)
    gpu_identity = _diagnostic_identity(gpu_config)
    source_identity = _source_identity(replay_frames)

    workdir.mkdir(parents=True, exist_ok=True)
    used: set[Path] = set()
    frame_records = []
    cpu_gpu_decisions = []
    stored_gpu_decisions = []
    for replay_frame in replay_frames:
        index = replay_frame["index"]
        atoms = replay_frame["atoms"]
        cpu_case = case_paths[(index, "cpu")]
        gpu_case = case_paths[(index, "gpu")]
        _reset_case_directory(workdir, cpu_case, used)
        fresh_cpu = fresh_runner(cpu_config, atoms, cpu_case)
        _validate_fresh_record(fresh_cpu, cpu_config, cpu_identity)
        if not (cpu_case / "INPUT").is_file():
            raise AssertionError("fresh CPU replay INPUT is absent")

        _reset_case_directory(workdir, gpu_case, used)
        fresh_gpu = fresh_runner(gpu_config, atoms, gpu_case)
        _validate_fresh_record(fresh_gpu, gpu_config, gpu_identity)
        if not (gpu_case / "INPUT").is_file():
            raise AssertionError("fresh GPU replay INPUT is absent")
        assert_paired_inputs(cpu_case / "INPUT", gpu_case / "INPUT")

        cpu_gpu = compare_records(fresh_cpu, fresh_gpu)
        stored_gpu = compare_records(fresh_gpu, replay_frame["stored_gpu"])
        cpu_gpu_decisions.append(cpu_gpu)
        stored_gpu_decisions.append(stored_gpu)
        frame_records.append({
            "index": index,
            "source_positions_path": str(positions),
            "source_validation_path": str(validation_json),
            "cpu_case_path": str(cpu_case),
            "gpu_case_path": str(gpu_case),
            "xyz_cell_max_abs_delta_angstrom":
                replay_frame["xyz_cell_max_abs_delta_angstrom"],
            "fresh_cpu": fresh_cpu,
            "fresh_gpu": fresh_gpu,
            "stored_gpu": replay_frame["stored_gpu"],
            "fresh_cpu_gpu_decision": cpu_gpu,
            "stored_gpu_fresh_gpu_decision": stored_gpu,
        })

    classification = classify_replay(
        cpu_gpu_decisions, stored_gpu_decisions, indices)
    result = {
        "schema_version": 1,
        "kind": "gpu-pw-socket-replay-diagnostic",
        "source_validation": {
            "validation_json_path": str(validation_json),
            "validation_json_sha256": source_hash,
            "positions_path": str(positions),
            "positions_sha256": positions_hash,
            "ipi_version": payload["ipi_version"],
            "backend": payload["backend"],
            "device": payload["device"],
            "precision": payload["precision"],
            "source_identity": source_identity,
        },
        "selected_frames": list(indices),
        "cpu_identity": cpu_identity,
        "gpu_identity": gpu_identity,
        "pseudopotential": {
            "path": str(pseudopotential.resolve()),
            "sha256": pseudopotential_hash,
        },
        "frames": frame_records,
        "classification": classification,
        "thresholds": copy.deepcopy(
            ase_validation.IDENTICAL_DOUBLE_LIMITS),
    }
    run_validation.assert_json_ready(result)
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text)
    return result


def main(argv=None) -> int:
    run_diagnostic(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
