# GPU PW Socket Replay Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and run a same-geometry fresh CPU/GPU PW/double replay that distinguishes a general GPU backend difference from state accumulated during consecutive variable-cell socket SCFs.

**Architecture:** Add one focused Python diagnostic beside the existing i-PI variable-cell validator. Its pure layer reconstructs selected frames and derives replayable decisions; its execution layer launches isolated FileIO SCFs while parsing socket-equivalent raw Kohn-Sham total energy from each fresh ABACUS log. A single serial V100 Slurm job produces the final auditable JSON.

**Tech Stack:** Python 3, NumPy, ASE, existing `socketio_variable_cell` helpers, `unittest`, ABACUS PW CPU/GPU executables, Slurm, official staged runtime assets.

## Global Constraints

- Work only on `feature/variable-cell-ipi-socket`; never modify `feature/fixed-cell-ipi-socket`.
- Keep `/home/gengjianrui/bin/abacus-develop` at fixed-cell HEAD `8b60f83c3e62af75ad57c4c6c61a52a8acdc4d60` and clean.
- Reuse the staged CPU executable SHA256 `2c645b927dc1ddf286352cc311dfe125ac0aef9c78026224508cc1b572842c86` and GPU executable SHA256 `5b518662652ba6ea3aaae9b75aa26c8aa804bfbd1d115ad00b8c730139552547`; do not rebuild ABACUS.
- Replay exactly steps `0,1,5,8,42,50` from Job 762695 unless an input validation failure proves the source artifact is inconsistent.
- Every replay frame uses one canonical ASE `Atoms` object for both devices and a fresh process/directory per device; never share charge density or wavefunctions.
- CPU and GPU INPUT files must be identical except for `device cpu` versus `device gpu`; both are PW/double with `ecutwfc 50`, `kspacing 0.45`, `scf_thr 1e-9`, `scf_nmax 100`, atomic charge initialization, forces, and stress; socket keywords must be absent.
- Compare energy using exactly one raw `#TOTAL ENERGY# ... eV` value from `running_scf.log`, not ASE FileIO's possible `E_KS(sigma->0)` value.
- Reuse `IDENTICAL_DOUBLE_LIMITS` and `identical_frame_decision`; do not add or relax scientific tolerances.
- Set `OMP_NUM_THREADS=1` for all real calculations.
- Use one serial single-GPU Slurm job on `4V100` with one task and one GPU; do not specify CPU or memory resources and do not enable MPS.
- Preserve finite built-in JSON types, exact executable/source provenance, raw logs, result SHA256, Slurm exit status, and no-residual-process evidence.
- Follow ABACUS governance: LF text, no new global dependencies, focused tests, and exact verification output.

---

### Task 1: Pure replay frame and decision layer

**Files:**
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py`
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py`

**Interfaces:**
- Consumes: `socketio_variable_cell.assert_valid_frame`, `IDENTICAL_DOUBLE_LIMITS`, `identical_frame_decision`, and `run_validation.assert_json_ready`.
- Produces:
  - `DEFAULT_FRAME_INDICES: tuple[int, ...] = (0, 1, 5, 8, 42, 50)`
  - `parse_frame_indices(value: str, frame_count: int) -> tuple[int, ...]`
  - `load_replay_frames(result_path: Path, positions_path: Path, indices: tuple[int, ...]) -> tuple[dict, list[dict]]`
  - `compare_records(reference: dict, candidate: dict) -> dict`
  - `classify_replay(cpu_gpu: list[dict], stored_gpu: list[dict], indices: tuple[int, ...]) -> str`

- [ ] **Step 1: Write failing frame-selection and reconstruction tests**

Add tests using temporary JSON and a two-atom six-frame extended XYZ fixture. The tests must include these assertions:

```python
class ReplayFrameTests(unittest.TestCase):
    def test_parse_frame_indices_requires_unique_sorted_in_range_values(self):
        self.assertEqual(replay.parse_frame_indices("0,1,5", 6), (0, 1, 5))
        for value in ("", "1,1", "2,1", "-1", "6"):
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
        self.assertEqual(payload["ipi_version"], "3.2.0")
```

The fixture JSON must identify `backend=pw`, `device=gpu`, `precision=double`, contain six finite `trajectory.steps`, and use positive-determinant cells. Add rejection tests for frame-count mismatch, non-finite cell, wrong backend/device/precision, non-3.2.0 i-PI, and atom-symbol mismatch.

- [ ] **Step 2: Run the focused tests and verify RED**

Run:

```bash
python3 -m unittest -v interfaces.ASE_interface.examples.ipi_variable_cell.test_diagnose_pw_replay.ReplayFrameTests
```

Expected: FAIL because `diagnose_pw_replay` and its functions do not exist.

- [ ] **Step 3: Implement minimal frame selection and reconstruction**

Use `ase.io.read(str(positions_path), index=":")`. Store each reconstructed item as:

```python
{
    "index": index,
    "atoms": atoms,
    "stored_gpu": stored_step,
    "xyz_cell_max_abs_delta_angstrom": float(
        np.max(np.abs(xyz_atoms.cell.array - json_cell)))
}
```

Reject non-built-in/non-finite JSON via `run_validation.assert_json_ready`. Always require the JSON and XYZ frame counts to match; additionally require exactly 51 frames when replaying the real Job 762695 source with `DEFAULT_FRAME_INDICES` (unit-test fixtures may use their smaller matching frame count). Call `assert_valid_frame(json_cell, atoms.positions)` after applying the JSON cell with `scale_atoms=False`.

- [ ] **Step 4: Write failing comparison and ordered-classification tests**

Create synthetic records with energy, forces, and six-component ASE stress. Cover all ordered outcomes:

```python
def test_classification_order(self):
    all_pass = [{"pass": True}] * 6
    later_fail = [{"pass": True}, {"pass": False}] + [{"pass": True}] * 4
    self.assertEqual(replay.classify_replay(
        [{"pass": False}] + all_pass[1:], all_pass, replay.DEFAULT_FRAME_INDICES),
        "general_gpu_backend_difference")
    self.assertEqual(replay.classify_replay(
        all_pass, later_fail, replay.DEFAULT_FRAME_INDICES),
        "continuous_socket_state_suspected")
    self.assertEqual(replay.classify_replay(
        all_pass, all_pass, replay.DEFAULT_FRAME_INDICES),
        "trajectory_geometry_explains_difference")
```

Also test that a stored step-0 failure yields `inconclusive`, and that wrong decision counts/types are rejected.

- [ ] **Step 5: Run the decision tests and verify RED**

Run:

```bash
python3 -m unittest -v interfaces.ASE_interface.examples.ipi_variable_cell.test_diagnose_pw_replay.ReplayDecisionTests
```

Expected: FAIL because `compare_records` and `classify_replay` do not exist.

- [ ] **Step 6: Implement comparison and classification**

`compare_records` must calculate absolute energy error, maximum absolute force error, six absolute stress errors, and their reference scales, then call:

```python
ase_validation.identical_frame_decision(
    energy_error, abs(reference["energy_ev"]),
    force_error, np.max(np.abs(reference_forces)),
    stress_errors, np.abs(reference_stress),
    ase_validation.IDENTICAL_DOUBLE_LIMITS)
```

`classify_replay` applies the exact ordered rules from the design and returns only one of the four specified strings.

- [ ] **Step 7: Run pure tests and quality checks**

Run:

```bash
python3 -m unittest -v interfaces.ASE_interface.examples.ipi_variable_cell.test_diagnose_pw_replay
python3 -m py_compile interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py
python3 -m flake8 --select=E9,F63,F7,F82 interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py
git diff --check
```

Expected: all tests PASS, py_compile and fatal flake8 exit 0, diff check clean.

- [ ] **Step 8: Commit Task 1**

```bash
git add interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py
git commit -m "test: add gpu pw replay decision layer"
```

### Task 2: Fresh SCF executor, provenance, and CLI

**Files:**
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py`
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py`
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/README.md`

**Interfaces:**
- Consumes: Task 1 `load_replay_frames`, `compare_records`, and `classify_replay`; existing `socketio_variable_cell.Config`, `_load_abacus_api`, `_profile`, `_common_kwargs`, `_identity`, `_frame_record`, and `_find_log`.
- Produces:
  - `parse_raw_total_energy(log_path: Path) -> float`
  - `normalize_paired_input(text: str) -> str`
  - `assert_paired_inputs(cpu_input: Path, gpu_input: Path) -> None`
  - `run_fresh_frame(config: Config, atoms, directory: Path) -> dict`
  - `run_diagnostic(args: argparse.Namespace) -> dict`
  - CLI options `--validation-json`, `--positions`, `--cpu-abacus`, `--gpu-abacus`, `--pp-orb-root`, `--frames`, `--workdir`, and `--output`.

- [ ] **Step 1: Write failing raw-energy and paired-INPUT tests**

Use exact representative log text:

```python
def test_parse_raw_total_energy_requires_exactly_one_value(self):
    log = self.write("running_scf.log", "#TOTAL ENERGY# -206.48515779761 eV\n")
    self.assertEqual(replay.parse_raw_total_energy(log), -206.48515779761)
    for text in ("", "#TOTAL ENERGY# -1 eV\n#TOTAL ENERGY# -2 eV\n",
                 "#TOTAL ENERGY# nan eV\n"):
        log.write_text(text)
        with self.assertRaises(AssertionError):
            replay.parse_raw_total_energy(log)

def test_paired_inputs_differ_only_by_device(self):
    replay.assert_paired_inputs(self.cpu_input, self.gpu_input)
    self.gpu_input.write_text(self.gpu_input.read_text() + "socket_driver 1\n")
    with self.assertRaises(AssertionError):
        replay.assert_paired_inputs(self.cpu_input, self.gpu_input)
```

Assert that `socket_driver` and `socket_variable_cell` are rejected in either file and all required PW/double fields have exact values.

- [ ] **Step 2: Run focused parser tests and verify RED**

Run:

```bash
python3 -m unittest -v interfaces.ASE_interface.examples.ipi_variable_cell.test_diagnose_pw_replay.ReplayExecutionTests
```

Expected: FAIL because execution helpers do not exist.

- [ ] **Step 3: Implement raw-energy parsing and fresh frame execution**

`parse_raw_total_energy` uses one finite numeric match for `#TOTAL ENERGY#`; it must not accept `!FINAL_ETOT_IS` or `E_KS(sigma->0)` as substitutes.

`run_fresh_frame` must:

```python
Abacus, _, _ = ase_validation._load_abacus_api()
fresh = atoms.copy()
fresh.calc = Abacus(profile=ase_validation._profile(config),
                    directory=directory,
                    **ase_validation._common_kwargs(config))
fileio_energy = float(fresh.get_potential_energy())
fresh.get_forces()
fresh.get_stress()
record = ase_validation._frame_record(
    config, ase_validation._identity(config), fresh, directory, "fileio")
record["fileio_energy_ev"] = fileio_energy
record["energy_ev"] = parse_raw_total_energy(
    ase_validation._find_log(directory))
record["energy_provenance"] = "running_scf.log #TOTAL ENERGY#"
```

Require exactly one SCF/stress frame, a banner matching the requested device, and a finite record. Never reuse a directory.

- [ ] **Step 4: Write failing orchestration tests with a fake fresh runner**

Inject `fresh_runner=run_fresh_frame` into `run_diagnostic` so tests can provide a deterministic fake without launching ABACUS. Assert:

- exactly twelve calls ordered frame-major then CPU/GPU;
- every CPU/GPU pair receives numerically identical cells and positions;
- output contains executable hashes/versions, source input hashes, frame paths, per-frame fresh CPU/GPU and stored-GPU decisions, and one ordered classification;
- `assert_json_ready` accepts the result;
- a fake unconverged/non-finite/missing result fails rather than writing output.

- [ ] **Step 5: Run orchestration tests and verify RED**

Run:

```bash
python3 -m unittest -v interfaces.ASE_interface.examples.ipi_variable_cell.test_diagnose_pw_replay.ReplayOrchestrationTests
```

Expected: FAIL because `run_diagnostic` and CLI parsing are incomplete.

- [ ] **Step 6: Implement orchestration and CLI**

Create immutable CPU/GPU `Config` objects with `basis="pw"`, `precision="double"`, device-specific executable and directory. Before calculations, hash all inputs and executables. For each selected frame, remove only its explicit not-yet-used case directory, launch fresh CPU then fresh GPU, audit paired INPUT, and build decisions.

The top-level JSON must contain:

```python
{
    "schema_version": 1,
    "kind": "gpu-pw-socket-replay-diagnostic",
    "source_validation": {...},
    "selected_frames": [0, 1, 5, 8, 42, 50],
    "cpu_identity": {...},
    "gpu_identity": {...},
    "frames": [...],
    "classification": classification,
    "thresholds": ase_validation.IDENTICAL_DOUBLE_LIMITS,
}
```

Write with `json.dumps(..., indent=2, allow_nan=False) + "\n"` only after every case succeeds.

- [ ] **Step 7: Document exact diagnostic use and scientific boundary**

Add a README section containing the full CLI shape, the raw-energy semantic distinction, the four classifications, and the statement that this diagnostic does not itself accept GPU PW or change production thresholds.

- [ ] **Step 8: Run the full focused verification**

Run:

```bash
python3 -m unittest -v interfaces.ASE_interface.examples.ipi_variable_cell.test_diagnose_pw_replay
python3 interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py --self-test
python3 -m py_compile interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py
python3 -m flake8 --select=E9,F63,F7,F82 interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py
python3 tools/03_code_analysis/agent_governance_check.py --staged
git diff --check
```

Expected: all unit/self-tests pass, all static checks exit 0, no new governance error.

- [ ] **Step 9: Commit Task 2**

```bash
git add interfaces/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/test_diagnose_pw_replay.py interfaces/ASE_interface/examples/ipi_variable_cell/README.md
git commit -m "test: add fresh gpu pw replay diagnostic"
```

### Task 3: Stage and run the real twelve-SCF V100 replay

**Files:**
- Runtime create: `/home/gengjianrui/bin/abacus-variable-cell-runtime/jobs/gpu-pw-replay.sbatch`
- Runtime stage: `/home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py`
- Runtime output: `/home/gengjianrui/bin/abacus-variable-cell-runtime/results/gpu-pw-replay-<jobid>.json`
- Report only; do not commit runtime work/log/result files.

**Interfaces:**
- Consumes: Task 2 CLI and Jobs 762695 source JSON/positions, staged executables, PP/ORB assets, module dependency chain.
- Produces: one result JSON with SHA256, twelve isolated raw case directories, Slurm/GPU identity, and an independently reviewed classification.

- [ ] **Step 1: Stage exact source and verify identities without rebuilding**

Copy only the changed Python/README assets into the existing runtime mirror, update `SOURCE_COMMIT` to the exact implementation HEAD, and verify:

```bash
sha256sum bin/abacus_basic_para bin/abacus_basic_gpu
bin/abacus_basic_para --version
bin/abacus_basic_gpu --version
python ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py --help
```

Expected executable hashes are the Global Constraints values; both versions are `v3.11.0-beta7` under the loaded module dependency chain.

- [ ] **Step 2: Create and review the single-GPU Slurm script**

The script must request `4V100`, one node/task/GPU, `rush-1o2gpu`, set `OMP_NUM_THREADS=1`, record `nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv`, load `abacus/develop-git-079fd0c-260724-sm70-auto`, and run:

```bash
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/examples/ipi_variable_cell/diagnose_pw_replay.py \
  --validation-json /home/gengjianrui/bin/abacus-variable-cell-runtime/results/gpu-pw-isotropic-762695.json \
  --positions /home/gengjianrui/bin/abacus-variable-cell-runtime/work/gpu-pw-isotropic-762695/trajectory/isotropic-abacus_vc_isotropic_1775430_1785917255506141486.positions_0.xyz \
  --cpu-abacus /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_para \
  --gpu-abacus /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_gpu \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --frames 0,1,5,8,42,50 \
  --workdir /home/gengjianrui/bin/abacus-variable-cell-runtime/work/gpu-pw-replay-${SLURM_JOB_ID} \
  --output /home/gengjianrui/bin/abacus-variable-cell-runtime/results/gpu-pw-replay-${SLURM_JOB_ID}.json
```

Run `bash -n` and obtain a fresh independent script review before submission.

- [ ] **Step 3: Submit exactly one job and monitor to terminal state**

Confirm the user queue is empty, submit once, and monitor with `squeue`, `scontrol`, and `sacct`. Do not submit another ABACUS job until this job and its review are complete.

- [ ] **Step 4: Audit result and raw evidence**

Require Slurm `COMPLETED 0:0`, one V100 `compute_cap=7.0`, twelve fresh directories, twelve unique converged SCFs, CPU/GPU INPUT equivalence except device, no socket keywords, finite JSON, exact selected frames, executable/source hashes, zero fatal stderr, and no residual process/socket.

Independently recompute per-frame raw-energy, force, stress, and cell residuals from logs/JSON. Record the result JSON SHA256 and the ordered classification. Do not reinterpret `E_KS(sigma->0)` as socket energy.

- [ ] **Step 5: Independent result review and stop condition**

Dispatch a fresh reviewer with the exact job/result/raw evidence. The reviewer returns Spec PASS/FAIL and Critical/Important/Minor findings.

If the diagnostic itself is valid, record its classification. Do not change production code or thresholds inside this plan. If the classification is `continuous_socket_state_suspected` or `general_gpu_backend_difference`, start a separate systematic-debugging design/plan only after this task review passes.

- [ ] **Step 6: Commit no runtime evidence and record verification**

Confirm both git worktrees are clean, append exact commands/results to the SDD report/ledger, and leave all runtime evidence outside git. The task is complete only after the independent result review has no open Critical/Important finding.
