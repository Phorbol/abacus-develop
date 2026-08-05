# GPU PW Socket Replay Diagnostic Design

## Purpose

Determine whether the reproducible energy-conservation drift in the official
i-PI 3.2 GPU/PW/double isotropic trajectory is caused by the general ABACUS GPU
PW backend at those geometries, or by state retained across consecutive socket
SCFs while the cell changes.

The motivating runs are Jobs 762630 and 762695. They ran on two different V100
GPUs and reproduced low/high/trajectory conserved-energy drifts of approximately
0.00808, 0.00883, and 0.01164 eV. The CPU/PW/double trajectory drift was about
3.18e-7 eV. Protocol parsing, binary64 transport, GPU execution, stress/virial
conversion, and ordinary SCF non-convergence have already been excluded as the
direct cause.

## Scope

Add one focused diagnostic driver under
`interfaces/ASE_interface/examples/ipi_variable_cell/` and focused unit tests.
The driver will replay selected saved trajectory geometries as fresh,
non-socket ABACUS SCFs with both the CPU and GPU PW/double executables.

The diagnostic will not change the socket protocol, production ABACUS code,
i-PI XML, scientific thresholds, INPUT defaults, or the fixed-cell branch. It
will not declare GPU/PW acceptance; it only produces evidence for the next root
cause decision.

## Inputs and Provenance

The command-line driver will require:

- the source validation JSON;
- its matching i-PI positions trajectory;
- the CPU and GPU ABACUS executable paths;
- the PP/ORB root;
- a work directory and output JSON path.

The default diagnostic frame set is steps `0,1,5,8,42,50`, covering the common
initial frame, first visible energy growth, five-step probe scale, first
millielectronvolt-scale divergence, maximum observed drift, and final frame.
The frame list remains an explicit CLI value in the recorded manifest.

The implementation will record input-file SHA256 values, executable SHA256
values and versions, source marker, selected frame indices, module environment,
and every generated case path. All emitted numerical values must be finite
built-in JSON types.

## Data Flow

1. Load and validate the official i-PI result JSON and ASE extended-XYZ
   trajectory. Require matching frame counts and selected indices.
2. Build one canonical ASE `Atoms` object per selected frame using positions
   from the trajectory and the higher-precision cell stored in the JSON.
3. Validate positive determinant, finite coordinates, and atom identity. Record
   the maximum discrepancy between the XYZ cell and JSON cell. Use the same
   canonical `Atoms` object for the CPU and GPU cases.
4. For each frame, create two fresh directories and two fresh ABACUS processes.
   Reuse `socketio_variable_cell.Config`, `_common_kwargs`, `_profile`, and the
   existing ABACUS/ASE writer rather than manually formatting STRU or INPUT.
5. Keep both cases identical except for executable and `device cpu` versus
   `device gpu`. Both use PW, double precision, `ecutwfc 50`, `kspacing 0.45`,
   `scf_thr 1e-9`, `scf_nmax 100`, atomic charge initialization, forces, and
   stress. Socket keywords must be absent.
6. Parse fresh energy, force, and stress through the existing FileIO/raw-log
   validation path. Confirm one converged SCF and the declared device for every
   case.
7. Compare fresh CPU versus fresh GPU with the existing identical-frame
   thresholds. Separately compare each fresh result with the stored socket frame
   and record absolute residuals without using the broad conserved-energy
   tolerance.

Each replay starts a new ABACUS process. No charge density, wavefunction, or
working directory may be shared between frames or devices.

## Decision Output

The result JSON will report measurements and one ordered diagnostic
classification. Every relationship below uses the repository's existing
PW/double identical-frame energy, force, and stress thresholds; the diagnostic
does not invent another tolerance:

- `general_gpu_backend_difference`: fresh CPU and GPU fail the identical-frame
  decision at any selected frame;
- `continuous_socket_state_suspected`: fresh CPU and GPU pass at every frame,
  the saved GPU socket step 0 agrees with its fresh replay, and at least one
  later saved GPU socket frame fails against its fresh GPU replay;
- `trajectory_geometry_explains_difference`: fresh CPU and GPU pass and every
  saved GPU socket frame passes against its fresh GPU replay;
- `inconclusive`: none of the above relationships is supported.

The ordered classification does not claim that a general backend difference
excludes an additional socket-state effect. The JSON retains every per-frame
residual so that result remains reviewable. An `inconclusive` result is valid
diagnostic output, not a reason to fabricate a pass.

## Error Handling

Fail before launching ABACUS for missing/mismatched files, non-finite values,
frame-count mismatch, invalid cells, unsafe frame indices, or CPU/GPU INPUT
differences beyond the device field. Fail the job for a nonzero process exit,
missing log, wrong device identity, unconverged SCF, absent energy/force/stress,
or non-finite output. Preserve completed case directories and logs for audit.

## Verification

Focused tests will cover frame selection, cell/position reconstruction,
CPU/GPU INPUT equivalence except `device`, finite JSON serialization, decision
classification, and rejection of mismatched or non-finite inputs. Tests use
synthetic fixtures and do not launch ABACUS.

One serial Slurm job will then run all twelve fresh SCFs (six frames times two
devices) on one V100 allocation, with `OMP_NUM_THREADS=1`. The job will use the
already staged CPU and GPU executables and the existing module dependency chain;
it will not rebuild ABACUS. Exact commands, hashes, Slurm status, raw logs, and
the result JSON SHA256 will be recorded before any scientific conclusion.
