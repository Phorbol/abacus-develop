# Variable-Cell i-PI Socket Calculator Design

Date: 2026-08-03

Status: approved in design discussion; implementation not started

Target branch: `feature/variable-cell-ipi-socket`

Base commit: `8b60f83c3e62af75ad57c4c6c61a52a8acdc4d60`

## Context

ABACUS currently has a working fixed-cell i-PI socket client. ASE or i-PI sends
positions and a cell through `POSDATA`; ABACUS runs an SCF calculation and
returns energy and forces through `FORCEREADY`. The current implementation
rejects any cell change and returns a zero virial.

This feature adds an explicitly enabled variable-cell path for PW and LCAO.
It must support volume changes, anisotropic strain, and shear, and must return
a physically correct virial to both ASE and the official i-PI server. The
existing fixed-cell feature branch must not be modified.

The implementation will first target the current develop line. Only after the
develop implementation and validation are complete will a separate LTS 3.10.x
backport branch be created.

## Goals

- Accept a complete finite, right-handed, nonsingular 3x3 cell from `POSDATA`.
- Rebuild all cell-dependent PW and LCAO state before every calculation whose
  cell changed.
- Return energy, forces, and virial in the exact i-PI wire units and layout.
- Support ASE `UnitCellFilter` and `FrechetCellFilter`.
- Support official i-PI isotropic and flexible NPT barostats.
- Validate CPU and single-V100 PW and LCAO paths, including stress signs,
  units, cell layout, precision boundaries, and short-time stability.
- Preserve fixed-cell behavior unless variable-cell support is explicitly
  enabled.
- Keep the implementation compatible with C++11 and the ABACUS governance
  requirements.

## Non-goals

- Changing the i-PI protocol or adding cross-endian serialization.
- Supporting batched or shared-memory variants of the newer i-PI protocol.
- Treating the nine cell matrix entries as nine independent physical degrees
  of freedom. A periodic cell has six strain degrees of freedom after rigid
  rotations are removed.
- Claiming thermodynamic convergence from the short NPT validation runs.
- Mixing LTS compatibility code into the develop feature branch.
- Adding a new compiler, container, or complete ABACUS installation.

## User Interface

### ASE

The default remains fixed-cell:

```python
calc = AbacusSocketIO(...)
```

Variable-cell operation is an explicit opt-in:

```python
calc = AbacusSocketIO(..., variable_cell=True)
```

When `variable_cell=True`, the generated ABACUS input will contain:

```text
calculation          scf
socket_driver        1
socket_variable_cell 1
cal_force            1
cal_stress           1
```

The constructor option is authoritative for `AbacusSocketIO`. If an explicit
`socket_variable_cell` value in the user input conflicts with the constructor
option, the Python interface raises `ValueError` instead of silently choosing
one value.

### Direct official i-PI use

Users who launch the official i-PI server directly enable the feature in
ABACUS INPUT with:

```text
calculation          scf
socket_driver        1
socket_variable_cell 1
```

The ABACUS input layer automatically enables force and stress calculation for
this mode.

### Input validation

`socket_variable_cell=1` requires:

- `socket_driver=1`;
- `calculation=scf`;
- an initially valid 3D periodic cell;
- zero `press1`, `press2`, and `press3`.

The external pressure is owned by ASE's filter or the i-PI barostat. Applying
ABACUS `press1/press2/press3` at the same time would double count an external
stress. The first implementation declares support only for the validated
Kohn-Sham PW and LCAO paths.

The input addition must update both `docs/parameters.yaml` and
`docs/advanced/input_files/input-main.md`.

## Wire Contract

The base i-PI protocol is retained exactly.

| Field | Wire representation | Wire unit |
|---|---|---|
| Header | 12 one-byte ASCII characters, space padded | none |
| Atom count and string length | 4-byte `int32` | count |
| Cell | 9 IEEE-754 binary64 values | Bohr |
| Inverse cell | 9 IEEE-754 binary64 values | Bohr^-1 |
| Positions | `3 * nat` IEEE-754 binary64 values | Bohr |
| Energy | IEEE-754 binary64 | Hartree |
| Forces | `3 * nat` IEEE-754 binary64 values | Hartree/Bohr |
| Virial | 9 IEEE-754 binary64 values | Hartree |

The C++ socket boundary uses `std::int32_t`, not an assumed native `int`, and
has compile-time checks for a four-byte integer and an eight-byte IEEE-754
`double`. Counts are validated before multiplication or allocation, including
overflow checks for `3 * nat` and byte counts.

The established i-PI protocol uses native endian values. This implementation
therefore supports client and server processes with the same endian order and
documents that restriction. It does not add an incompatible byte swap.

All socket floating-point values are binary64. A GPU SCF may internally use
single or mixed precision, but the result is converted numerically to `double`
before it reaches the socket buffer. A four-byte GPU value must never be
reinterpreted as an eight-byte socket value.

References:

- <https://docs.ipi-code.org/distributed.html>
- <https://raw.githubusercontent.com/i-pi/i-pi/main/ipi/interfaces/sockets.py>
- <https://gitlab.com/ase/ase/-/raw/master/ase/calculators/socketio.py>

## Cell Layout and Mapping

ASE stores the three lattice vectors as rows. ASE's socket implementation sends
`cell.T`, while i-PI stores lattice vectors as columns. ABACUS stores lattice
vectors as rows in `UnitCell::latvec`.

Named conversion helpers will implement the mapping:

```text
H_wire       = transpose(A_ASE)
A_ABACUS     = transpose(H_wire)
Hinv_wire    = inverse(H_wire)
```

No call site should reproduce this conversion with incidental element indices.
The same rule applies to physical tensor serialization: the 3x3 virial is
transposed into the i-PI wire order exactly as ASE's `sendforce()` expects.

The supplied inverse matrix is checked for finiteness and consistency but is
not trusted as ABACUS internal state. ABACUS recomputes reciprocal-cell data
from the validated cell.

The socket path accepts a general right-handed nonsingular 3x3 cell. Official
i-PI removes rigid cell rotations and represents its evolving cell as an upper
triangular matrix. This still retains the six independent strain degrees of
freedom, including all three shears.

## Stress, Virial, Sign, and Units

Define:

- `sigma_ab`: ABACUS `cal_stress()` output in Ry/Bohr^3;
- `sigma_ase`: ASE stress convention in Hartree/Bohr^3;
- `W_ipi`: i-PI virial in Hartree;
- `V`: cell volume in Bohr^3.

The existing ABACUS ASE file parser negates ABACUS's printed stress. ASE's
socket calculator defines stress from the received virial as:

```text
sigma_ase = -W_ipi / V
```

The candidate conversion is therefore:

```text
sigma_ase = -0.5 * sigma_ab
W_ipi     =  0.5 * V * sigma_ab
```

The factor `0.5` is the exact Ry-to-Hartree conversion. This path performs no
intermediate conversion through eV, Angstrom, GPa, or kbar.

The formula is a contract to verify, not an assumption accepted from names in
the source. Existing ABACUS MD code contains variables named `virial` that are
still stress-like quantities with dimensions of energy per volume. The socket
implementation must use dimension-bearing names such as
`stress_ry_per_bohr3` and `virial_hartree`.

The sign and shear factors must be independently established by:

1. comparing socket stress with normal FileIO stress for the identical frame;
2. central finite differences in all six independent strain directions;
3. checking the direction of hydrostatic filter and barostat response.

For a diagonal strain:

```text
dE / d epsilon_ii = -W_ii
```

For symmetric shear
`F = I + gamma/2 * (e_i e_j^T + e_j e_i^T)`:

```text
dE / d gamma = -W_ij
```

Using symmetric shear avoids an accidental factor of two from engineering
shear notation.

ABACUS stress should be symmetric. A large antisymmetric component is treated
as a calculation error. A component below a documented absolute-plus-relative
noise threshold is checked and then removed with `(sigma + sigma.T) / 2`
before serialization.

## Frame Validation

A `POSDATA` frame is represented as plain temporary data and validated before
any `UnitCell` or ESolver state is changed.

Validation includes:

- exact field lengths;
- positive atom count equal to the STRU atom count;
- checked size arithmetic;
- finite cell, inverse-cell, and position values;
- `det(H) > 0`, rejecting left-handed, zero-volume, and singular cells;
- a scaled nonsingularity check, initially requiring a 2-norm condition number
  below `1e12`;
- a dimensionless `H * Hinv - I` residual checked with absolute and relative
  tolerance;
- finite energy, force, stress, and virial results;
- solver convergence through a narrow explicit solver-status interface or the
  existing solver failure mechanism, without introducing a new global control
  dependency.

There is no arbitrary physical maximum cell-step limit. Numerically valid large
steps are logged, while physical runaway is detected by the integration tests.

The first-frame atom-order warning is retained. The i-PI protocol carries no
chemical species, so the ASE wrapper must keep its current species grouping and
force reverse mapping.

## Atomic Cell Update

Each accepted frame uses a two-phase process.

### Phase 1: prepare

1. The root rank reads the complete frame into temporary buffers.
2. It validates types, sizes, finiteness, cell geometry, and inverse consistency.
3. It broadcasts a single validation result and the validated frame to all MPI
   ranks.
4. No persistent ABACUS state has changed at this point.

### Phase 2: commit

1. Keep `lat0` fixed and set `latvec = A_ABACUS / lat0`.
2. Convert the received absolute Cartesian positions to direct coordinates
   using the new validated cell.
3. Assign the new direct coordinates.
4. Call `unitcell::setup_cell_after_vc()` so volume, reciprocal vectors,
   Cartesian positions, and other UnitCell-derived quantities are consistent.
5. Set both `cell_parameter_updated` and `ionic_position_updated` when the cell
   changed; use the positions-only path when it did not.
6. Run the ESolver so PW or LCAO rebuilds all cell-dependent state.

An invalid phase-1 frame leaves the previous UnitCell untouched. Once phase 2
has started, ESolver caches may be partially rebuilt, so a later failure is not
recoverable. Such a failure closes the socket and terminates all ranks in a
coordinated manner instead of attempting to roll back and continue.

## Computed-Frame Publication

Every calculation creates a new result object initially marked invalid. It is
published only after energy, forces, stress, and virial have all been computed,
validated as finite, and associated with the current frame.

`hasdata` becomes true only after publication. `GETFORCE` can send only the
published frame. The driver never responds with:

- a zero virial standing in for a failed stress calculation;
- NaN or infinity;
- the previous frame's virial;
- an unconverged result presented as successful.

For LCAO, force is calculated before stress so the existing cached stress path
is used correctly. For PW, stress is calculated after the same frame's SCF and
force calculation.

## Protocol and MPI Failure Handling

The accepted state sequence is:

```text
NEEDINIT -> INIT -> READY -> POSDATA -> HAVEDATA -> GETFORCE -> READY
```

The following are fatal protocol errors:

- `POSDATA` before initialization;
- another `POSDATA` while a result is waiting for `GETFORCE`;
- `GETFORCE` without a complete current result;
- negative, overflowing, or unreasonable lengths;
- an atom-count mismatch;
- an unknown header;
- disconnect during a payload.

A peer close is normal only while the driver is waiting for the next header in
an idle state. The base protocol has no portable error response, so ABACUS logs
the failure, closes the socket, and exits instead of returning fabricated
forces.

Only the root rank owns socket I/O. Every root I/O or semantic failure is
broadcast as one status and message before ranks enter later collectives. All
ranks then take the same exit path, preventing one rank from leaving while
others wait in a UnitCell or SCF collective.

Diagnostics include mode, socket step, protocol state, failed field, and useful
scaled cell diagnostics such as determinant, condition number, and inverse
residual. Full position buffers are not dumped.

## Fixed-Cell Compatibility

`AbacusSocketIO(variable_cell=False)` remains the default.

In fixed mode:

- no new INPUT keyword is emitted;
- cell changes are rejected using the established fixed-cell path;
- species grouping and force reverse mapping are unchanged;
- no variable-cell rebuild occurs;
- existing energy and force results remain regression-compatible.

A zero tensor must not be advertised as a calculated stress. The Python
calculator will expose stress only when real stress calculation is enabled:

- variable mode always enables and exposes real stress;
- fixed mode with `cal_stress=1` computes and exposes real stress;
- default fixed mode keeps the current energy/force cost and treats stress as
  unavailable rather than returning a fake zero.

Changing the wire integer implementation from native `int` to `int32_t` must
produce byte-identical messages on the supported x86_64 system.

An older ABACUS executable that does not recognize `socket_variable_cell` must
fail explicitly. The Python wrapper must not silently fall back to fixed-cell
operation.

## Test Strategy

### Unit and focused tests

Extend the existing socket test target and add focused tests for:

- exact 12-byte headers;
- exact int32 and binary64 payloads;
- partial reads and writes;
- clean EOF versus mid-payload disconnect;
- negative and overflowing counts;
- asymmetric synthetic cell and tensor transpose round trips;
- cell/inverse consistency;
- NaN, infinity, negative determinant, singular and ill-conditioned cells;
- stress-to-virial sign and unit conversion;
- result publication and stale-data prevention;
- fixed-mode unchanged-cell acceptance and changed-cell rejection;
- synchronized MPI error propagation where practical in the module test suite.

The new INPUT item receives parsing, reset, validation, and help/documentation
tests. Source additions are wired deterministically through the relevant
`CMakeLists.txt`.

### Real DFT reference system

Use a small Si system with the same pseudopotential and compatible LCAO orbital
data for PW and LCAO. The reference cell is deliberately triclinic and the
atoms are displaced slightly so off-diagonal stresses have a measurable signal.

For each backend, evaluate:

- isotropic strain;
- three diagonal strains;
- `xy`, `xz`, and `yz` symmetric shear;
- positive and negative perturbations at `1e-4`, `3e-4`, and `1e-3`.

Each result record includes executable identity, commit, module, precision,
cell, volume, condition number, SCF convergence, energy, forces, raw ABACUS
stress, socket virial, and ASE stress.

### CPU and GPU matrix

| Backend | CPU double | Single V100 double | GPU single/mixed smoke |
|---|---|---|---|
| PW | full suite | full suite | yes |
| LCAO | full suite | full suite | yes |

The full suite comprises FileIO/socket comparison, all six finite-difference
directions, `UnitCellFilter`, `FrechetCellFilter`, official i-PI isotropic NPT,
and official i-PI flexible NPT.

Reference GPU tests use `precision=double`. LCAO also uses
`gint_precision=double`. PW single and LCAO single or mixed precision receive
additional smoke coverage but do not define the sign or reference tolerance.

### ASE filter acceptance

For both filters and each PW/LCAO CPU/GPU path:

- start from a cell containing volume, anisotropic, and shear distortion;
- complete at least three optimizer-accepted cell steps;
- retain positive, well-conditioned cells;
- finish with lower energy and lower maximum stress than the initial frame;
- show at least one nonzero shear response in the expected direction;
- agree with the finite-difference cell-force direction.

Optimizer trial points need not all decrease energy because a line search may
reject them.

### Official i-PI acceptance

For each PW/LCAO CPU/GPU combination:

- run at least ten valid isotropic-NPT steps;
- run at least ten valid flexible-NPT steps.

CPU-double PW and LCAO each also run a 50-step stability trajectory. Every step
must return a new real virial, retain a positive nonsingular cell, and keep all
energies, forces, virials, and barostat momenta finite. Flexible NPT must change
off-diagonal cell entries. Pressure response must have the correct direction,
and the short runs must show no explosive volume change or obvious one-way
conserved-quantity drift.

These trajectories validate the interface and short-time integration stability,
not converged NPT thermodynamic sampling.

### Numerical acceptance

- Finite-difference stress: relative error at most 2 percent and absolute error
  at most `5e-4 eV/Angstrom^3`.
- CPU/GPU double stress: relative error at most 1 percent and absolute error at
  most `1e-4 eV/Angstrom^3`.
- Identical-frame FileIO/socket agreement must be materially tighter than the
  finite-difference threshold.
- Cell and wire-layout unit tests use near-binary64 tolerances.
- The three strain sizes must show a finite-difference convergence plateau.

If a threshold fails, first tighten SCF convergence, PW cutoff, grid accuracy,
or other basis controls. Test references or tolerances are not loosened merely
to make a failure pass.

## Build and Runtime Strategy

Reuse the installed develop module dependency stack:

```text
abacus/develop-git-079fd0c-260724-sm70-auto
```

It provides the matching NVHPC GNU branch, OpenMPI, FFTW, libxc, ELPA, CUDA,
and GPU architecture support. Existing build directories are inspected for
compatibility and used for incremental compilation and relinking of affected
targets. The implementation does not install or rebuild an unrelated complete
ABACUS software stack.

All ABACUS runtime and MPI tests set `OMP_NUM_THREADS=1`. Socket and MPI tests
run outside restricted sandboxes so sandbox socket warnings are not mistaken
for ABACUS failures.

Single-V100 jobs use an SAI QOS that permits one GPU, such as
`rush-1o2gpu` or `flood-1o2gpu`. Slurm scripts do not request explicit CPU core
or memory counts. They start from a clean shell, load the ABACUS module, and
record the job ID and GPU identity.

Verification reports contain exact commands, exit status, ABACUS version,
source commit, module name, device and precision settings, Slurm job IDs, log
paths, and measured numerical errors.

## Branch and Backport Sequence

1. Develop in the isolated `feature/variable-cell-ipi-socket` worktree based on
   `8b60f83c3`.
2. Never modify `feature/fixed-cell-ipi-socket` for this feature.
3. Complete develop implementation and the full validation matrix.
4. Create a separate `feature/variable-cell-ipi-socket-lts-3.10` worktree from
   the fork's LTS 3.10.x branch.
5. Backport only the required API and implementation changes.
6. Validate the LTS branch independently with
   `abacus/LTSv3.10.1-sm70-auto`.
7. Commit and push the develop and LTS branches separately.

## Governance

- Do not add new `GlobalV`, `GlobalC`, or `PARAM` cross-layer control paths.
  Pass variable-cell mode and validation dependencies explicitly where
  practical.
- Do not encode workflow switches as mutable state that can be changed from
  multiple places.
- Keep header dependencies minimal and retain the C++11 baseline.
- Do not add default arguments to existing interfaces.
- Add focused tests for the INPUT behavior, socket serialization, cell update,
  PW/LCAO behavior, and heterogeneous precision paths.
- Update both required INPUT documentation files.
- Run the governance checker against the branch diff and report its exact
  output.

## Completion Criteria

The develop feature is complete only when:

- all focused unit and INPUT tests pass;
- fixed-cell regression tests pass;
- CPU PW and LCAO numerical validation passes;
- single-V100 PW and LCAO validation passes;
- both ASE cell filters pass;
- official i-PI isotropic and flexible NPT tests pass;
- sign, transpose, unit, and precision-boundary evidence is recorded;
- documentation and governance checks pass;
- the verified branch is committed and pushed without modifying the fixed-cell
  branch.

The LTS feature is complete only after a separate backport and independent LTS
runtime validation.
