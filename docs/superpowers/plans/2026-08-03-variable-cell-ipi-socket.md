# Variable-Cell i-PI Socket Calculator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an explicitly enabled PW/LCAO variable-cell i-PI socket calculator that safely updates a complete 3x3 cell and returns a correctly signed, correctly serialized virial to ASE and official i-PI.

**Architecture:** Keep socket byte I/O, pure frame validation/tensor conversion, ABACUS UnitCell mutation, and protocol state management separate. Validate an entire POSDATA frame before committing it, use existing `setup_cell_after_vc()` plus ESolver cell-update flags for PW/LCAO rebuilds, and publish a result only after converged energy, forces, stress, and virial are finite. Preserve fixed-cell operation behind the default `variable_cell=False` path.

**Tech Stack:** C++11, ABACUS `UnitCell`/PW/LCAO ESolver APIs, POSIX sockets, MPI, GoogleTest/CTest, Python 3.10+, ASE SocketIOCalculator, official i-PI 3.2.0, CMake, Slurm on SAI V100 nodes.

## Global Constraints

- Work only in `/tmp/abacus-variable-cell-ipi-socket` on branch `feature/variable-cell-ipi-socket`; never modify `feature/fixed-cell-ipi-socket`.
- Preserve the base i-PI protocol: 12-byte ASCII headers, native-endian 4-byte integers, and native-endian IEEE-754 8-byte floating-point values.
- Socket units are Bohr, Bohr^-1, Hartree, Hartree/Bohr, and Hartree virial; do not convert through eV, Angstrom, GPa, or kbar in the C++ socket path.
- Use `W_ipi = 0.5 * V * sigma_ab` only after FileIO, finite-difference, and pressure-direction tests confirm the sign and shear convention.
- Require finite, right-handed, nonsingular cells with `det(H) > 0` and 2-norm condition number below `1e12`.
- Keep C++11 compatibility; do not add default arguments to existing interfaces.
- Do not add new cross-layer control through `GlobalV`, `GlobalC`, or `PARAM`; pass mode and tolerances explicitly where practical.
- Do not hide protocol workflow switches in mutable booleans controlled from multiple places; use one explicit state enum.
- Add focused tests before implementation changes, following `superpowers:test-driven-development` RED-GREEN-REFACTOR discipline.
- Set `OMP_NUM_THREADS=1` for ABACUS, MPI, and runtime tests.
- Run socket and MPI tests outside restricted sandboxes.
- Reuse the `abacus/develop-git-079fd0c-260724-sm70-auto` dependency stack and incrementally build affected targets; do not install a second complete ABACUS stack.
- Update `docs/parameters.yaml` and `docs/advanced/input_files/input-main.md` for the new INPUT behavior.
- Record exact commands, executable versions, commits, module names, Slurm job IDs, and numerical errors.
- Enforce the approved stress limits: finite difference `rtol=2e-2`,
  `atol=5e-4 eV/Angstrom^3`; CPU/GPU double `rtol=1e-2`,
  `atol=1e-4 eV/Angstrom^3`; identical-frame FileIO/socket
  `rtol=2e-3`, `atol=5e-5 eV/Angstrom^3`. Single/mixed smoke uses
  `rtol=5e-2`, `atol=2e-3 eV/Angstrom^3` only as an additional boundary
  check and never establishes sign or reference values.
- The LTS 3.10.x backport is a separate future plan, created only after the develop implementation passes all acceptance checks.

---

## File Structure

### New production files

- `source/source_relax/socket_frame.h`: dependency-light C++11 data types and declarations for 3x3 wire matrices, validation diagnostics, transpose/layout conversion, and stress-to-virial conversion.
- `source/source_relax/socket_frame.cpp`: finite checks, checked size arithmetic, direct one-sided-Jacobi 3x3 SVD condition number, inverse residual, stress symmetry checks, and virial serialization.

### New test and validation files

- `source/source_relax/test/socket_frame_test.cpp`: pure numerical and layout tests, including asymmetric matrices and invalid-cell cases.
- `interfaces/ASE_interface/examples/socketio_variable_cell.py`: reproducible ASE finite-difference, UnitCellFilter, and FrechetCellFilter validator for PW/LCAO and CPU/GPU.
- `interfaces/ASE_interface/examples/ipi_variable_cell/init.xyz`: skewed two-atom Si initial structure for official i-PI validation.
- `interfaces/ASE_interface/examples/ipi_variable_cell/isotropic.xml`: official i-PI isotropic-NPT input.
- `interfaces/ASE_interface/examples/ipi_variable_cell/flexible.xml`: official i-PI flexible-NPT input.
- `interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py`: launches official i-PI and ABACUS, checks trajectory/cell/virial stability, and writes JSON results.
- `interfaces/ASE_interface/examples/ipi_variable_cell/README.md`: exact CPU/GPU commands and interpretation limits.
- `docs/superpowers/verification/2026-08-03-variable-cell-ipi-socket.md`: filled during real verification with commands, versions, job IDs, measured errors, and pass/fail status.

### Existing files to modify

- `source/source_relax/socket_ipi.h`, `source/source_relax/socket_ipi.cpp`: use explicit `std::int32_t` wire integers and checked byte counts.
- `source/source_relax/socket_driver.cpp`: enforce the protocol state enum, validate/commit variable cells, rebuild UnitCell/ESolver state, reject stale results, and return real virials.
- `source/source_relax/CMakeLists.txt`: add `socket_frame.cpp` deterministically.
- `source/source_relax/test/CMakeLists.txt`: add the focused `MODULE_RELAX_socket_frame_test` target and any sources needed by driver tests.
- `source/source_relax/test/socket_ipi_test.cpp`: exact int32/binary64, partial-payload, and overflow-adjacent socket tests.
- `source/source_io/module_parameter/input_parameter.h`: add `bool socket_variable_cell = false` next to `socket_driver`.
- `source/source_io/module_parameter/read_input_item_system.cpp`: register, reset, synchronize, and validate `socket_variable_cell`.
- `source/source_io/test_serial/read_input_item_test.cpp`: INPUT item validation and auto-setting tests.
- `source/source_io/test/read_input_ptest.cpp`: default-value test.
- `source/source_esolver/esolver_ks.cpp`: publish the actual per-run SCF convergence result in the existing `ESolver::conv_esolver` field.
- `interfaces/ASE_interface/abacuslite/core.py`: add `variable_cell`, INPUT conflict checks, fixed-cell gating, and honest stress property exposure.
- `interfaces/ASE_interface/examples/socketio.py`: preserve the fixed-cell benchmark and clarify real-stress opt-in.
- `docs/advanced/interface/ase.md`: document fixed and variable socket modes, cell conventions, pressure ownership, and examples.
- `docs/parameters.yaml`: add generated INPUT metadata.
- `docs/advanced/input_files/input-main.md`: add generated user-facing INPUT documentation.

---

### Task 1: Make Wire Integer and Payload Sizes Explicit

**Files:**
- Modify: `source/source_relax/socket_ipi.h`
- Modify: `source/source_relax/socket_ipi.cpp`
- Modify: `source/source_relax/socket_driver.cpp` (mechanical integer API call-site rename only)
- Modify: `source/source_relax/test/socket_ipi_test.cpp`

**Interfaces:**
- Produces: `std::int32_t IpiSocket::read_int32()` and `void IpiSocket::write_int32(std::int32_t value)`.
- Produces: checked `read_doubles(std::size_t)`/`write_doubles(...)` that reject byte-count overflow before calling POSIX I/O.
- Consumed later by: `Socket_Driver::socket_driver()` and protocol-state tests.

- [ ] **Step 1: Add failing byte-level int32 and binary64 tests**

Add tests that exchange known bit patterns rather than only numeric values:

```cpp
TEST(IpiSocketTest, Int32UsesExactlyFourNativeEndianBytes)
{
    const std::int32_t expected = INT32_C(0x12345678);
    // Peer receives exactly sizeof(std::int32_t), then closes.
    // Assert memcmp(received.data(), &expected, 4) == 0.
    socket.write_int32(expected);
}

TEST(IpiSocketTest, DoubleUsesExactlyEightNativeEndianBytes)
{
    const double expected = -1234.5;
    // Peer receives exactly sizeof(double), then closes.
    // Assert memcmp(received.data(), &expected, 8) == 0.
    socket.write_double(expected);
}
```

Also add `ReadInt32HandlesSplitPayload` where the peer sends two bytes at a
time, and `ReadInt32RejectsMidPayloadClose`. Exercise the write-all loop with
a large double vector while the peer reads in small chunks, and confirm the
reconstructed byte stream is exact.

- [ ] **Step 2: Configure the isolated build, compile, and verify RED**

Run:

```bash
module purge
module load abacus/develop-git-079fd0c-260724-sm70-auto
cmake -S . -B /tmp/abacus-variable-cell-build-cpu \
  -DBUILD_TESTING=ON -DENABLE_MPI=ON -DENABLE_LCAO=ON \
  -DENABLE_ELPA=ON -DUSE_CUDA=OFF
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_ipi_test -j4
```

Expected: compilation fails because `read_int32`/`write_int32` do not exist.
This compile-failure RED is explicitly approved for the brand-new C++ API;
runtime behavior and boundary cases must still demonstrate assertion-failure RED.

- [ ] **Step 3: Implement explicit wire types and compile-time checks**

Use this public API in `socket_ipi.h`:

```cpp
#include <cstdint>

std::int32_t read_int32();
void write_int32(std::int32_t value);
```

In `socket_ipi.cpp`, add C++11-compatible checks:

```cpp
#include <limits>

static_assert(sizeof(std::int32_t) == 4, "i-PI requires a 4-byte integer");
static_assert(sizeof(double) == 8, "i-PI requires an 8-byte float");
static_assert(std::numeric_limits<double>::is_iec559,
              "i-PI requires IEEE-754 double precision");
```

Rename every internal socket integer call; do not retain ambiguous `read_int()` or `write_int()` compatibility wrappers. Before allocating or multiplying, check `n <= SIZE_MAX / sizeof(double)` and throw `std::overflow_error` with the requested element count.

- [ ] **Step 4: Build and run GREEN**

Rebuild only the target in the same isolated build tree:

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_ipi_test -j4
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure -R '^MODULE_RELAX_socket_ipi_test$'
```

Expected: all socket byte/partial-read tests pass.

- [ ] **Step 5: Commit Task 1**

```bash
git add source/source_relax/socket_ipi.h \
        source/source_relax/socket_ipi.cpp \
        source/source_relax/socket_driver.cpp \
        source/source_relax/test/socket_ipi_test.cpp
git commit -m "fix: use explicit i-PI wire integer type"
```

---

### Task 2: Add the Pure Cell and Virial Numerical Contract

**Files:**
- Create: `source/source_relax/socket_frame.h`
- Create: `source/source_relax/socket_frame.cpp`
- Create: `source/source_relax/test/socket_frame_test.cpp`
- Modify: `source/source_relax/CMakeLists.txt`
- Modify: `source/source_relax/test/CMakeLists.txt`

**Interfaces:**
- Produces: `SocketFrame::Matrix9`, a row-major `std::array<double, 9>`.
- Produces: `SocketFrame::CellValidation validate_ipi_cell(const Matrix9&, const Matrix9&, double, double, double)`.
- Produces: `Matrix9 transpose_matrix9(const Matrix9&)`.
- Produces: `bool checked_position_count(std::int32_t, int, std::size_t&, std::string&)`, called before allocation or payload reads.
- Produces: `bool validate_positions(const std::vector<double>&, std::size_t, std::string&)` for size and finite-value checks.
- Produces: `SocketFrame::VirialConversion make_ipi_virial(const Matrix9&, double, double, double)`.
- Consumed later by: socket driver frame preflight and result publication.

- [ ] **Step 1: Define the dependency-light header in the failing test**

The test must expect these exact declarations:

```cpp
namespace SocketFrame
{
using Matrix9 = std::array<double, 9>;

struct CellValidation
{
    bool ok;
    std::string message;
    double determinant_bohr3;
    double condition_number_2;
    double inverse_residual;
    Matrix9 computed_inverse_wire_bohr_inv;
};

struct VirialConversion
{
    bool ok;
    std::string message;
    Matrix9 wire_virial_hartree;
    double max_antisymmetric_component;
};

Matrix9 transpose_matrix9(const Matrix9& values);
CellValidation validate_ipi_cell(const Matrix9& cell_wire,
                                 const Matrix9& inverse_wire,
                                 double max_condition_number,
                                 double inverse_absolute_tolerance,
                                 double inverse_relative_tolerance);
bool validate_positions(const std::vector<double>& positions_bohr,
                        std::size_t coordinate_count,
                        std::string& message);
bool checked_position_count(std::int32_t nat_socket,
                            int nat_expected,
                            std::size_t& coordinate_count,
                            std::string& message);
VirialConversion make_ipi_virial(const Matrix9& stress_ry_per_bohr3,
                                  double volume_bohr3,
                                  double antisymmetric_absolute_tolerance,
                                  double antisymmetric_relative_tolerance);
}
```

- [ ] **Step 2: Write failing layout, validity, and sign tests**

Cover these exact cases:

```cpp
TEST(SocketFrameTest, TransposeKeepsAllNineUniqueEntries)
{
    Matrix9 in = {{1, 2, 3, 4, 5, 6, 7, 8, 9}};
    Matrix9 expected = {{1, 4, 7, 2, 5, 8, 3, 6, 9}};
    EXPECT_EQ(expected, transpose_matrix9(in));
}

TEST(SocketFrameTest, VirialUsesPositiveHalfVolumeAndWireTranspose)
{
    Matrix9 stress = {{1, 2, 3, 2, 5, 6, 3, 6, 9}};
    VirialConversion out = make_ipi_virial(stress, 4.0, 1e-12, 1e-12);
    Matrix9 expected = {{2, 4, 6, 4, 10, 12, 6, 12, 18}};
    ASSERT_TRUE(out.ok) << out.message;
    EXPECT_EQ(expected, out.wire_virial_hartree);
}
```

Also test: a right-handed triclinic cell; internally recomputed inverse agrees
with a known inverse; inconsistent supplied inverse; negative determinant; zero
determinant; condition number below/at/above `1e12`; NaN/Inf in either matrix;
wrong/negative atom count; count rejection before any allocation; nonfinite
positions; small stress asymmetry is averaged; large stress asymmetry is
rejected; nonpositive volume is rejected. `checked_position_count()` must first
require `nat_socket == nat_expected`, then prove `3 * nat_socket` is
representable as `std::size_t`.

- [ ] **Step 3: Build the new target and verify RED**

Add the target name to `source/source_relax/test/CMakeLists.txt` but no implementation body yet:

```cmake
AddTest(
  TARGET MODULE_RELAX_socket_frame_test
  SOURCES socket_frame_test.cpp ../socket_frame.cpp
)
```

Run:

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_frame_test -j4
```

Expected: link or assertion failures for the unimplemented contract.

- [ ] **Step 4: Implement stable 3x3 validation and conversion**

Implement a scaled one-sided Jacobi SVD directly on the three columns of the
3x3 matrix. Scale the matrix by its largest absolute element first, track the
right singular vectors while rotating column pairs `(0,1)`, `(0,2)`,
`(1,2)` until their dot products are below
`32 * epsilon * sqrt(norm_i * norm_j)`, and derive singular values from final
column norms. Reject failure to converge after 32 sweeps. Use the resulting
`V S^-1 U.T` to produce `computed_inverse_wire_bohr_inv`; do not use
`Matrix3::Inverse()`, whose singular fallback is unsuitable here. This avoids
forming `H.T * H`, which would square a condition number near `1e12`.

Compute the determinant from the scaled matrix in `long double`, rescale it
only after its sign is established, and reject a nonfinite representable volume.

Compute the inverse residual as the maximum absolute component of
`H * Hinv - I`; accept it only when it is below
`inverse_absolute_tolerance + inverse_relative_tolerance * condition_number_2 * epsilon`.

For virial conversion, reject nonfinite data and excessive asymmetry, average
the accepted stress with its transpose, multiply by `0.5 * volume_bohr3`, then
transpose to wire order. Initial driver thresholds are explicit constants:
`max_condition_number=1e12`, inverse residual
`64*epsilon + 64*epsilon*condition_number`, and stress antisymmetry
`1e-10 + 1e-8*max_abs(stress)` in Ry/Bohr^3. Any later threshold change must
be justified by recorded double/single precision evidence, not by merely
relaxing a failing reference.

- [ ] **Step 5: Run numerical GREEN and sanitizer-friendly edge cases**

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_frame_test -j4
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure -R '^MODULE_RELAX_socket_frame_test$'
```

Expected: every matrix/layout/invalid-input test passes, including the rotated
diagonal matrix with known condition number and the deliberately asymmetric
wire tensor.

- [ ] **Step 6: Commit Task 2**

```bash
git add source/source_relax/socket_frame.h \
        source/source_relax/socket_frame.cpp \
        source/source_relax/test/socket_frame_test.cpp \
        source/source_relax/CMakeLists.txt \
        source/source_relax/test/CMakeLists.txt
git commit -m "feat: add socket frame numerical contract"
```

---

### Task 3: Add the Explicit Variable-Cell INPUT Flag

**Files:**
- Modify: `source/source_io/module_parameter/input_parameter.h`
- Modify: `source/source_io/module_parameter/read_input_item_system.cpp`
- Modify: `source/source_io/test_serial/read_input_item_test.cpp`
- Modify: `source/source_io/test/read_input_ptest.cpp`
- Modify: `docs/parameters.yaml`
- Modify: `docs/advanced/input_files/input-main.md`

**Interfaces:**
- Produces: `Input_para::socket_variable_cell`, default `false`.
- Produces: INPUT keyword `socket_variable_cell`.
- Guarantees: enabled mode implies `socket_driver`, `cal_force`, and `cal_stress`, uses `calculation=scf`, `esolver_type=ksdft`, `basis_type` PW/LCAO, and zero `press1/2/3`.
- Consumed later by: ASE INPUT generation and `Socket_Driver`.

- [ ] **Step 1: Write failing default and validation tests**

In `read_input_ptest.cpp` add:

```cpp
EXPECT_FALSE(param.inp.socket_variable_cell);
```

In `read_input_item_test.cpp`, add cases that expect:

```cpp
param.input.socket_variable_cell = true;
param.input.socket_driver = false;
EXPECT_EXIT(item.check_value(item, param), ::testing::ExitedWithCode(1), "");

param.input.socket_driver = true;
param.input.calculation = "scf";
param.input.esolver_type = "ksdft";
param.input.basis_type = "pw";
param.input.press1 = param.input.press2 = param.input.press3 = 0.0;
EXPECT_NO_THROW(item.check_value(item, param));
```

Add death tests for nonzero `press1`, `press2`, or `press3`, non-KS solver,
and unsupported basis. Add reset tests asserting `cal_force` and `cal_stress`
become true when the mode is enabled.

- [ ] **Step 2: Build INPUT tests and verify RED**

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_IO_read_item_serial MODULE_IO_input_test_para -j4
```

Expected: compilation fails because `socket_variable_cell` is absent.

- [ ] **Step 3: Implement the INPUT item without new globals**

Add next to `socket_driver`:

```cpp
bool socket_variable_cell = false; ///< accept cell updates from i-PI POSDATA
```

Register `socket_variable_cell` directly after `socket_driver`. Its reset hook
sets `cal_force` and `cal_stress` true when enabled. Its check hook validates
the mode against the final `Parameter` object and emits a concrete error naming
the conflicting field. Do not add a default argument or a new `PARAM` access in
the socket layer.

- [ ] **Step 4: Update both generated INPUT documentation sources**

Document that the default is false, the mode is explicit, complete 3x3 cell
updates are accepted, stress is mandatory, the external barostat owns pressure,
and `press1/2/3` must remain zero.

- [ ] **Step 5: Run INPUT GREEN**

```bash
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure -R 'MODULE_IO_(read_item_serial|input_test_para)$'
```

Expected: parser, reset, conflict, and default-value tests pass. The executable-level
help and `--check-input` checks are deliberately performed in Task 8 after the
incremental executable build exists.

- [ ] **Step 6: Commit Task 3**

```bash
git add source/source_io/module_parameter/input_parameter.h \
        source/source_io/module_parameter/read_input_item_system.cpp \
        source/source_io/test_serial/read_input_item_test.cpp \
        source/source_io/test/read_input_ptest.cpp \
        docs/parameters.yaml \
        docs/advanced/input_files/input-main.md
git commit -m "feat: add variable-cell socket input"
```

---

### Task 4: Expose the ASE Opt-In and Honest Stress Properties

**Files:**
- Modify: `interfaces/ASE_interface/abacuslite/core.py:591-746`
- Modify: `interfaces/ASE_interface/abacuslite/core.py:749-784`

**Interfaces:**
- Produces: `AbacusSocketIO(..., variable_cell=False, **kwargs)`.
- Produces: `_socket_inp(inp, variable_cell)` and `_input_bool(value, name)`.
- Guarantees: variable mode emits `socket_variable_cell=1` and `cal_stress=1`; fixed mode still rejects cell changes.
- Guarantees: default fixed mode does not advertise fake stress; fixed mode with real `cal_stress=1` and variable mode do advertise stress.

- [ ] **Step 1: Write failing Python tests**

Add tests in `TestAbacusCalculator`:

```python
def test_socketio_variable_cell_input_enables_stress(self):
    inp = AbacusSocketIO._socket_inp({}, variable_cell=True)
    self.assertEqual(inp['socket_variable_cell'], 1)
    self.assertEqual(inp['cal_force'], 1)
    self.assertEqual(inp['cal_stress'], 1)

def test_socketio_rejects_variable_cell_input_conflict(self):
    with self.assertRaisesRegex(ValueError, 'socket_variable_cell'):
        AbacusSocketIO._socket_inp(
            {'socket_variable_cell': 0}, variable_cell=True)

def test_variable_cell_skips_fixed_cell_guard(self):
    calc = object.__new__(AbacusSocketIO)
    calc.variable_cell = True
    calc._reference_cell = None
    first = Atoms('Si', cell=[5, 5, 5], pbc=True)
    second = first.copy()
    second.cell[0, 1] = 0.2
    calc._check_cell_change(first)
    calc._check_cell_change(second)
```

Also test accepted boolean spellings (`True`, `1`, `"true"`, `"1"`), rejected
ambiguous strings, retained fixed-cell rejection, fixed default implemented
properties excluding stress, fixed `cal_stress=1` including stress, and variable
mode including stress.

- [ ] **Step 2: Run Python tests and verify RED**

Create an isolated Python environment without installing ABACUS itself:

```bash
module purge
module load conda/anaconda3
python3 -m venv /tmp/abacus-variable-cell-venv
/tmp/abacus-variable-cell-venv/bin/pip install -e interfaces/ASE_interface
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/abacuslite/core.py
```

Expected: failures for the missing constructor flag and new helper behavior.

- [ ] **Step 3: Implement the Python mode and property contract**

Add `variable_cell=False` as a keyword before `**kwargs`, parse it with
`self._input_bool(variable_cell, 'variable_cell')` so the string `"false"`
cannot become truthy accidentally, then call
`self._socket_inp(inp, self.variable_cell)`. Rename `_check_fixed_cell()` to
`_check_cell_change()` and return immediately in variable mode.

Set an instance copy of `implemented_properties` before the superclass setup:

```python
real_stress = self.variable_cell or self._input_bool(
    inp.get('cal_stress', False), 'cal_stress')
self.implemented_properties = ['energy', 'free_energy', 'forces']
if real_stress:
    self.implemented_properties.append('stress')
```

Only convert/populate stress in `calculate()` when real stress is enabled.
Do not return the driver's legacy zero virial as a calculated fixed-mode stress.

- [ ] **Step 4: Run Python GREEN**

```bash
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/abacuslite/core.py
```

Expected: all embedded `core.py` unit tests pass without launching ABACUS.

- [ ] **Step 5: Commit Task 4**

```bash
git add interfaces/ASE_interface/abacuslite/core.py
git commit -m "feat: expose variable-cell ASE socket mode"
```

---

### Task 5: Publish the Actual KS SCF Convergence State

**Files:**
- Modify: `source/source_esolver/esolver_ks.cpp:122-171`
- Modify: `source/source_relax/socket_driver.cpp`
- Create: `source/source_relax/test/socket_driver_test.cpp`
- Modify: `source/source_relax/test/CMakeLists.txt`

**Interfaces:**
- Produces: the existing `ModuleESolver::ESolver::conv_esolver` field reflects the most recent `runner()` call for PW and LCAO.
- Produces: socket result publication refuses a frame whose solver reports nonconvergence.
- Consumed later by: the complete variable-cell driver implementation in Task 6.

- [ ] **Step 1: Add a failing forced-nonconvergence driver test**

Create the driver test target and a UNIX-socket peer with a minimal
`FakeESolver`. Its `runner()` sets the inherited `conv_esolver=false`; the
peer completes INIT and one fixed-cell POSDATA frame. Assert that the driver
closes the frame without publishing `HAVEDATA` and that the diagnostic contains
`SCF did not converge`.

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_driver_test -j4
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure -R '^MODULE_RELAX_socket_driver_test$'
```

Expected: the old driver reaches `HAVEDATA` because it does not inspect the
solver convergence field.

- [ ] **Step 3: Remove the shadowed convergence state and gate publication**

In `ESolver_KS::runner()`, replace the local convergence variable with the
existing member:

```cpp
this->conv_esolver = false;
// pass this->conv_esolver to iter_finish()
// use this->conv_esolver in the break condition
// pass this->conv_esolver to after_scf()
```

In the driver, immediately after `runner()`, reject the frame when
`p_esolver->conv_esolver` is false. Do not add another virtual method or global
status flag, and do not publish partially computed energy, forces, or virial.

- [ ] **Step 4: Run convergence GREEN**

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_driver_test -j4
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure -R '^MODULE_RELAX_socket_driver_test$'
```

Include a second fake case whose `runner()` sets `conv_esolver=true` and
assert it reaches `HAVEDATA`. Real PW and LCAO convergence are exercised in
Tasks 9 and 10.

- [ ] **Step 5: Commit Task 5**

```bash
git add source/source_esolver/esolver_ks.cpp \
        source/source_relax/socket_driver.cpp \
        source/source_relax/test/socket_driver_test.cpp \
        source/source_relax/test/CMakeLists.txt
git commit -m "fix: publish KS solver convergence state"
```

---

### Task 6: Implement Atomic Cell Commit, Protocol State, and Real Virial

**Files:**
- Modify: `source/source_relax/socket_driver.cpp`
- Modify: `source/source_relax/test/CMakeLists.txt`
- Modify: `source/source_relax/test/socket_driver_test.cpp`

**Interfaces:**
- Consumes: `Input_para::socket_variable_cell`.
- Consumes: all `SocketFrame` validation and conversion functions.
- Consumes: `IpiSocket::read_int32()`/`write_int32()`.
- Produces: `DriverState { NeedInit, Ready, HasData }` as the only protocol workflow state.
- Produces: atomic variable-cell UnitCell updates and complete `ComputedFrame` publication.

- [ ] **Step 1: Write failing state-machine and stale-result tests**

Build a UNIX-socket test peer and a minimal `FakeESolver` derived from
`ModuleESolver::ESolver`. Its `runner()` records `ucell.latvec`, `omega`, and
update flags; `cal_energy()`, `cal_force()`, and `cal_stress()` return unique
finite values. Test these sequences:

```text
STATUS -> NEEDINIT
INIT -> STATUS -> READY
POSDATA(new triclinic cell) -> STATUS -> HAVEDATA
GETFORCE -> STATUS -> READY
```

Assert the received physical results:

```cpp
EXPECT_DOUBLE_EQ(energy_hartree, 0.5 * fake_energy_ry);
EXPECT_DOUBLE_EQ(forces_hartree_bohr[0], 0.5 * fake_force_ry_bohr[0]);
EXPECT_DOUBLE_EQ(virial_wire[0], 0.5 * updated_volume * fake_stress(0, 0));
```

Use nine unique cell values to assert ABACUS row-vector mapping. Add death or
exception tests for POSDATA before INIT, duplicate POSDATA before GETFORCE,
GETFORCE without data, unknown header, atom mismatch, invalid inverse, and
mid-frame disconnect. Add fixed-mode unchanged-cell acceptance and changed-cell
rejection. Confirm a failed second frame cannot send the first frame's result,
and that an exception after UnitCell commit terminates the session rather than
attempting a rollback.

- [ ] **Step 2: Build the driver test and verify RED**

Add a `MODULE_RELAX_socket_driver_test` target with the exact driver, socket,
frame, UnitCell update, and base sources it needs. Build only this target.

Expected: changed-cell test exits with `variable-cell socket updates are not supported yet`, and state-order tests fail because the old two-booleans state accepts/exits inconsistently.

- [ ] **Step 3: Introduce explicit state and unpublished result objects**

Inside `socket_driver.cpp`, define:

```cpp
enum class DriverState { NeedInit, Ready, HasData };

struct ComputedFrame
{
    bool valid = false;
    double energy_hartree = 0.0;
    std::vector<double> forces_hartree_per_bohr;
    SocketFrame::Matrix9 virial_wire_hartree = {{0.0}};
};
```

Only assign the completed local frame to the published frame after convergence,
finite energy/force/stress checks, and successful virial conversion. On
`GETFORCE`, require `state == HasData && published.valid`, send it, clear it,
and transition to `Ready`.

- [ ] **Step 4: Implement preflight and atomic UnitCell commit**

For every POSDATA frame:

1. root reads complete buffers with checked int32 sizes;
2. root validates cell, inverse, count, and positions with `SocketFrame`;
3. validation status/message and frame are broadcast once;
4. fixed mode compares against the original cell and retains its rejection;
5. variable mode computes the absolute ABACUS row-vector cell in Bohr as
   `A_abacus_bohr = transpose(cell_wire)`;
6. transpose the internally SVD-recomputed wire inverse from
   `CellValidation` to obtain `A_abacus_bohr^-1`, then calculate new direct
   coordinates into a temporary vector as
   `taud = Cartesian_positions_bohr * A_abacus_bohr^-1`; the inverse received
   from i-PI is checked for consistency but is never trusted for state mutation;
7. only then assign `latvec = A_abacus_bohr / lat0` and `taud`;
8. call `unitcell::setup_cell_after_vc(ucell, ofs_running, inp.nspin)`;
9. set `cell_parameter_updated` and `ionic_position_updated` correctly;
10. call `runner()`.

Keep `lat0` fixed. Compare cell change against
`32 * epsilon * max(1, max_abs_component(cell))`; below that scale-aware
threshold use the positions-only path and do not force a PW/LCAO cell rebuild.
Retain the existing first-frame species/order warning and force reverse mapping.

- [ ] **Step 5: Compute and publish force, stress, and virial**

After `runner()`:

```cpp
if (!p_esolver->conv_esolver)
{
    throw std::runtime_error("socket step SCF did not converge");
}
```

Compute force, then stress when `inp.cal_stress` is true. Convert the 3x3
`ModuleBase::matrix` to row-major `Matrix9`, call `make_ipi_virial()`, and send
the returned wire-order tensor. In variable mode, absence of stress is fatal.
In fixed mode without stress exposure, retain a protocol zero virial but never
advertise it as an ASE calculated stress.

- [ ] **Step 6: Make failures MPI-coordinated**

Root converts every I/O or semantic error to one integer status plus message,
broadcasts them before any later collective, then all ranks call the same fatal
path. Treat peer close as normal only in `NeedInit` or `Ready`, not `HasData` or
mid-payload. Unknown headers are fatal rather than silent loop exits.

- [ ] **Step 7: Run driver, frame, socket, and UnitCell GREEN**

```bash
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target MODULE_RELAX_socket_ipi_test \
           MODULE_RELAX_socket_frame_test \
           MODULE_RELAX_socket_driver_test \
           MODULE_CELL_unitcell_test_setupcell -j4
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure \
  -R 'MODULE_RELAX_socket_(ipi|frame|driver)_test|MODULE_CELL_unitcell_test_setupcell'
```

Expected: all pass with no stale result and correct changed-cell values.

- [ ] **Step 8: Commit Task 6**

```bash
git add source/source_relax/socket_driver.cpp \
        source/source_relax/test/socket_driver_test.cpp \
        source/source_relax/test/CMakeLists.txt
git commit -m "feat: update cells and virials in socket driver"
```

---

### Task 7: Add Reproducible ASE and Official i-PI Validation Assets

**Files:**
- Create: `interfaces/ASE_interface/examples/socketio_variable_cell.py`
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/init.xyz`
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/isotropic.xml`
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/flexible.xml`
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py`
- Create: `interfaces/ASE_interface/examples/ipi_variable_cell/README.md`

**Interfaces:**
- Produces: command-line validators accepting `--abacus`, `--basis pw|lcao`, `--device cpu|gpu`, `--precision double|single`, `--prepare-only`, `--workdir`, and `--output`.
- Produces: `--pp-orb-root` so staged Slurm runs resolve pseudopotentials and
  numerical orbitals without depending on a source checkout.
- Produces: JSON containing all energy/force/stress/virial/cell errors and stability checks.
- Consumed later by: CPU and V100 acceptance tasks.

- [ ] **Step 1: Write validator self-tests with a deterministic analytic calculator**

Add `--self-test` to exercise deformation generation, six-component finite
differences, JSON schema, and pass/fail threshold logic without launching
ABACUS. Use an analytic quadratic cell energy whose exact stress is known:

```python
energy = 0.5 * np.sum(k * strain**2)
stress = np.array([k[i] * strain[i] for i in range(6)]) / volume
```

Assert the validator recovers all six components and catches a deliberately
negated virial.

- [ ] **Step 2: Run self-tests and verify RED**

```bash
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/examples/socketio_variable_cell.py --self-test
```

Expected: file absent, then failing assertions until the six-component logic is implemented.

- [ ] **Step 3: Implement ASE validation modes**

The script must:

- construct the same displaced triclinic Si2 frame for PW and LCAO;
- run identical-frame FileIO/socket comparisons;
- scan `delta = 1e-4, 3e-4, 1e-3` for `xx, yy, zz, yz, xz, xy`;
- use symmetric off-diagonal strain with `gamma/2` entries;
- run at least three accepted UnitCellFilter steps;
- run at least three accepted FrechetCellFilter steps;
- require positive determinant and finite values for every frame;
- require lower final energy and maximum stress than the initial accepted frame;
- require a nonzero shear response in the finite-difference-consistent direction;
- write raw values and `atol + rtol` decisions to JSON.

Every real-result JSON record includes executable version/hash, source commit,
module, backend, device, precision settings, cell, volume, condition number,
SCF convergence, energy, forces, raw ABACUS stress, socket virial, ASE stress,
and all applied thresholds. The test fails if a required field is absent.
The three strain sizes must show a convergence plateau; a single perturbation
that happens to pass does not satisfy the finite-difference check.

Use `precision=double` for reference paths and set `gint_precision=double` for
LCAO reference paths.

- [ ] **Step 4: Implement pinned official i-PI inputs and runner**

Pin validation to official `i-PI==3.2.0`. `isotropic.xml` uses isotropic NPT;
`flexible.xml` uses the flexible MTTK barostat and an upper-triangular skewed
cell. Both use a conservative timestep and barostat time constant, write cell,
potential, conserved quantity, and checkpoint output, and connect through a
unique UNIX socket name supplied by the runner.

`run_validation.py` launches i-PI, waits for its UNIX socket, launches ABACUS
with `socket_variable_cell=1`, checks ten or fifty completed steps, and always
terminates both processes in `finally`.

Before the requested trajectory, the runner performs a paired five-step
isotropic direction probe from identical coordinates, cell, zero barostat
momentum, and random seed. It obtains the initial hydrostatic pressure from the
FileIO reference, sets official i-PI targets to `p_initial - 2 GPa` and
`p_initial + 2 GPa` using i-PI's declared pressure unit, and requires the
higher-pressure replica to finish at smaller volume than the lower-pressure
replica. Record both target pressures, volume sequences, and the comparison in
JSON; this is the barostat-level sign test and does not replace six-strain
finite differences.

- [ ] **Step 5: Run validator self-tests GREEN**

```bash
/tmp/abacus-variable-cell-venv/bin/pip install 'i-PI==3.2.0'
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/examples/socketio_variable_cell.py --self-test
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py --self-test
```

Expected: analytic sign, shear-factor, schema, and lifecycle tests pass.

- [ ] **Step 6: Commit Task 7**

```bash
git add interfaces/ASE_interface/examples/socketio_variable_cell.py \
        interfaces/ASE_interface/examples/ipi_variable_cell
git commit -m "test: add variable-cell socket validation workflows"
```

---

### Task 8: Build the Develop Executables Incrementally

**Files:**
- No source changes expected.
- Create only ignored/out-of-tree build products under `/tmp`.

**Interfaces:**
- Produces: `/tmp/abacus-variable-cell-build-cpu/abacus_basic_para`.
- Produces: `/tmp/abacus-variable-cell-build-gpu/abacus_basic_gpu`.
- Produces: a compute-node-visible runtime bundle at
  `/home/gengjianrui/bin/abacus-variable-cell-runtime`; this contains only the
  two executables, Python interface/validators, PP/ORB inputs, venv, and logs,
  not a second ABACUS installation.
- Consumed later by: CPU and Slurm validation.

- [ ] **Step 1: Record module and compiler identity**

```bash
module purge
module load abacus/develop-git-079fd0c-260724-sm70-auto
module list
module show abacus/develop-git-079fd0c-260724-sm70-auto
cmake --version
mpirun --version
```

Save output in the verification document, including the module-provided
OpenMPI, FFTW, libxc, BLAS, ELPA, NVHPC/CUDA paths. Confirm CMake resolves those
installed dependencies and does not download or build replacement libraries.

- [ ] **Step 2: Incrementally build the CPU LCAO-capable executable**

```bash
cmake -S . -B /tmp/abacus-variable-cell-build-cpu \
  -DBUILD_TESTING=ON -DENABLE_MPI=ON -DENABLE_LCAO=ON \
  -DENABLE_ELPA=ON -DUSE_CUDA=OFF
cmake --build /tmp/abacus-variable-cell-build-cpu \
  --target abacus_basic_para -j4
```

This executable runs both `basis_type=pw` and `basis_type=lcao`; no installation step is performed.

- [ ] **Step 3: Configure and incrementally build the GPU executable**

```bash
cmake -S . -B /tmp/abacus-variable-cell-build-gpu \
  -DBUILD_TESTING=ON -DENABLE_MPI=ON -DENABLE_LCAO=ON \
  -DENABLE_ELPA=ON -DUSE_CUDA=ON
cmake --build /tmp/abacus-variable-cell-build-gpu \
  --target abacus_basic_gpu -j4
```

- [ ] **Step 4: Record executable identities**

```bash
/tmp/abacus-variable-cell-build-cpu/abacus_basic_para --version
/tmp/abacus-variable-cell-build-gpu/abacus_basic_gpu --version
git rev-parse HEAD
```

Expected: both identify the current branch commit and start without missing shared libraries.

- [ ] **Step 5: Verify INPUT help and check a valid variable-cell case**

Use the Task 7 validator to generate, but not execute, the exact Si2 PW socket
case, then check it:

```bash
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/examples/socketio_variable_cell.py \
  --prepare-only --basis pw --device cpu --precision double \
  --pp-orb-root tests/PP_ORB \
  --workdir /tmp/abacus-variable-cell-input-check \
  --output /tmp/abacus-variable-cell-input-check/prepare.json
/tmp/abacus-variable-cell-build-cpu/abacus_basic_para \
  -h socket_variable_cell
cd /tmp/abacus-variable-cell-input-check
/tmp/abacus-variable-cell-build-cpu/abacus_basic_para --check-input
```

Expected: help states the explicit opt-in and the valid case exits zero. Record
the case INPUT and command output in the verification document.

- [ ] **Step 6: Stage only the runtime artifacts on shared storage**

`/tmp` is node-local and cannot be referenced by Slurm jobs. Create a compact
shared bundle without installing another ABACUS tree:

```bash
mkdir -p /home/gengjianrui/bin/abacus-variable-cell-runtime/bin \
         /home/gengjianrui/bin/abacus-variable-cell-runtime/jobs \
         /home/gengjianrui/bin/abacus-variable-cell-runtime/logs \
         /home/gengjianrui/bin/abacus-variable-cell-runtime/results
cp /tmp/abacus-variable-cell-build-cpu/abacus_basic_para \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/
cp /tmp/abacus-variable-cell-build-gpu/abacus_basic_gpu \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/
rsync -a interfaces/ASE_interface/ \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/
rsync -a tests/PP_ORB/ \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB/
module purge
module load conda/anaconda3
python3 -m venv /home/gengjianrui/bin/abacus-variable-cell-runtime/venv
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/pip install \
  -e /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface \
  'i-PI==3.2.0'
```

Record checksums of both staged executables and the source commit. All CPU/GPU
runtime scripts below use this explicit shared path.

---

### Task 9: Run CPU PW and LCAO Numerical Acceptance

**Files:**
- Create/update: `docs/superpowers/verification/2026-08-03-variable-cell-ipi-socket.md`
- No production source change unless a failing test is diagnosed under `superpowers:systematic-debugging` and fixed with a new RED-GREEN commit.

**Interfaces:**
- Consumes: CPU executable and Task 7 validators.
- Produces: recorded PW/LCAO FileIO, finite-difference, filter, and i-PI evidence.

- [ ] **Step 1: Create exact CPU-MISC Slurm scripts**

`/home/gengjianrui/bin/abacus-variable-cell-runtime/jobs/cpu-pw-ase.sbatch`
contains:

```bash
#!/bin/bash
#SBATCH --job-name=abacus-vc-cpu-pw
#SBATCH --partition=CPU-MISC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/home/gengjianrui/bin/abacus-variable-cell-runtime/logs/cpu-pw-ase-%j.out

module purge
module load abacus/develop-git-079fd0c-260724-sm70-auto
export OMP_NUM_THREADS=1
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/examples/socketio_variable_cell.py \
  --abacus '/home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_para' \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --basis pw --device cpu --precision double \
  --output /home/gengjianrui/bin/abacus-variable-cell-runtime/results/cpu-pw-ase.json
```

Create `cpu-lcao-ase.sbatch` in the same jobs directory by changing only the
job name, basis, and result name to `lcao`. Create four explicit i-PI scripts
in that directory from the same header using
`run_validation.py`, the same executable, `--device cpu`, and every Cartesian
product of `--basis pw|lcao` with `--mode isotropic|flexible`; use
`--steps 50` and a distinct result JSON for each.

- [ ] **Step 2: Submit the two ASE jobs and record IDs**

```bash
mkdir -p /home/gengjianrui/bin/abacus-variable-cell-runtime/logs \
         /home/gengjianrui/bin/abacus-variable-cell-runtime/results \
         /home/gengjianrui/bin/abacus-variable-cell-runtime/jobs
cpu_pw_job=$(sbatch --parsable \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/jobs/cpu-pw-ase.sbatch)
cpu_lcao_job=$(sbatch --parsable \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/jobs/cpu-lcao-ase.sbatch)
squeue -j "${cpu_pw_job},${cpu_lcao_job}"
sacct -j "${cpu_pw_job},${cpu_lcao_job}" \
  --format=JobID,State,ExitCode,Elapsed,NodeList
```

Record both returned job IDs. Use `squeue` only for monitoring and rerun the
`sacct` command after both jobs leave the queue for final evidence. Expected:
FileIO/socket agreement, all six finite differences, UnitCellFilter, and
FrechetCellFilter pass for PW and LCAO; the LCAO reference uses
`gint_precision=double`.

- [ ] **Step 3: Submit official i-PI CPU isotropic and flexible NPT**

Submit the four scripts created in Step 1 and record their job IDs. The command
body in, for example, `cpu-pw-flexible.sbatch` is exactly:

```bash
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/examples/ipi_variable_cell/run_validation.py \
  --abacus '/home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_para' \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --basis pw --device cpu --mode flexible --steps 50 \
  --output /home/gengjianrui/bin/abacus-variable-cell-runtime/results/cpu-pw-flexible.json
```

The isotropic script changes `--mode` and the output basename to `isotropic`;
the LCAO scripts also change `--basis`. Submit them with `sbatch --parsable`,
store all four returned IDs in shell variables, and use the same `squeue` then
`sacct` pattern as Step 2. Expected: positive cells, finite quantities, correct
pressure direction, changing flexible-cell shear, and no explosive drift.

- [ ] **Step 4: Record CPU evidence and commit**

Create the verification document with a table of every measured maximum error,
the exact commands, Slurm job IDs/final states, executable version, and shared
JSON paths. Then:

```bash
git add docs/superpowers/verification/2026-08-03-variable-cell-ipi-socket.md
git commit -m "docs: record CPU variable-cell socket verification"
```

---

### Task 10: Run Single-V100 PW/LCAO Double and Single/Mixed Acceptance

**Files:**
- Update: `docs/superpowers/verification/2026-08-03-variable-cell-ipi-socket.md`
- Create temporary Slurm scripts and JSON logs outside the repository.

**Interfaces:**
- Consumes: GPU executable and Task 7 validators.
- Produces: Slurm job IDs and GPU PW/LCAO double plus single/mixed evidence.

- [ ] **Step 1: Create a clean, exact one-V100 PW script**

Write
`/home/gengjianrui/bin/abacus-variable-cell-runtime/jobs/gpu-pw-double-ase.sbatch`
with this exact content; do not add CPU or memory directives:

```bash
#!/bin/bash
#SBATCH --job-name=abacus-vc-socket
#SBATCH --partition=4V100PX
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --qos=rush-1o2gpu
#SBATCH --output=/home/gengjianrui/bin/abacus-variable-cell-runtime/logs/gpu-pw-ase-%j.out

module purge
module load abacus/develop-git-079fd0c-260724-sm70-auto
source /opt/sai_config/mps_mapping.d/${SLURM_JOB_PARTITION}.bash
export OMP_NUM_THREADS=1
nvidia-smi
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/examples/socketio_variable_cell.py \
  --abacus '/home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_gpu' \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --basis pw --device gpu --precision double \
  --output /home/gengjianrui/bin/abacus-variable-cell-runtime/results/gpu-pw-double-ase.json
```

If `rush-1o2gpu` is rejected by accounting limits, change only the QOS line to
`flood-1o2gpu` and record that change; do not add CPU or memory directives.
Create the LCAO double script by changing the job/result names and
`--basis lcao`.

- [ ] **Step 2: Submit GPU PW double and LCAO double validators**

Submit bounded jobs for the two ASE validators and record IDs immediately with
`sbatch --parsable` into `gpu_pw_job` and `gpu_lcao_job`. Monitor using
`squeue -j "${gpu_pw_job},${gpu_lcao_job}"` and inspect completion using
`sacct -j "${gpu_pw_job},${gpu_lcao_job}" --format=JobID,State,ExitCode,Elapsed,NodeList`.
Expected: both full double suites pass CPU/GPU stress criteria.

- [ ] **Step 3: Submit official i-PI GPU NPT validators**

Create four scripts from the same resource header. Replace the ASE command with:

```bash
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  /home/gengjianrui/bin/abacus-variable-cell-runtime/ASE_interface/examples/ipi_variable_cell/run_validation.py \
  --abacus '/home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_gpu' \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --basis pw --device gpu --mode flexible --steps 10 \
  --output /home/gengjianrui/bin/abacus-variable-cell-runtime/results/gpu-pw-flexible.json
```

Use distinct `pw/lcao` and `isotropic/flexible` values and result names for
the other three scripts. Expected: the same finite/positive/stable criteria as
CPU, with off-diagonal flexible-cell motion.

- [ ] **Step 4: Submit precision-boundary smoke jobs**

Copy the exact ASE scripts from Step 1. For PW, change to
`--precision single` and `gpu-pw-single-ase.json`. For LCAO, select the
validator's supported mixed mode, which writes `gint_precision=mix`, and use
`gpu-lcao-mixed-ase.json`. Inspect protocol output through the validator and
confirm every payload remains binary64. These smoke jobs use wider measured
comparison tolerances but must remain finite and physically directed.

- [ ] **Step 5: Record job evidence and commit**

Add job IDs, node/GPU identity, module, precision settings, maximum CPU/GPU
differences, NPT checks, and JSON/log paths to the verification document:

```bash
git add docs/superpowers/verification/2026-08-03-variable-cell-ipi-socket.md
git commit -m "docs: record V100 variable-cell socket verification"
```

---

### Task 11: Update User Documentation and Fixed-Cell Example

**Files:**
- Modify: `docs/advanced/interface/ase.md:106-175`
- Modify: `interfaces/ASE_interface/examples/socketio.py`
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/README.md`

**Interfaces:**
- Produces: complete user instructions for fixed mode, variable mode, ASE filters, official i-PI, pressure ownership, units, and limitations.

- [ ] **Step 1: Add documentation assertions to review checklist**

The documentation must include all literal strings:

```text
variable_cell=True
socket_variable_cell 1
press1/press2/press3 must be zero
Hartree
Hartree/Bohr
virial
UnitCellFilter
FrechetCellFilter
flexible
```

Verify with `rg` before editing; expected: several strings are absent.

- [ ] **Step 2: Document the two modes and scientific conventions**

Explain that fixed mode rejects cell changes; variable mode accepts full 3x3
transport with six physical strain degrees; i-PI uses upper-triangular cells;
ASE obtains stress as `-virial/volume`; ABACUS returns a real virial only when
stress is enabled; and external pressure belongs to the ASE/i-PI controller.

Include concise PW/LCAO examples for both filters and direct official i-PI.

- [ ] **Step 3: Run documentation and Python example checks**

```bash
rg -n 'variable_cell=True|socket_variable_cell 1|UnitCellFilter|FrechetCellFilter|flexible' \
  docs/advanced/interface/ase.md \
  interfaces/ASE_interface/examples/socketio.py \
  interfaces/ASE_interface/examples/ipi_variable_cell/README.md
/tmp/abacus-variable-cell-venv/bin/python -m py_compile \
  interfaces/ASE_interface/examples/socketio.py \
  interfaces/ASE_interface/examples/socketio_variable_cell.py \
  interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py
```

Expected: all required concepts are present and Python files compile.

- [ ] **Step 4: Commit Task 11**

```bash
git add docs/advanced/interface/ase.md \
        interfaces/ASE_interface/examples/socketio.py \
        interfaces/ASE_interface/examples/ipi_variable_cell/README.md
git commit -m "docs: explain variable-cell socket workflows"
```

---

### Task 12: Final Verification, Review, and Push

**Files:**
- Update if necessary: `docs/superpowers/verification/2026-08-03-variable-cell-ipi-socket.md`
- No production edits during verification without a new failing test and a separate fix commit.

**Interfaces:**
- Produces: a reviewable, pushed develop feature branch with complete evidence.

- [ ] **Step 1: Invoke verification discipline**

Use `superpowers:verification-before-completion` before making any completion
claim. Run fresh commands rather than citing earlier output.

- [ ] **Step 2: Run the complete focused suite**

```bash
OMP_NUM_THREADS=1 ctest --test-dir /tmp/abacus-variable-cell-build-cpu \
  --output-on-failure \
  -R 'MODULE_RELAX_socket_(ipi|frame|driver)_test|MODULE_IO_(read_item_serial|input_test_para)|MODULE_CELL_unitcell_test_setupcell'
/tmp/abacus-variable-cell-venv/bin/python \
  interfaces/ASE_interface/abacuslite/core.py
```

Expected: zero failures.

- [ ] **Step 3: Run governance and diff hygiene checks**

```bash
python3 tools/03_code_analysis/agent_governance_check.py \
  --base 8b60f83c3e62af75ad57c4c6c61a52a8acdc4d60 \
  --head HEAD --format text
git diff --check 8b60f83c3e62af75ad57c4c6c61a52a8acdc4d60..HEAD
git status --short --branch
```

Expected: no governance findings, no whitespace errors, clean worktree.

- [ ] **Step 4: Audit every acceptance artifact**

Open all CPU/GPU JSON outputs and verify the verification document reports the
same numbers. Confirm every promised PW/LCAO, CPU/V100, filter, finite-
difference, isotropic-NPT, flexible-NPT, double, and single/mixed row has an
explicit PASS or a documented failure. A missing row blocks completion.

- [ ] **Step 5: Request code review**

Use `superpowers:requesting-code-review` against base `8b60f83c3` and address
all correctness findings with focused tests and separate commits.

- [ ] **Step 6: Push only the variable-cell branch**

```bash
git push -u origin feature/variable-cell-ipi-socket
```

Then verify:

```bash
git rev-parse HEAD
git rev-parse origin/feature/variable-cell-ipi-socket
git -C /home/gengjianrui/bin/abacus-develop status --short --branch
```

Expected: local and remote variable-cell commits match, and the original
fixed-cell checkout remains clean on `feature/fixed-cell-ipi-socket`.

- [ ] **Step 7: Stop before LTS work**

Report the develop results and ask for the LTS phase. Create the LTS branch and
backport plan only after the develop branch is accepted; do not cherry-pick or
edit the LTS branch in this task.
