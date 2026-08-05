# Official i-PI Property Output Contract Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the pinned official i-PI 3.2.0 isotropic and flexible validators bind and emit both Si-atom force vectors while preserving the existing 22-column binary64 result schema.

**Architecture:** Keep XML rendering and read-only simulation construction, then exercise official i-PI `PropertyOutput.bind` on a disposable output file to validate the output contract without starting a socket or ABACUS.  Use two explicitly indexed `atom_f` entries so the two-atom force payload stays six columns.

**Tech Stack:** Python 3.12, official i-PI 3.2.0, XML templates, existing self-test harness, pytest-free assertion style, Git/CMake governance checks.

## Global Constraints

- Work only in `/tmp/abacus-variable-cell-ipi-socket` on `feature/variable-cell-ipi-socket`; keep `/home/gengjianrui/bin/abacus-develop` clean and unchanged.
- Follow strict RED then GREEN TDD; production/template edits are forbidden before the focused test fails for the expected property-contract reason.
- Keep the workflow pinned to official i-PI 3.2.0; do not add runtime version adaptation.
- Preserve exactly 22 numeric property columns and the existing `values[3:9].reshape(2, 3)` binary64 force layout.
- Do not change ABACUS C++, solver selection, stress/virial conventions, thresholds, fixtures, or job scripts.
- Do not rebuild ABACUS or submit a Slurm job in this implementation task.
- Use LF, remain compatible with the repository's Python baseline, and run exact focused verification.

---

### Task 1: Validate and fix the official i-PI property output contract

**Files:**
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py:29-112,589-917`
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/isotropic.xml:3`
- Modify: `interfaces/ASE_interface/examples/ipi_variable_cell/flexible.xml:3`
- Test: `interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py::_self_test`

**Interfaces:**
- Consumes: official `Simulation.load_from_xml`, `ipi.engine.outputs.PropertyOutput`, and `ipi.engine.properties.getkey`.
- Produces: `validate_official_xml(xml_text: str, directory: Path)` that rejects an unbindable or wrong-width property output and returns the read-only `Simulation` on success.
- Produces: exactly two force fields, `atom_f{electronvolt/angstrom}(0)` followed by `atom_f{electronvolt/angstrom}(1)`, totaling six numeric force columns.

- [ ] **Step 1: Add the focused failing self-test**

Inside `_self_test`, render both templates and independently require this exact
ordered property list (keep the expected tuple local to the test so it does not
merely compare production data with itself):

```python
expected = (
    "step", "potential{electronvolt}", "conserved{electronvolt}",
    "atom_f{electronvolt/angstrom}(0)",
    "atom_f{electronvolt/angstrom}(1)",
    "volume", "cell_h", "virial_md",
)
```

Extract the sole `PropertyOutput.outlist` from each returned simulation and
assert it equals `expected`.  Also mutate one rendered document by replacing
`atom_f{electronvolt/angstrom}(0)` with `not_a_property`; require
`validate_official_xml` to raise `AssertionError`, otherwise raise
`AssertionError("unrecognized i-PI output property was accepted")`.

- [ ] **Step 2: Run RED and preserve the expected failure**

Run outside the restricted sandbox:

```bash
OMP_NUM_THREADS=1 /home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py --self-test
```

Expected: nonzero exit because the current templates contain `forces`, not the
two indexed `atom_f` entries.  Record the exact failure and confirm it is not a
syntax/import/environment error.

- [ ] **Step 3: Implement the minimal official binding check**

Add an immutable module-level expected tuple with the exact eight entries above.
After `Simulation.load_from_xml(..., read_only=True)` succeeds:

```python
property_outputs = [output for output in simulation.outtemplate
                    if isinstance(output, PropertyOutput)]
if len(simulation.syslist) != 1 or len(property_outputs) != 1:
    raise AssertionError("official i-PI output contract requires one system and one property output")
actual = tuple(str(item) for item in property_outputs[0].outlist)
if actual != EXPECTED_IPI_PROPERTIES:
    raise AssertionError("official i-PI property output contract is wrong")
```

Create a disposable `PropertyOutput` with the same `outlist`, bind it to
`simulation.syslist[0]`, call `print_header`, and close/remove the disposable
file in `finally`.  Convert `KeyError`, `ValueError`, or `RuntimeError` from
official binding/header validation into an `AssertionError` with the original
exception chained.  Compute the declared column count from the official
system `property_dict` using `getkey`, treating missing/one-sized properties as
one column, and require exactly 22.

- [ ] **Step 4: Correct both templates minimally**

Replace the single invalid field in both templates:

```text
forces{electronvolt/angstrom}
```

with the ordered pair:

```text
atom_f{electronvolt/angstrom}(0), atom_f{electronvolt/angstrom}(1)
```

Do not change any other XML field, unit, seed, timestep, pressure, thermostat,
barostat, cell, trajectory, or checkpoint setting.

- [ ] **Step 5: Run GREEN and focused regressions**

Run outside the restricted sandbox with `OMP_NUM_THREADS=1`:

```bash
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py --self-test
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python -m py_compile \
  interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python -m flake8 \
  --select E9,F63,F7,F82 \
  interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  interfaces/ASE_interface/examples/socketio_variable_cell.py --self-test
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  interfaces/ASE_interface/abacuslite/core.py -v -k socketio -k variable_cell
```

Then run prepare-only for isotropic and flexible CPU/PW using the existing
runtime executable/module environment.  Require official i-PI 3.2.0 contract
validation, exact source provenance, no `ks_solver` override, and no process or
socket residue.  Run `git diff --check` and the branch governance checker.

- [ ] **Step 6: Commit and self-review**

Review the diff for exact two-atom ordering, units, 22-column preservation,
cleanup on every exception path, no unrelated changes, and pristine test
output.  Then commit only the three implementation files:

```bash
git add interfaces/ASE_interface/examples/ipi_variable_cell/run_validation.py \
  interfaces/ASE_interface/examples/ipi_variable_cell/isotropic.xml \
  interfaces/ASE_interface/examples/ipi_variable_cell/flexible.xml
git commit -m "fix: validate official i-pi property outputs"
```
