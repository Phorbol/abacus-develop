# Official i-PI Property Output Contract Fix

## Problem

The real CPU/PW isotropic acceptance job 761238 failed before ABACUS started.
i-PI 3.2.0 rejected the generated property name `forces` while binding the
output stream.  i-PI exposes per-atom force vectors as `atom_f`, with a
zero-based atom argument.  The existing prepare/self-test path called
`Simulation.load_from_xml(..., read_only=True)`, which returns before output
binding and therefore accepted the invalid property name.

## Selected design

Keep the validation pinned to official i-PI 3.2.0.  Both isotropic and flexible
templates will output exactly these two three-component force properties, in
this order:

```text
atom_f{electronvolt/angstrom}(0)
atom_f{electronvolt/angstrom}(1)
```

The resulting property row remains 22 numeric columns, so the existing parser
continues to map columns 3:9 to a binary64 `(2, 3)` force array.

`validate_official_xml` will retain the side-effect-free read-only simulation
construction, then validate the property output using official i-PI objects:

- require the exact ordered property list used by this two-atom fixture;
- require exactly one property output and one system;
- bind a temporary `PropertyOutput` to that system, exercising i-PI's real
  property-key validation without starting sockets, ABACUS, or MD;
- print its header so declared vector sizes are exercised;
- require the declared total to remain 22 columns;
- close and remove the temporary output in all cases.

## Testing

TDD RED must be observed before editing either template or production
validation logic:

- the self-test expects the exact two indexed `atom_f` entries, so the current
  `forces` template fails;
- a mutation to an unrecognized property must be rejected by
  `validate_official_xml`; the current read-only-only implementation accepts it
  and therefore fails the test.

GREEN consists of the two template replacements plus the official output-bind
contract check.  Run the official i-PI 3.2.0 self-test, prepare-only checks for
both barostats, Python compile/lint checks, and the existing socket/ASE focused
tests.  No ABACUS binary rebuild is required because no C++ source changes.

After independent review, restage the changed interface files and source marker
with a durable marker-last transaction, then retry only failed Job 761238.
Continue the remaining CPU jobs strictly serially only if that retry passes all
physics, units, layout, serialization, and shutdown gates.

## Rejected alternatives

- A string-only assertion would be fast but would not reproduce i-PI's output
  binding failure.
- Removing forces would weaken the required acceptance evidence.
- Runtime version-adaptive selection between `forces` and `atom_f` is needless
  complexity because the workflow is deliberately pinned to i-PI 3.2.0.
