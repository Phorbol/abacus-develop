# Variable-cell ABACUS socket validation with official i-PI

These inputs are pinned to official i-PI 3.2.0. `isotropic.xml` exercises
the isotropic NPT barostat; `flexible.xml` exercises the flexible MTTK
barostat with an upper-triangular skew cell. Both use conservative time steps
and barostat time constants and write potential, conserved quantity, cell,
virial, positions, and a checkpoint.

Use the existing ABACUS develop module and record the exact module/binary in
every validation artifact. Do not rebuild the full branch merely to prepare
cases:

```bash
module av abacus
module load abacus/develop-git-<site-build>
ABACUS_BIN=$(command -v abacus)
abacus --version
```

For later real Task 9/10 runs, use the exact Task 8 shared runtime paths below,
which contain the staged executables built from this feature branch:

```text
/home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_para
/home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_gpu
/home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB
/home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python
```

Install the pinned upstream i-PI tag into the existing validation environment:

```bash
/tmp/abacus-variable-cell-venv/bin/pip install \
  "git+https://github.com/i-pi/i-pi.git@v3.2.0"
```

Deterministic tests do not launch ABACUS or an i-PI trajectory:

```bash
/tmp/abacus-variable-cell-venv/bin/python ../socketio_variable_cell.py --self-test
/tmp/abacus-variable-cell-venv/bin/python run_validation.py --self-test
```

Prepare staged inputs (the runner also parses rendered XML with i-PI 3.2.0):

```bash
python run_validation.py --prepare-only --basis pw --device cpu \
  --precision double --abacus "$ABACUS_BIN" \
  --pp-orb-root /staged/PP_ORB --workdir prepared-pw
python run_validation.py --prepare-only --basis lcao --device gpu \
  --precision double --mode flexible --abacus "$ABACUS_BIN" \
  --pp-orb-root /staged/PP_ORB --workdir prepared-lcao-gpu
```

Each prepare-only command writes `<workdir>/prepare.json` with absolute case,
rendered XML, pseudopotential/orbital paths, ABACUS version/hash/source/module,
backend/device/precision settings, and official i-PI parser metadata.

Real acceptance examples:

```bash
OMP_NUM_THREADS=1 /home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  ../socketio_variable_cell.py --basis pw --device cpu --precision double \
  --abacus "mpirun -np 4 /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_para" \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --workdir ase-pw-cpu
OMP_NUM_THREADS=1 /home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  ../socketio_variable_cell.py --basis lcao --device cpu --precision double \
  --abacus "mpirun -np 4 /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_para" \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --workdir ase-lcao-cpu
OMP_NUM_THREADS=1 /home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  ../socketio_variable_cell.py --basis pw --device gpu --precision double \
  --abacus "mpirun -np 4 /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_gpu" \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --workdir ase-pw-gpu
OMP_NUM_THREADS=1 /home/gengjianrui/bin/abacus-variable-cell-runtime/venv/bin/python \
  run_validation.py --basis lcao --device gpu --precision double \
  --mode flexible --steps 50 \
  --abacus "mpirun -np 4 /home/gengjianrui/bin/abacus-variable-cell-runtime/bin/abacus_basic_gpu" \
  --pp-orb-root /home/gengjianrui/bin/abacus-variable-cell-runtime/PP_ORB \
  --workdir ipi-lcao-gpu
```

Every real runner first obtains hydrostatic pressure from an ABACUS FileIO
reference, then runs two identical five-step isotropic replicas at
`p_initial - 2 GPa` and `p_initial + 2 GPa`. Each replica records the initial
sample at step 0 plus five completed MD steps, for six samples total. The higher-pressure replica
must finish at smaller volume. This barostat-direction check complements, but
does not replace, the ASE validator's six central finite differences at
`1e-4`, `3e-4`, and `1e-3`.

Short 10/50-step trajectories establish protocol sign, layout, finite values,
and local numerical stability only. They are not evidence of thermodynamic
equilibration. Double precision is the reference. Single/mixed precision uses
looser smoke thresholds for identical-frame, finite-difference, filter, and
pressure-offset significance decisions. It must still preserve sign, layout,
and finite behavior, but it does not establish reference values.
