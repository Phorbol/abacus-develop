#!/usr/bin/env bash
set -u

mpi_compiler=$1
test_executable=$2
mpi_bin_dir=$(dirname "$mpi_compiler")
mpirun_executable="${mpi_bin_dir}/mpirun"
test_output=$(mktemp /tmp/abacus-socket-mpi-failure.XXXXXX)
trap 'rm -f "$test_output"' EXIT

if [[ ! -x "$mpirun_executable" ]]; then
    echo "MPI test launcher is not executable: $mpirun_executable"
    exit 1
fi

set +e
/usr/bin/timeout 6 "$mpirun_executable" --oversubscribe -np 2 \
    "$test_executable" >"$test_output" 2>&1
launch_status=$?
set -e
cat "$test_output"

if [[ $launch_status -eq 0 ]]; then
    echo "MPI failure test unexpectedly exited zero"
    exit 1
fi
if [[ $launch_status -eq 124 ]]; then
    echo "MPI failure test timed out"
    exit 1
fi

grep -Fq \
    "ABACUS_SOCKET_MPI_FATAL stage=runner rank=1 message=rank-selective runner failure" \
    "$test_output"
