#!/usr/bin/env bash
set -u

mpi_launcher=$1
numproc_flag_count=$2
numprocs=$3
test_executable=$4
preflags_count=$5
postflags_count=$6
shift 6
test_output=$(mktemp /tmp/abacus-socket-mpi-failure.XXXXXX)
trap 'rm -f "$test_output"' EXIT

if [[ ! $numproc_flag_count =~ ^[01]$ ]]; then
    echo "MPI test numproc flag count must be 0 or 1"
    exit 1
fi
if [[ ! $preflags_count =~ ^[0-9]+$ || ! $postflags_count =~ ^[0-9]+$ ]]; then
    echo "MPI test flag counts must be non-negative integers"
    exit 1
fi
if [[ $# -ne $((numproc_flag_count + preflags_count + postflags_count)) ]]; then
    echo "MPI test flag count does not match supplied arguments"
    exit 1
fi

numproc_flags=()
for ((index = 0; index < numproc_flag_count; ++index)); do
    numproc_flags+=("$1")
    shift
done
preflags=()
for ((index = 0; index < preflags_count; ++index)); do
    preflags+=("$1")
    shift
done
postflags=("$@")

if [[ ! -x "$mpi_launcher" ]]; then
    echo "MPI test launcher is not executable: $mpi_launcher"
    exit 1
fi

mpi_command=("$mpi_launcher")
mpi_command+=("${numproc_flags[@]}")
mpi_command+=("$numprocs")
mpi_command+=("${preflags[@]}")
mpi_command+=("$test_executable")
mpi_command+=("${postflags[@]}")

set +e
/usr/bin/timeout 6 "${mpi_command[@]}" >"$test_output" 2>&1
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
