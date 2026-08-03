#!/usr/bin/env bash
set -eu

if [[ ${1:-} == "2" ]]; then
    if [[ $# -ne 2 || $2 != */MODULE_RELAX_socket_driver_mpi_failure_test ]]; then
        echo "CMake empty-flag launcher received unexpected arguments: $*" >&2
        exit 83
    fi
    echo "ABACUS_SOCKET_MPI_FATAL stage=runner rank=1 message=rank-selective runner failure" >&2
    exit 17
fi

bash_executable=$1
wrapper=$2
requested_case=${3:-all}
test_directory=$(mktemp -d /tmp/abacus-socket-mpi-wrapper.XXXXXX)
trap 'rm -rf "$test_directory"' EXIT

fake_launcher="${test_directory}/portable-launcher"
fake_test_executable="${test_directory}/socket-driver-test"
touch "$fake_test_executable"

cat >"$fake_launcher" <<'LAUNCHER'
#!/usr/bin/env bash
set -eu

if [[ $EXPECTED_LAUNCH_CASE == "flagged" ]]; then
    expected=(
        "--process-count"
        "2"
        "--pre-flag"
        "pre value"
        "$EXPECTED_TEST_EXECUTABLE"
        "--post-flag"
        "post value"
    )
elif [[ $EXPECTED_LAUNCH_CASE == "empty" ]]; then
    expected=(
        "2"
        "--pre-flag"
        "pre value"
        "$EXPECTED_TEST_EXECUTABLE"
        "--post-flag"
        "post value"
    )
else
    echo "unknown fake launcher case: $EXPECTED_LAUNCH_CASE" >&2
    exit 84
fi

for argument in "$@"; do
    if [[ $argument == "--oversubscribe" ]]; then
        echo "fake launcher rejects OpenMPI-only --oversubscribe" >&2
        exit 81
    fi
done
if [[ $# -ne ${#expected[@]} ]]; then
    echo "fake launcher argument count mismatch: got $# expected ${#expected[@]}" >&2
    exit 80
fi
for ((index = 0; index < ${#expected[@]}; ++index)); do
    argument_index=$((index + 1))
    actual=${!argument_index}
    if [[ $actual != "${expected[index]}" ]]; then
        echo "fake launcher argument ${index} mismatch: got '$actual' expected '${expected[index]}'" >&2
        exit 82
    fi
done

echo "ABACUS_SOCKET_MPI_FATAL stage=runner rank=1 message=rank-selective runner failure" >&2
exit 17
LAUNCHER
chmod +x "$fake_launcher"

run_flagged_case()
{
    EXPECTED_LAUNCH_CASE=flagged \
    EXPECTED_TEST_EXECUTABLE="$fake_test_executable" \
        "$bash_executable" "$wrapper" \
        "$fake_launcher" 1 2 "$fake_test_executable" 2 2 \
        "--process-count" \
        "--pre-flag" "pre value" \
        "--post-flag" "post value"
}

run_empty_case()
{
    EXPECTED_LAUNCH_CASE=empty \
    EXPECTED_TEST_EXECUTABLE="$fake_test_executable" \
        "$bash_executable" "$wrapper" \
        "$fake_launcher" 0 2 "$fake_test_executable" 2 2 \
        "--pre-flag" "pre value" \
        "--post-flag" "post value"
}

case $requested_case in
    flagged)
        run_flagged_case
        ;;
    empty)
        run_empty_case
        ;;
    all)
        run_flagged_case
        run_empty_case
        ;;
    *)
        echo "unknown wrapper test case: $requested_case" >&2
        exit 85
        ;;
esac
