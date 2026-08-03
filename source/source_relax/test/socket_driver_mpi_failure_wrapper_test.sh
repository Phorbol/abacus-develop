#!/usr/bin/env bash
set -eu

bash_executable=$1
wrapper=$2
test_directory=$(mktemp -d /tmp/abacus-socket-mpi-wrapper.XXXXXX)
trap 'rm -rf "$test_directory"' EXIT

fake_launcher="${test_directory}/portable-launcher"
fake_test_executable="${test_directory}/socket-driver-test"
touch "$fake_test_executable"

cat >"$fake_launcher" <<'LAUNCHER'
#!/usr/bin/env bash
set -eu

expected=(
    "--process-count"
    "2"
    "--pre-flag"
    "pre value"
    "$EXPECTED_TEST_EXECUTABLE"
    "--post-flag"
    "post value"
)

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

EXPECTED_TEST_EXECUTABLE="$fake_test_executable" \
    "$bash_executable" "$wrapper" \
    "$fake_launcher" \
    "--process-count" \
    2 \
    "$fake_test_executable" \
    2 \
    2 \
    "--pre-flag" \
    "pre value" \
    "--post-flag" \
    "post value"
