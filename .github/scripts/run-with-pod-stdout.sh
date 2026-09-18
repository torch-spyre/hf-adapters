# shellcheck shell=bash
# Shared helper for the _test_matrix.yaml test steps.
#
# run_teed CMD [ARGS...] runs CMD with its combined stdout+stderr teed to PID 1's
# stdout when that is writable. On the self-hosted ARC runner pods PID 1 is the
# runner container's main process, so /proc/1/fd/1 is the pod stdout that the
# IBM Cloud Logs fluent-bit DaemonSet reads (via /var/log/containers, kubelet).
# A plain GHA run: step never reaches pod stdout: the runner agent captures the
# step process output and uploads it to the github.com web console over its
# encrypted websocket, so without this tee the test output is invisible to Cloud
# Logs. pipefail keeps CMD's exit code so a failed test still fails the step. The
# guard falls back to running CMD directly for local dev, where /proc/1/fd/1 is
# not the pod stdout (or not writable).
run_teed() {
  set -o pipefail
  if [[ -w /proc/1/fd/1 ]]; then
    "$@" 2>&1 | tee /proc/1/fd/1
  else
    "$@"
  fi
}
