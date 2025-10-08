#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

RELEASE_EVENT_FILE="${SCRIPT_DIR}/release_event.json"

if ! command -v act >/dev/null 2>&1; then
  echo "[error] act is not installed. Install it via \`brew install act\` or see the README." >&2
  exit 1
fi

if [[ -n "${ACT_FLAGS:-}" ]]; then
  # Split ACT_FLAGS on whitespace into an array
  read -ra ACT_FLAGS_ARRAY <<< "${ACT_FLAGS}"
else
  ACT_FLAGS_ARRAY=(--container-architecture linux/amd64)
fi

run_cmd() {
  echo "\n==> $*\n"
  "$@"
}

act_run() {
  run_cmd act "${ACT_FLAGS_ARRAY[@]}" "$@"
}

run_push_workflows() {
  run_cmd act push -W .github/workflows/test.yml
  act_run push -W .github/workflows/performance_monitoring.yml -j performance_monitoring
  act_run push -W .github/workflows/performance_monitoring.yml -j memory_analysis
  act_run pull_request -W .github/workflows/performance_monitoring.yml -j benchmark_regression_detection
  act_run push -W .github/workflows/ci.yml -j test
  act_run push -W .github/workflows/test.yml -j test-rust
  act_run push -W .github/workflows/test.yml -j test-python
  act_run pull_request -W .github/workflows/test.yml -j benchmark
}

run_release_workflows() {
  # Build artifacts from ci.yml that only run on releases
  act_run release -W .github/workflows/ci.yml -e "${RELEASE_EVENT_FILE}" -j build-wheels
  act_run release -W .github/workflows/ci.yml -e "${RELEASE_EVENT_FILE}" -j build-sdist

  # Publish workflow will attempt to upload to PyPI; we expect the final step to fail locally.
  set +e
  act "${ACT_FLAGS_ARRAY[@]}" release -W .github/workflows/publish.yml -e "${RELEASE_EVENT_FILE}" -j build-and-publish \
    -s PYPI_TOKEN=fake-token -s TWINE_USERNAME="__token__" -s TWINE_PASSWORD="fake-token"
  status=$?
  set -e
  if [[ $status -ne 0 ]]; then
    echo "[warn] publish workflow exited with ${status}."
    echo "       This usually happens because the final upload step cannot reach PyPI during local tests."
  fi
}

usage() {
  cat <<USAGE
Usage: $(basename "$0") [push|release|all]
  push    Run workflows triggered on push/pull_request events.
  release Run release-only workflows (build wheels/sdist + publish pipeline).
  all     Run both (default).
USAGE
}

MODE=${1:-all}
case "$MODE" in
  push)
    run_push_workflows
    ;;
  release)
    run_release_workflows
    ;;
  all)
    run_push_workflows
    run_release_workflows
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    echo "[error] Unknown mode: $MODE" >&2
    usage
    exit 1
    ;;
 esac
