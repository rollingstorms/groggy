# Workflow Dry-Run Guide

This mini-guide keeps the commands we run before cutting a release or merging a large change. It pairs with `run_workflows.sh`, which wraps [`act`](https://github.com/nektos/act) so we can execute the GitHub Actions workflows locally using Docker.

## Prerequisites

- Docker Desktop (or another Docker runtime) running locally.
- [`act` installed](https://github.com/nektos/act#installation). Homebrew formula: `brew install act`.
- Enough disk space for the runner images (≈1–2 GB on first run).

## First-Time Setup

1. Pull the lightweight runner so repeated runs stay fast:
   ```bash
   act --pull
   ```
2. Verify Docker is available:
   ```bash
   docker info
   ```

## Quick Usage

Run **all push-style checks** (CI, tests, performance monitoring):
```bash
./documentation/workflow_testing/run_workflows.sh push
```

Run **all release workflows** (wheel/SDist builds plus publish pipeline using dry-run secrets):
```bash
./documentation/workflow_testing/run_workflows.sh release
```

Run **everything**:
```bash
./documentation/workflow_testing/run_workflows.sh all
```

Each command prints the underlying `act` calls so you can copy/paste or retry a single job manually.

## Release Event Payload

`act` needs a minimal release payload so the release-only jobs (e.g. `publish.yml`) see a tag. The helper script auto-passes `release_event.json` from this directory:

```json
{
  "release": {
    "tag_name": "v0.0.0-test"
  },
  "ref": "refs/tags/v0.0.0-test"
}
```

Feel free to tweak the tag name when you are rehearsing an actual release candidate.

## Tips

- Use `ACT_LOG_LEVEL=debug` when a job behaves differently than it does on GitHub.
- Pass `--container-architecture linux/amd64` if you are on Apple Silicon and the workflow needs x86 images.
- If a workflow needs secrets, export them before running or extend the script with `-s SECRET=value` flags.
- To iterate on a single job, run `act <event> -W <workflow file> -j <job>` directly; the script just stitches those together for convenience.
