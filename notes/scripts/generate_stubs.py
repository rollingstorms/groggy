#!/usr/bin/env python3
"""
Generate Groggy type stubs using stubgen.

This script assumes the extension module has already been built and is importable.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path("/tmp/groggy_stubgen")
    output_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root / "python-groggy" / "python")

    cmd = [
        "stubgen",
        "--inspect-mode",
        "-m",
        "groggy._groggy",
        "-o",
        str(output_dir),
    ]

    print("Generating stubs:", " ".join(cmd))
    result = subprocess.run(cmd, env=env, check=False, text=True)
    if result.returncode != 0:
        return result.returncode

    generated = output_dir / "groggy" / "_groggy.pyi"
    if not generated.exists():
        print("Expected stub was not generated:", generated)
        return 1

    target = repo_root / "python-groggy" / "python" / "groggy" / "_groggy.pyi"
    shutil.copyfile(generated, target)
    print("Updated stub:", target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
