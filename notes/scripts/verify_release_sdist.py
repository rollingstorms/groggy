"""Build an extracted source distribution and qualify its wheel without installing.

Usage: python notes/scripts/verify_release_sdist.py dist --out /tmp/source-wheels
The input is one .tar.gz or a directory containing exactly one. Maturin must be
available. --offline uses cached Cargo dependencies; no Python packages install.
"""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sdist", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    archives = sorted(args.sdist.glob("*.tar.gz")) if args.sdist.is_dir() else [args.sdist]
    if len(archives) != 1:
        parser.error("Expected exactly one source distribution")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("Output directory must be empty")
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out.resolve()
    with tempfile.TemporaryDirectory(prefix="groggy-sdist-check-") as folder:
        root = Path(folder).resolve()
        with tarfile.open(archives[0], "r:gz") as archive:
            for member in archive.getmembers():
                destination = (root / member.name).resolve()
                if root not in destination.parents or not (member.isfile() or member.isdir()):
                    raise ValueError("Unsafe source member: " + member.name)
            archive.extractall(root)
        projects = list(root.glob("*/pyproject.toml"))
        if len(projects) != 1:
            raise ValueError("Expected one source project")
        project = projects[0].parent
        locks = [project / "Cargo.lock", project / "python-groggy/Cargo.lock"]
        if not all(path.is_file() for path in locks):
            raise ValueError("Source distribution must retain both resolved Cargo lockfiles")
        command = ["maturin", "build", "--release", "--locked", "--out", str(output),
                   "--interpreter", sys.executable]
        if args.offline:
            command.append("--offline")
        env = dict(os.environ, PYO3_PYTHON=sys.executable,
                   PYO3_USE_ABI3_FORWARD_COMPATIBILITY="0")
        subprocess.run(command, cwd=project, env=env, check=True)
        print("Source SHA256: " + hashlib.sha256(archives[0].read_bytes()).hexdigest(),
              file=sys.stderr)
        verifier = Path(__file__).with_name("verify_release_wheel.py")
        command = [sys.executable, str(verifier), str(output)]
        if args.report:
            command += ["--report", str(args.report.resolve())]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
