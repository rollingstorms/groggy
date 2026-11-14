"""Check a built wheel without installing it or importing the source checkout.

Usage: python notes/scripts/verify_release_wheel.py /path/to/groggy.whl
Uses an isolated subprocess with site-packages disabled and a temporary unpacked
wheel. This is not a cross-platform installation or optional integration test.
"""
import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import zipfile
from email.parser import BytesParser
from pathlib import Path


SMOKE = r'''
import importlib.metadata
import json
from pathlib import Path
import runpy
import sys
from tempfile import TemporaryDirectory

root = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(root))
import groggy as gr
from groggy import _groggy

assert root in Path(gr.__file__).resolve().parents, gr.__file__
assert root in Path(_groggy.__file__).resolve().parents, _groggy.__file__
version = importlib.metadata.version("groggy")
assert version == gr.__version__ == _groggy.__version__ == sys.argv[2]
missing = [name for name in gr.__all__ if not hasattr(gr, name)]
assert not missing, missing

graph = gr.Graph()
assert graph.node_count() == graph.edge_count() == 0
node = graph.add_node(score=1.0000000000000002)
before = graph.read_view()
graph.writer().set_node_attr(node, "score", 2.5)
assert before.node_attr(node, "score") == 1.0000000000000002
assert graph.read_view().node_attr(node, "score") == 2.5
assert gr.GraphMatrix.from_data([[2., 0.], [0., 3.]]).determinant() == 6.
run_workflow = runpy.run_path(sys.argv[3])["run_workflow"]
with TemporaryDirectory(prefix="groggy-wheel-workflow-") as folder:
    workflows = [run_workflow(folder), run_workflow(folder, durable=True)]
print(json.dumps({"version": version, "exports": len(gr.__all__),
                  "package": str(Path(gr.__file__).resolve()),
                  "workflows": workflows}))
'''


def verify(wheel_path, tests, release_tag):
    repo = Path(__file__).resolve().parents[2]
    manifest = repo / "python-groggy/Cargo.toml"
    workflow = Path(__file__).with_name("core_release_workflow.py")
    expected = re.search(r'^version = "([^"]+)"', manifest.read_text(), re.M)[1]
    core_version = re.search(r'^version = "([^"]+)"', (repo / "Cargo.toml").read_text(), re.M)[1]
    if core_version != expected:
        raise ValueError("Core and binding crate versions disagree")
    if release_tag and release_tag not in (expected, "v" + expected):
        raise ValueError("Release tag disagrees with crate version: " + release_tag)
    with tempfile.TemporaryDirectory(prefix="groggy-wheel-check-") as work:
        root = Path(work)
        with zipfile.ZipFile(wheel_path) as wheel:
            metadata_files = [name for name in wheel.namelist()
                              if name.endswith(".dist-info/METADATA")]
            assert len(metadata_files) == 1, metadata_files
            metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
            assert metadata["Name"] == "groggy"
            assert metadata["Version"] == expected, metadata["Version"]
            assert metadata["Requires-Python"].replace(" ", "") == ">=3.8,<3.13"
            assert "groggy/py.typed" in wheel.namelist()
            assert "groggy/_groggy.pyi" in wheel.namelist()
            for name in wheel.namelist():
                destination = (root / name).resolve()
                if root.resolve() not in destination.parents:
                    raise ValueError("Wheel member escapes extraction directory: " + name)
            wheel.extractall(root)
        result = subprocess.run(
            [sys.executable, "-I", "-S", "-c", SMOKE, str(root), expected, str(workflow)],
            cwd=root, capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError("Wheel smoke failed:\n" + result.stdout + result.stderr)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if tests != "none":
            # Tests and their child interpreters must use the unpacked artifact.
            # Site-packages is enabled here only for test dependencies; the
            # independent smoke above deliberately runs with it disabled.
            env = dict(os.environ, PYTHONPATH=str(root))
            if tests == "maintained":
                command = [sys.executable, str(repo / "notes/scripts/verify_immutable_view_migration.py")]
            else:
                command = [sys.executable, "-m", "pytest", "-q",
                           str(repo / "tests/test_core_release_workflow.py"),
                           str(repo / "tests/test_python_owner_lifetime.py")]
            subprocess.run(command, cwd=root, env=env, check=True,
                           stdout=sys.stderr, stderr=sys.stderr, timeout=1800)
        return {"wheel": wheel_path.name,
                "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                "smoke": json.loads(result.stdout), "tests": tests}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path, help="Wheel file or directory of wheels for this interpreter")
    parser.add_argument("--tests", choices=["none", "ownership", "maintained"], default="none",
                        help="Optional artifact-backed tests; requires pytest and suite dependencies")
    parser.add_argument("--report", type=Path, help="Write a JSON receipt after every check passes")
    parser.add_argument("--release-tag", help="Require a published tag to match the crate version")
    args = parser.parse_args()
    wheels = sorted(args.wheel.glob("*.whl")) if args.wheel.is_dir() else [args.wheel]
    if not wheels:
        parser.error("No wheels found")
    if args.report and args.report.exists():
        parser.error("Report already exists; choose a fresh path")
    report = {"python": sys.version, "platform": platform.platform(),
              "artifacts": [verify(path, args.tests, args.release_tag) for path in wheels]}
    encoded = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
