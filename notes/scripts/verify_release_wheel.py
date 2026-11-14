"""Check a built wheel without installing it or importing the source checkout.

Usage: python notes/scripts/verify_release_wheel.py /path/to/groggy.whl
Uses an isolated subprocess and a temporary unpacked wheel. Existing interpreter
dependencies remain available; this is not a cross-platform installation test.
"""
import argparse
import json
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
import sys

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
print(json.dumps({"version": version, "exports": len(gr.__all__),
                  "package": str(Path(gr.__file__).resolve())}))
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[2] / "python-groggy/Cargo.toml"
    expected = re.search(r'^version = "([^"]+)"', manifest.read_text(), re.M)[1]
    with tempfile.TemporaryDirectory(prefix="groggy-wheel-check-") as work:
        root = Path(work)
        with zipfile.ZipFile(args.wheel) as wheel:
            metadata_files = [name for name in wheel.namelist()
                              if name.endswith(".dist-info/METADATA")]
            assert len(metadata_files) == 1, metadata_files
            metadata = BytesParser().parsebytes(wheel.read(metadata_files[0]))
            assert metadata["Name"] == "groggy"
            assert metadata["Version"] == expected, metadata["Version"]
            assert "groggy/py.typed" in wheel.namelist()
            assert "groggy/_groggy.pyi" in wheel.namelist()
            for name in wheel.namelist():
                destination = (root / name).resolve()
                if root.resolve() not in destination.parents:
                    raise ValueError("Wheel member escapes extraction directory: " + name)
            wheel.extractall(root)
        result = subprocess.run(
            [sys.executable, "-I", "-c", SMOKE, str(root), expected],
            cwd=root, capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError("Wheel smoke failed:\n" + result.stdout + result.stderr)
        print(json.dumps({"wheel": args.wheel.name,
                          "smoke": json.loads(result.stdout)}, indent=2))
        if result.stderr:
            print(result.stderr, file=sys.stderr)


if __name__ == "__main__":
    main()
