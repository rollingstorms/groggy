"""One cross-surface acceptance fixture, shared with the executable example."""
import runpy
from pathlib import Path

import pytest


@pytest.mark.parametrize("durable", [False, True])
def test_core_workflow_retains_temporal_table_and_numerical_contracts(tmp_path, durable):
    example = Path(__file__).resolve().parents[1] / "notes/scripts/core_release_workflow.py"
    run = runpy.run_path(str(example))["run_workflow"]
    result = run(tmp_path, durable=durable)
    assert result["initial"] < result["edited"] < result["restored"]
