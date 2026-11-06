"""Real visualization snapshots retain selection scope and frozen values."""
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("captured", [False, True, "components", "array"])
def test_visualization_snapshot_keeps_only_selected_entities_after_owner_changes(captured):
    code = textwrap.dedent('''
        import json, re, sys, types, urllib.request
        from groggy import Graph
        html = []
        display = types.ModuleType("IPython.display")
        display.HTML = lambda value: value
        display.display = html.append
        sys.modules["IPython.display"] = display
        graph = Graph(directed=True)
        graph.add_nodes(5)
        graph.add_edge(1, 3)
        graph.add_edge(1, 3)
        graph.add_edge(3, 3)
        graph.add_edge(0, 4)
        selected = graph.subgraph_from_edges([1, 2])
        if CAPTURED is True:
            selected = selected.pin_view()
        elif CAPTURED == "components":
            selected = selected.connected_components()
        elif CAPTURED == "array":
            selected = selected.neighborhood([1], hops=1)
        accessor = selected.viz
        graph.remove_node(1)
        accessor.show(layout="grid")
        port = int(re.search(r"127\\.0\\.0\\.1:(\\d+)", html[-1]).group(1))
        client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with client.open(f"http://127.0.0.1:{port}/debug/snapshot", timeout=3) as response:
            snapshot = json.load(response)
        assert {str(n["id"]) for n in snapshot["nodes"]} == {"1", "3"}, snapshot
        assert {(str(e["id"]), str(e["source"]), str(e["target"])) for e in snapshot["edges"]} == {("1", "1", "3"), ("2", "3", "3")}, snapshot
    ''').replace("CAPTURED", repr(captured))
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
