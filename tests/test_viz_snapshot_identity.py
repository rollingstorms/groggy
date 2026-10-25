"""Real server reuse is scoped to one immutable visualization capture."""
import subprocess
import sys
import textwrap


def test_separate_captures_start_distinct_servers_and_the_same_accessor_reuses_its_server():
    # Process isolation releases all background servers/threads at test completion.
    # Replace notebook rendering only; both HTTP and websocket paths are real.
    code = textwrap.dedent('''
        import json
        import re
        import sys
        import types
        import urllib.request
        from groggy import EdgesTable, GraphTable, NodesTable

        html = []
        display = types.ModuleType("IPython.display")
        display.HTML = lambda value: value
        display.display = html.append
        sys.modules["IPython.display"] = display

        first = GraphTable(NodesTable([70, 12]), EdgesTable([(900, 70, 12)])).viz
        second = NodesTable([44]).viz
        first.show(layout="grid")
        first_port = int(re.search(r"127\\.0\\.0\\.1:(\\d+)", html[-1]).group(1))
        try:
            second.update(layout="grid")
        except RuntimeError as error:
            assert "No active visualization server" in str(error)
        else:
            raise AssertionError("A separate capture reused the first server")

        second.show(layout="grid")
        second_port = int(re.search(r"127\\.0\\.0\\.1:(\\d+)", html[-1]).group(1))
        assert first_port != second_port
        first.show(layout="grid")
        assert int(re.search(r"127\\.0\\.0\\.1:(\\d+)", html[-1]).group(1)) == first_port

        def snapshot(port):
            # Force a direct loopback connection regardless of proxy environment.
            client = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with client.open(f"http://127.0.0.1:{port}/debug/snapshot", timeout=3) as response:
                return json.load(response)

        one = snapshot(first_port)
        two = snapshot(second_port)
        assert {str(node["id"]) for node in one["nodes"]} == {"70", "12"}
        assert {str(node["id"]) for node in two["nodes"]} == {"44"}
        assert [(str(edge["id"]), str(edge["source"]), str(edge["target"]))
                for edge in one["edges"]] == [("900", "70", "12")]
        assert two["edges"] == []
    ''')
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
