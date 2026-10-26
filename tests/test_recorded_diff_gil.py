"""Recorded diff reconstruction runs without holding Python's runtime lock."""
import sys
import threading

from groggy import Graph


def test_recorded_diff_releases_gil_while_preserving_typed_results():
    graph = Graph()
    graph.add_nodes(30_000)
    state = graph.current_state_id
    ready, begin, progress, done = (threading.Event() for _ in range(4))

    def worker():
        ready.set()
        begin.wait()
        progress.set()
        done.wait()

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(timeout=5)
    interval = sys.getswitchinterval()
    try:
        # Disable ordinary bytecode switching around entry into the native call.
        sys.setswitchinterval(10.0)
        begin.set()
        result = graph.diff_commits(0, state)
        progressed = progress.is_set()
    finally:
        sys.setswitchinterval(interval)
        done.set()
        thread.join(timeout=5)
    assert progressed, "Python worker did not progress during native reconstruction"
    assert not thread.is_alive()
    assert result["nodes_added"] == list(range(30_000))
    assert result["edge_endpoint_changes"] == []
    assert result["from_state"] == 0 and result["to_state"] == state
