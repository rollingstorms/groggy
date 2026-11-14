"""Isolate cyclic GC of thread-bound Graph versus detached immutable GraphView.

Run with the in-tree extension. Each case uses a fresh interpreter to isolate GC
from other retained Python objects. Final release is safe on any thread; live
Graph access remains confined to its creating thread.
"""
import json
import subprocess
import sys

CASE = r'''
import gc, json, sys, threading
from groggy import Graph
errors = []
sys.unraisablehook = lambda event: errors.append(str(event.exc_value))
gc.disable()
graph = Graph()
graph.add_nodes(1)
value = graph.read_view() if CASE == "view_worker" else graph
del graph
cycle = [value]
cycle.append(cycle)
del value, cycle
if CASE == "graph_owner":
    gc.collect()
else:
    worker = threading.Thread(target=gc.collect)
    worker.start()
    worker.join()
print(json.dumps(errors))
'''


def main():
    for name in ("graph_owner", "view_worker", "graph_worker"):
        result = subprocess.run(
            [sys.executable, "-c", "CASE = " + repr(name) + "\n" + CASE],
            check=True, capture_output=True, text=True,
        )
        errors = json.loads(result.stdout)
        assert errors == [], errors
        print(name + ": " + json.dumps(errors))


if __name__ == "__main__":
    main()
