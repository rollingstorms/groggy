"""Run against an explicitly selected in-tree extension via PYTHONPATH."""
import json
import resource
import statistics
import time

from groggy import Graph
from groggy.algorithms.base import RustAlgorithmHandle


def profile():
    rows = []
    for cliques in (16, 128):
        graph = Graph()
        nodes = [graph.add_node() for _ in range(cliques * 3)]
        for c in range(cliques):
            a, b, d = nodes[3 * c:3 * c + 3]
            graph.add_edge(a, b)
            graph.add_edge(b, d)
            graph.add_edge(d, a)
            graph.add_edge(d, nodes[(3 * c + 3) % len(nodes)])
        view = graph.induced_subgraph(nodes)
        for phases in (1, 10):
            samples = []
            for _ in range(5):
                algo = RustAlgorithmHandle("community.louvain", {
                    "max_iter": 100, "max_phases": phases,
                    "resolution": 1.0, "output_attr": "community",
                })
                start = time.perf_counter()
                view.apply(algo)
                samples.append(time.perf_counter() - start)
            rows.append({
                "nodes": len(nodes), "edges": cliques * 4,
                "max_phases": phases, "seconds": samples,
                "median_seconds": statistics.median(samples),
                "communities": len({graph.get_node_attr(n, "community") for n in nodes}),
            })
    return {"profile": "debug in-tree build, five repetitions, no speedup guarantee",
            "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rows": rows}


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
