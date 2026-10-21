"""Bounded debug profile; choose the in-tree extension with PYTHONPATH."""
import json
import resource
import statistics
import time

from groggy import Graph, algorithms


def profile():
    rows = []
    for cliques in (16, 128):
        graph = Graph()
        nodes = [graph.add_node() for _ in range(cliques * 3)]
        for c in range(cliques):
            a, b, d = nodes[3 * c:3 * c + 3]
            for u, v in [(a, b), (b, d), (d, a), (d, nodes[(3 * c + 3) % len(nodes)])]:
                graph.add_edge(u, v)
        view = graph.induced_subgraph(nodes)
        for phases in (1, 10):
            samples = []
            for _ in range(5):
                start = time.perf_counter()
                view.apply(algorithms.community.leiden(
                    max_iter=100, max_phases=phases, seed=42,
                ))
                samples.append(time.perf_counter() - start)
            rows.append({"nodes": len(nodes), "edges": cliques * 4,
                         "max_phases": phases, "seconds": samples,
                         "median_seconds": statistics.median(samples),
                         "communities": len({graph.get_node_attr(n, "community") for n in nodes})})
    return {"profile": "debug in-tree build, five repetitions, seed 42; algorithms differ before/after",
            "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rows": rows}


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
