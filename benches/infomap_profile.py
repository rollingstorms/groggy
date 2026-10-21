"""Debug profile for the corrected two-level map-equation implementation."""
import json
import resource
import statistics
import time
from groggy import Graph
from groggy.algorithms.base import RustAlgorithmHandle


def profile():
    rows = []
    for cliques in (4, 16):
        graph = Graph()
        nodes = [graph.add_node() for _ in range(cliques * 3)]
        for c in range(cliques):
            a, b, d = nodes[3 * c:3 * c + 3]
            for u, v in [(a, b), (b, d), (d, a), (d, nodes[(3 * c + 3) % len(nodes)])]:
                graph.add_edge(u, v)
        view = graph.induced_subgraph(nodes)
        for trials in (1, 4):
            samples = []
            for _ in range(5):
                algo = RustAlgorithmHandle("community.infomap", {
                    "teleportation": 0.15, "num_trials": trials,
                    "max_iter": 30, "seed": 42, "output_attr": "community",
                })
                start = time.perf_counter()
                view.apply(algo)
                samples.append(time.perf_counter() - start)
            rows.append({"nodes": len(nodes), "edges": cliques * 4,
                         "num_trials": trials, "seconds": samples,
                         "median_seconds": statistics.median(samples),
                         "communities": len({graph.get_node_attr(n, "community") for n in nodes})})
    return {"profile": "debug in-tree build, five repetitions; before/after use different objectives",
            "max_rss_platform_units": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "rows": rows}


if __name__ == "__main__":
    print(json.dumps(profile(), indent=2))
