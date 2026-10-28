"""Bounded frontend/runtime receipts for optimizer ordering repairs.

Run with the existing in-tree release extension: python benches/builder_order.py
No external graph libraries or packages are used. Graph creation is excluded.
"""
import json
import statistics
from time import perf_counter

from groggy import Graph
from groggy.builder import AlgorithmBuilder


def make_builder(ordered):
    builder = AlgorithmBuilder("ordered effects" if ordered else "pure terminal output")
    values = builder.init_nodes(default=3)
    if ordered:
        builder.attach_as("score", values)
        builder.attach_as("copy", builder.load_attr("score"))
        builder.attach_as("score", builder.init_nodes(default=4))
    else:
        builder.attach_as("result", values * 2 + 1)
    return builder


def main():
    cases = []
    for count in [1_000, 10_000]:
        for ordered in [False, True]:
            samples = {False: [], True: []}
            for iteration in range(5):
                for optimize in ([False, True] if iteration % 2 == 0 else [True, False]):
                    graph = Graph()
                    graph.add_nodes([{"score": 7} for _ in range(count)])
                    builder = make_builder(ordered)
                    start = perf_counter()
                    algorithm = builder.build(optimize=optimize)
                    compile_seconds = perf_counter() - start
                    start = perf_counter()
                    graph.view().apply(algorithm, persist=True)
                    execute_seconds = perf_counter() - start
                    for node in [0, count - 1]:
                        if ordered:
                            assert graph.get_node_attr(node, "score") == 4
                            assert graph.get_node_attr(node, "copy") == 3
                        else:
                            assert graph.get_node_attr(node, "result") == 7
                    samples[optimize].append({
                        "compile_seconds": compile_seconds,
                        "execute_seconds": execute_seconds,
                        "compiled_steps": len(algorithm.steps),
                    })
            cases.append({"nodes": count, "ordered_effects": ordered,
                          "samples": {str(k): v for k, v in samples.items()},
                          "medians": {str(k): {metric: statistics.median(row[metric] for row in v)
                                              for metric in ["compile_seconds", "execute_seconds"]}
                                      for k, v in samples.items()}})
    print(json.dumps({"extension": "in-tree release", "fixture": "builder order correctness receipts",
                      "ordering": "alternating optimize=False/True each sample",
                      "cases": cases}, indent=2))


if __name__ == "__main__":
    main()
