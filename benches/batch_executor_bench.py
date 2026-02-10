#!/usr/bin/env python3
"""
Benchmark Batch Executor speedups versus step-by-step fallback.

Usage:
  python benches/batch_executor_bench.py --nodes 10000 --edges 50000 --iters 50 --repeats 3
"""

import argparse
import os
import time

from pathlib import Path

repo_root = Path(__file__).parent.parent
python_pkg = repo_root / "python-groggy" / "python"
import sys
sys.path.insert(0, str(python_pkg))
sys.path.insert(0, str(repo_root))

import groggy as gr
from groggy.builder import AlgorithmBuilder


def build_iterative_algo(builder: AlgorithmBuilder, iterations: int) -> AlgorithmBuilder:
    ranks = builder.init_nodes(default=1.0)

    with builder.iterate(iterations):
        neighbor_sum = builder.graph_ops.neighbor_agg(ranks, agg="sum")
        damped = builder.core.mul(neighbor_sum, 0.85)
        ranks = builder.core.add(damped, 0.15)
        ranks = builder.var("ranks", ranks)

    builder.attach_as("rank", ranks)
    return builder


def build_algo(disable_batch: bool, iterations: int):
    if disable_batch:
        os.environ["GROGGY_DISABLE_BATCH"] = "1"
    else:
        os.environ.pop("GROGGY_DISABLE_BATCH", None)

    builder = AlgorithmBuilder("batch_bench")
    return build_iterative_algo(builder, iterations).build()


def time_apply(graph: gr.Graph, algo, repeats: int) -> float:
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        graph.view().apply(algo)
        timings.append(time.perf_counter() - start)
    return sum(timings) / len(timings)


def build_graph(nodes: int, edges: int) -> gr.Graph:
    graph = gr.Graph(directed=True)
    for i in range(nodes):
        graph.add_node(i)

    import random

    random.seed(42)
    edges_added = 0
    while edges_added < edges:
        src = random.randint(0, nodes - 1)
        dst = random.randint(0, nodes - 1)
        if src == dst:
            continue
        graph.add_edge(src, dst)
        edges_added += 1
    return graph


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nodes", type=int, default=10000)
    parser.add_argument("--edges", type=int, default=50000)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    graph = build_graph(args.nodes, args.edges)

    algo_batch = build_algo(disable_batch=False, iterations=args.iters)
    algo_fallback = build_algo(disable_batch=True, iterations=args.iters)

    batch_time = time_apply(graph, algo_batch, args.repeats)
    fallback_time = time_apply(graph, algo_fallback, args.repeats)

    speedup = fallback_time / batch_time if batch_time > 0 else float("inf")

    print("Batch Executor Benchmark")
    print(f"nodes={args.nodes} edges={args.edges} iters={args.iters} repeats={args.repeats}")
    print(f"batch_avg_s={batch_time:.4f}")
    print(f"fallback_avg_s={fallback_time:.4f}")
    print(f"speedup={speedup:.2f}x")
    print("notes=Use GROGGY_DISABLE_BATCH=1 to force fallback in other scripts")


if __name__ == "__main__":
    main()
