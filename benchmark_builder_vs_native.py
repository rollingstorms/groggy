"""
Lightweight benchmark comparing native PageRank to builder implementations.

Run with:
    python benchmark_builder_vs_native.py --nodes 50000 200000 --variant loop async
"""
from __future__ import annotations

import argparse
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

from groggy import Graph, print_profile
from groggy.algorithms import centrality, community
from groggy.builder import algorithm


@algorithm("pagerank_loop")
def pagerank_loop(sG, damping: float = 0.85, max_iter: int = 100):
    """Batch-optimized Jacobi PageRank."""
    builder = sG.builder

    ranks = builder.var("ranks", sG.nodes(1.0))
    degrees = builder.graph_ops.degree()
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)

    node_count = sG.N
    one_minus = builder.core.constant(1 - damping)
    teleport_numer = builder.core.broadcast_scalar(one_minus, ranks)
    node_count_map = builder.core.broadcast_scalar(node_count, ranks)
    teleport = builder.core.div(teleport_numer, node_count_map)

    # Message-pass LPA also uses iterate() to bypass batch compilation until
    # execution blocks participate in batch plans.
    with builder.iterate(max_iter):
        contrib = ranks * inv_degrees
        neighbor_sum = sG @ contrib
        ranks = builder.var("ranks", damping * neighbor_sum + teleport)

    return ranks.normalize()


@algorithm("pagerank_async")
def pagerank_async(sG, damping: float = 0.85, max_iter: int = 100):
    """Gauss-Seidel PageRank using the message_pass execution block."""
    builder = sG.builder

    ranks = builder.var("ranks", sG.nodes(1.0))
    degrees = builder.graph_ops.degree()
    inv_degrees = builder.core.recip(degrees, epsilon=1e-9)

    node_count = sG.N
    one_minus = builder.core.constant(1 - damping)
    teleport_numer = builder.core.broadcast_scalar(one_minus, ranks)
    node_count_map = builder.core.broadcast_scalar(node_count, ranks)
    teleport = builder.core.div(teleport_numer, node_count_map)

    # Message-pass blocks aren't batch-compatible yet, so use legacy iterate() to avoid
    # broken batch plans while we keep the async semantics.
    with builder.iterate(max_iter):
        contrib = ranks * inv_degrees
        with builder.message_pass(
            target=ranks,
            include_self=False,
            ordered=True,
            name="pagerank_async",
        ) as mp:
            neighbor_sum = builder.graph_ops.neighbor_agg(contrib, "sum")
            updated = damping * neighbor_sum + teleport
            mp.apply(updated)

        # Make loop-carried dependency explicit
    ranks = builder.var("ranks", ranks)

    return ranks.normalize()


@algorithm("lpa_loop")
def lpa_loop(sG, max_iter: int = 10):
    """Label Propagation using message_pass for async updates."""
    builder = sG.builder

    labels = builder.var("labels", sG.nodes(unique=True))

    with builder.iterate(max_iter):
        with builder.message_pass(
            target=labels,
            include_self=True,
            ordered=True,
            name="lpa_async",
        ) as mp:
            neighbor_labels = mp.pull(labels)
            updated = builder.core.mode(neighbor_labels, tie_break="lowest")
            mp.apply(updated)

        labels = builder.var("labels", labels)

    return labels


@dataclass
class RunResult:
    time: float
    total: float
    values: Dict[int, float]


def create_random_graph(num_nodes: int, avg_degree: int, seed: int) -> Graph:
    """Create an undirected random graph with roughly num_nodes*avg_degree/2 edges."""
    random.seed(seed)
    graph = Graph()

    nodes = [graph.add_node() for _ in range(num_nodes)]
    edges = set()
    target_edges = num_nodes * avg_degree // 2

    while len(edges) < target_edges:
        i = random.randrange(num_nodes)
        j = random.randrange(num_nodes)
        if i == j:
            continue
        a, b = sorted((i, j))
        if (a, b) in edges:
            continue
        edges.add((a, b))

    graph.add_edges([(nodes[a], nodes[b]) for (a, b) in edges])
    return graph


def extract_values(result_graph, attr: str) -> Dict[int, float]:
    return {node.id: getattr(node, attr) for node in result_graph.nodes}


def run_native_pagerank(sg, damping: float, max_iter: int) -> Tuple[RunResult, Dict]:
    start = time.perf_counter()
    result, profile = sg.apply(
        centrality.pagerank(
            max_iter=max_iter,
            damping=damping,
            output_attr="pagerank_native",
        ),
        persist=True,
        return_profile=True,
    )
    elapsed = time.perf_counter() - start
    values = extract_values(result, "pagerank_native")
    return RunResult(elapsed, sum(values.values()), values), profile


def run_builder_pagerank(
    sg,
    variant: str,
    damping: float,
    max_iter: int,
) -> Tuple[RunResult, Dict]:
    algo_factory = {
        "loop": pagerank_loop,
        "async": pagerank_async,
    }[variant]
    algo = algo_factory(damping=damping, max_iter=max_iter)

    strip_execution_block_batch_plans(algo)
    start = time.perf_counter()
    result, profile = sg.apply(algo, return_profile=True)
    elapsed = time.perf_counter() - start
    attr_name = algo_factory.__name__
    values = extract_values(result, attr_name)
    return RunResult(elapsed, sum(values.values()), values), profile


def run_native_lpa(sg, max_iter: int) -> Tuple[RunResult, Dict]:
    start = time.perf_counter()
    result, profile = sg.apply(
        community.lpa(max_iter=max_iter, output_attr="lpa_native"),
        persist=True,
        return_profile=True,
    )
    elapsed = time.perf_counter() - start
    values = extract_values(result, "lpa_native")
    total = float(sum(values.values()))
    return RunResult(elapsed, total, values), profile


def run_builder_lpa(sg, max_iter: int) -> Tuple[RunResult, Dict]:
    algo = lpa_loop(max_iter=max_iter)
    strip_execution_block_batch_plans(algo)
    start = time.perf_counter()
    result, profile = sg.apply(algo, return_profile=True)
    elapsed = time.perf_counter() - start
    attr_name = lpa_loop.__name__
    values = extract_values(result, attr_name)
    total = float(sum(values.values()))
    return RunResult(elapsed, total, values), profile


def summarize_diff(native: RunResult, builder: RunResult) -> Tuple[float, float]:
    diffs = [
        abs(builder.values.get(node_id, 0.0) - native_val)
        for node_id, native_val in native.values.items()
    ]
    if not diffs:
        return 0.0, 0.0
    avg = sum(diffs) / len(diffs)
    return avg, max(diffs)


def sample_nodes(values: Dict[int, float], count: int = 5) -> List[Tuple[int, float]]:
    node_ids = sorted(values.keys())[:count]
    return [(node_id, values[node_id]) for node_id in node_ids]


def summarize_lpa_match(native: RunResult, builder: RunResult) -> Tuple[int, int, float]:
    total = len(native.values)
    matches = sum(
        1 for node_id, native_label in native.values.items()
        if builder.values.get(node_id) == native_label
    )
    ratio = matches / total if total else 1.0
    return matches, total, ratio


def top_communities(values: Dict[int, float], limit: int = 5) -> List[Tuple[float, int]]:
    counts = Counter(values.values())
    return counts.most_common(limit)


def benchmark_pagerank(
    graph: Graph,
    label: str,
    variant: str,
    damping: float,
    max_iter: int,
    show_profile: bool,
):
    print(f"\n{'='*60}")
    print(f"PageRank Benchmark ({label}) - variant: {variant}")
    print(f"{'='*60}")

    sg = graph.view()

    native, native_profile = run_native_pagerank(sg, damping, max_iter)
    builder, builder_profile = run_builder_pagerank(sg, variant, damping, max_iter)

    avg_diff, max_diff = summarize_diff(native, builder)

    print(f"Native time : {native.time:.3f}s  (total={native.total:.6f})")
    print(f"Builder time: {builder.time:.3f}s  (total={builder.total:.6f})")
    print(f"Speed ratio : {builder.time / native.time:.2f}x")
    print(f"Avg diff    : {avg_diff:.8f}")
    print(f"Max diff    : {max_diff:.8f}")

    native_samples = sample_nodes(native.values)
    builder_samples = sample_nodes(builder.values)
    print("\nSample node ranks (first 5 IDs):")
    for (nid, native_val), (_, builder_val) in zip(native_samples, builder_samples):
        print(f"  Node {nid:>5}: native={native_val:.6f}  builder={builder_val:.6f}")

    if show_profile:
        print("\nNative profile:")
        print_profile(native_profile, show_steps=True, show_details=False)
        print("\nBuilder profile:")
        print_profile(builder_profile, show_steps=True, show_details=False)


def benchmark_lpa(
    graph: Graph,
    label: str,
    max_iter: int,
    show_profile: bool,
):
    print(f"\n{'='*60}")
    print(f"LPA Benchmark ({label})")
    print(f"{'='*60}")

    sg = graph.view()

    native, native_profile = run_native_lpa(sg, max_iter=max_iter)
    builder, builder_profile = run_builder_lpa(sg, max_iter=max_iter)

    matches, total_nodes, ratio = summarize_lpa_match(native, builder)

    print(f"Native time : {native.time:.3f}s")
    print(f"Builder time: {builder.time:.3f}s")
    print(f"Speed ratio : {builder.time / native.time:.2f}x")
    print(f"Match ratio : {ratio:.2%} ({matches}/{total_nodes})")

    native_comm = top_communities(native.values)
    builder_comm = top_communities(builder.values)

    print("\nNative top communities:")
    for comm, size in native_comm:
        print(f"  Label {comm}: {size} nodes")

    print("\nBuilder top communities:")
    for comm, size in builder_comm:
        print(f"  Label {comm}: {size} nodes")

    if show_profile:
        print("\nNative profile:")
        print_profile(native_profile, show_steps=True, show_details=False)
        print("\nBuilder profile:")
        print_profile(builder_profile, show_steps=True, show_details=False)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nodes",
        type=int,
        nargs="+",
        default=[50000],
        help="One or more node counts to benchmark (default: 50000)",
    )
    parser.add_argument(
        "--avg-degree",
        type=int,
        default=10,
        help="Average degree to target when generating graphs",
    )
    parser.add_argument(
        "--variant",
        choices=["loop", "async"],
        default="loop",
        help="Builder variant to benchmark",
    )
    parser.add_argument(
        "--damping",
        type=float,
        default=0.85,
        help="PageRank damping factor",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="Maximum PageRank iterations",
    )
    parser.add_argument(
        "--lpa-iterations",
        type=int,
        default=10,
        help="Maximum LPA iterations",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for random graph generation",
    )
    parser.add_argument(
        "--algo",
        choices=["pagerank", "lpa"],
        nargs="+",
        default=["pagerank"],
        help="Benchmark selection (can list multiple)",
    )
    parser.add_argument(
        "--show-profile",
        action="store_true",
        help="Print detailed profiler output for both runs",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None):
    args = parse_args(argv)
    selected = set(args.algo)

    for node_count in args.nodes:
        graph = create_random_graph(node_count, args.avg_degree, args.seed)
        label = f"{node_count} nodes"
        if "pagerank" in selected:
            benchmark_pagerank(
                graph=graph,
                label=label,
                variant=args.variant,
                damping=args.damping,
                max_iter=args.iterations,
                show_profile=args.show_profile,
            )
        if "lpa" in selected:
            benchmark_lpa(
                graph=graph,
                label=label,
                max_iter=args.lpa_iterations,
                show_profile=args.show_profile,
            )


def strip_execution_block_batch_plans(algorithm_obj) -> None:
    """Remove batch plans from loops that contain execution blocks."""
    for step in getattr(algorithm_obj, "steps", []):
        if step.get("type") != "iter.loop":
            continue
        body = step.get("body", [])
        if any(item.get("type") == "core.execution_block" for item in body):
            step.pop("batch_plan", None)
            step.pop("_batch_optimized", None)


if __name__ == "__main__":
    main()
