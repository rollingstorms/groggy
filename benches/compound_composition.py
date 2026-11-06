"""Bounded release receipt for one accepted component-collapse collection.

Run with builds quiet. Construction is excluded; measured collapse includes all
aggregation, source retention, checked topology effects and acceptance work.
"""
import argparse
import json
import statistics
import time
from pathlib import Path

from groggy import Graph


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    args = parser.parse_args()
    results = []
    for components in [10, 100, 500]:
        samples = []
        for _ in range(3):
            graph = Graph()
            graph.add_nodes([{"x": node} for node in range(2 * components)])
            graph.add_edges([(2 * i, 2 * i + 1) for i in range(components)])
            selected = graph.connected_components()
            head = graph.current_state_id
            wall = time.perf_counter_ns()
            cpu = time.process_time_ns()
            metas = selected.collapse(node_aggs={"total": ("sum", "x")}, node_strategy="collapse")
            cpu = time.process_time_ns() - cpu
            wall = time.perf_counter_ns() - wall
            assert len(metas) == components
            assert graph.node_count() == components and graph.edge_count() == 0
            assert sum(graph.get_node_attr(meta.id, "total") for meta in metas) == sum(range(2 * components))
            samples.append({"wall_ns": wall, "cpu_ns": cpu, "accepted_states": graph.current_state_id - head})
        results.append({"components": components, "source_nodes": components * 2, "source_edges": components, "samples": samples})
        print(components, statistics.median(row["wall_ns"] for row in samples) / 1e6, flush=True)
    args.output.write_text(json.dumps({"label": args.label, "fixture": "disconnected pairs, one integer property per node, destructive component collection collapse with sum; release in-tree extension; construction excluded", "cases": results}, indent=2) + "\n")


if __name__ == "__main__":
    main()
