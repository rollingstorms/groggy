#!/usr/bin/env python3
"""
Generate an SVG diagram from the meta API graph bundle (core types only).

This script uses Groggy to load the bundle and export nodes/edges to JSON,
then renders a simple SVG layout of core types only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import groggy


def _load_table_json(table, output_path: Path) -> Dict:
    table.to_json(str(output_path))
    return json.loads(output_path.read_text())


def _layout_nodes(nodes: List[Dict]) -> Dict[int, Tuple[float, float]]:
    core_nodes = [n for n in nodes if n.get("category") == "core"]

    positions: Dict[int, Tuple[float, float]] = {}
    center_x, center_y = 1000.0, 1000.0
    core_radius = 520.0

    for idx, node in enumerate(core_nodes):
        angle = 2 * math.pi * idx / max(len(core_nodes), 1)
        x = center_x + core_radius * math.cos(angle)
        y = center_y + core_radius * math.sin(angle)
        positions[node["node_id"]] = (x, y)

    return positions


def _render_svg(nodes: List[Dict], edges: List[Dict], positions: Dict[int, Tuple[float, float]]) -> str:
    width, height = 2000, 2000
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" ',
        'viewBox="0 0 2000 2000" style="background:#0f1115">',
        '<defs>',
        '<style>',
        '.edge { stroke: #9aa4b2; stroke-opacity: 0.15; stroke-width: 1; }',
        '.edge-core { stroke: #8ecae6; stroke-opacity: 0.22; stroke-width: 1.2; }',
        '.node-core { fill: #2a6f97; stroke: #9bd4ff; stroke-width: 1; }',
        '.node-inferred { fill: #1b2630; stroke: #5c6f82; stroke-width: 1; }',
        '.label { fill: #e6edf3; font: 10px ui-sans-serif, system-ui, -apple-system, sans-serif; }',
        '</style>',
        '</defs>',
    ]

    node_lookup = {n["node_id"]: n for n in nodes}

    for edge in edges:
        src = edge.get("source")
        tgt = edge.get("target")
        if src not in positions or tgt not in positions:
            continue
        x1, y1 = positions[src]
        x2, y2 = positions[tgt]
        lines.append(f'<line class="edge-core" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" />')

    for node in nodes:
        node_id = node["node_id"]
        x, y = positions[node_id]
        klass = "node-core"
        label = node.get("type_name", "Unknown")
        lines.append(f'<circle class="{klass}" cx="{x:.2f}" cy="{y:.2f}" r="8" />')
        lines.append(f'<text class="label" x="{x + 10:.2f}" y="{y + 3:.2f}">{label}</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> int:
    bundle_dir = Path("notes/meta_api_discovery/meta_api_graph_bundle")
    output_svg = Path("docs/img/connected-views-core.svg")
    output_svg.parent.mkdir(parents=True, exist_ok=True)

    graph_table = groggy.GraphTable.load_bundle(str(bundle_dir))
    nodes_json = _load_table_json(graph_table.nodes, Path("/tmp/meta_api_nodes.json"))
    edges_json = _load_table_json(graph_table.edges, Path("/tmp/meta_api_edges.json"))

    nodes = [n for n in nodes_json["data"] if n.get("category") == "core"]
    core_ids = {n["node_id"] for n in nodes}
    edges = [e for e in edges_json["data"] if e.get("source") in core_ids and e.get("target") in core_ids]
    connected_ids = {e.get("source") for e in edges} | {e.get("target") for e in edges}
    nodes = [n for n in nodes if n["node_id"] in connected_ids]
    positions = _layout_nodes(nodes)
    svg = _render_svg(nodes, edges, positions)

    output_svg.write_text(svg)
    print(f"Wrote SVG to {output_svg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
