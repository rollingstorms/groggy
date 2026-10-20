"""A small builder for one atomic graph write."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AddedNode:
    _owner: object
    position: int


@dataclass(frozen=True)
class AddedEdge:
    _owner: object
    position: int


class GraphBatch:
    """Stage nodes, edges, properties, and removals for ``Graph.apply_batch``.

    Existing entities use their integer IDs. ``add_node`` and ``add_edge`` return
    handles for entities created by this batch. A handle cannot be used in a
    different batch.
    """

    def __init__(self) -> None:
        self._owner = object()
        self._add_nodes = 0
        self._add_edges: list[tuple[Any, Any]] = []
        self._node_attrs: dict[str, list[tuple[Any, Any]]] = {}
        self._edge_attrs: dict[str, list[tuple[Any, Any]]] = {}
        self._remove_nodes: list[Any] = []
        self._remove_edges: list[Any] = []

    def _ref(self, value: Any, kind: type) -> Any:
        if isinstance(value, kind):
            if value._owner is not self._owner:
                raise ValueError("Batch handle belongs to another GraphBatch")
            return {"added": value.position}
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        raise TypeError(f"Expected a nonnegative entity ID or {kind.__name__} handle")

    def add_node(self, **attrs: Any) -> AddedNode:
        node = AddedNode(self._owner, self._add_nodes)
        self._add_nodes += 1
        for name, value in attrs.items():
            self.set_node_attr(node, name, value)
        return node

    def add_edge(self, source: int | AddedNode, target: int | AddedNode, **attrs: Any) -> AddedEdge:
        edge = AddedEdge(self._owner, len(self._add_edges))
        self._add_edges.append((source, target))
        for name, value in attrs.items():
            self.set_edge_attr(edge, name, value)
        return edge

    def set_node_attr(self, node: int | AddedNode, name: str, value: Any) -> None:
        self._node_attrs.setdefault(name, []).append((node, value))

    def set_edge_attr(self, edge: int | AddedEdge, name: str, value: Any) -> None:
        self._edge_attrs.setdefault(name, []).append((edge, value))

    def remove_node(self, node: int | AddedNode) -> None:
        self._remove_nodes.append(node)

    def remove_edge(self, edge: int | AddedEdge) -> None:
        self._remove_edges.append(edge)

    def to_dict(self) -> dict[str, Any]:
        """Validate handle ownership and produce the native batch description."""
        return {
            "add_nodes": self._add_nodes,
            "add_edges": [
                (self._ref(source, AddedNode), self._ref(target, AddedNode))
                for source, target in self._add_edges
            ],
            "node_attrs": {
                name: [(self._ref(node, AddedNode), value) for node, value in values]
                for name, values in self._node_attrs.items()
            },
            "edge_attrs": {
                name: [(self._ref(edge, AddedEdge), value) for edge, value in values]
                for name, values in self._edge_attrs.items()
            },
            "remove_nodes": [self._ref(node, AddedNode) for node in self._remove_nodes],
            "remove_edges": [self._ref(edge, AddedEdge) for edge in self._remove_edges],
        }
