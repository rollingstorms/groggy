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
    different batch. Pass ``schema=...`` to ``Graph.apply_batch`` to validate
    the complete proposed graph before allocation or publication. This policy
    applies to that call only.
    """

    def __init__(self, *, effective_at: int | None = None) -> None:
        self._effective_at = effective_at
        self._change_times = []
        self._owner = object()
        self._add_nodes = 0
        self._add_edges: list[tuple[Any, Any]] = []
        self._node_attrs: dict[str, list[tuple[Any, Any]]] = {}
        self._edge_attrs: dict[str, list[tuple[Any, Any]]] = {}
        self._unset_node_attrs: dict[str, list[Any]] = {}
        self._unset_edge_attrs: dict[str, list[Any]] = {}
        self._remove_nodes: list[Any] = []
        self._remove_edges: list[Any] = []
        self._node_external_ids = []
        self._edge_external_ids = []

    def _ref(self, value: Any, kind: type) -> Any:
        if isinstance(value, kind):
            if value._owner is not self._owner:
                raise ValueError("Batch handle belongs to another GraphBatch")
            return {"added": value.position}
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            return value
        raise TypeError(f"Expected a nonnegative entity ID or {kind.__name__} handle")

    def add_node(self, *, effective_at: int | None = None, **attrs: Any) -> AddedNode:
        node = AddedNode(self._owner, self._add_nodes)
        self._add_nodes += 1
        self._time("node_added", node, effective_at)
        for name, value in attrs.items():
            self.set_node_attr(node, name, value)
        return node

    def add_edge(self, source: int | AddedNode, target: int | AddedNode, *, effective_at: int | None = None, **attrs: Any) -> AddedEdge:
        edge = AddedEdge(self._owner, len(self._add_edges))
        self._add_edges.append((source, target))
        self._time("edge_added", edge, effective_at)
        for name, value in attrs.items():
            self.set_edge_attr(edge, name, value)
        return edge

    def set_node_attr(self, node: int | AddedNode, name: str, value: Any, *, effective_at: int | None = None) -> None:
        self._node_attrs.setdefault(name, []).append((node, value))
        self._time("node_attr", node, effective_at, name)

    def set_edge_attr(self, edge: int | AddedEdge, name: str, value: Any, *, effective_at: int | None = None) -> None:
        self._edge_attrs.setdefault(name, []).append((edge, value))
        self._time("edge_attr", edge, effective_at, name)

    def unset_node_attr(self, node: int | AddedNode, name: str, *, effective_at: int | None = None) -> None:
        """Make a property absent, retaining its history. None assignments remain null."""
        self._unset_node_attrs.setdefault(name, []).append(node)
        self._time("node_attr", node, effective_at, name)

    def unset_edge_attr(self, edge: int | AddedEdge, name: str, *, effective_at: int | None = None) -> None:
        self._unset_edge_attrs.setdefault(name, []).append(edge)
        self._time("edge_attr", edge, effective_at, name)

    def remove_node(self, node: int | AddedNode, *, effective_at: int | None = None) -> None:
        self._remove_nodes.append(node)
        self._time("node_removed", node, effective_at)

    def remove_edge(self, edge: int | AddedEdge, *, effective_at: int | None = None) -> None:
        self._remove_edges.append(edge)
        self._time("edge_removed", edge, effective_at)

    def set_node_external_id(self, node, external_id, *, effective_at=None):
        self._node_external_ids.append((node, external_id))
        self._time("node_attr", node, effective_at, "external_id")

    def set_edge_external_id(self, edge, external_id, *, effective_at=None):
        self._edge_external_ids.append((edge, external_id))
        self._time("edge_attr", edge, effective_at, "external_id")

    def _time(self, kind, entity, effective_at, name=None):
        if effective_at is not None:
            self._change_times.append((kind, entity, name, effective_at))

    def to_dict(self) -> dict[str, Any]:
        """Validate handle ownership and produce the native batch description."""
        return {
            "effective_at": self._effective_at,
            "change_times": [
                {"kind": kind, "entity": self._ref(entity, AddedNode if kind.startswith("node") else AddedEdge),
                 "name": name, "effective_at": at}
                for kind, entity, name, at in self._change_times
            ],
            "node_external_ids": [(self._ref(node, AddedNode), name) for node, name in self._node_external_ids],
            "edge_external_ids": [(self._ref(edge, AddedEdge), name) for edge, name in self._edge_external_ids],
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
            "unset_node_attrs": {
                name: [self._ref(node, AddedNode) for node in nodes]
                for name, nodes in self._unset_node_attrs.items()
            },
            "unset_edge_attrs": {
                name: [self._ref(edge, AddedEdge) for edge in edges]
                for name, edges in self._unset_edge_attrs.items()
            },
            "remove_nodes": [self._ref(node, AddedNode) for node in self._remove_nodes],
            "remove_edges": [self._ref(edge, AddedEdge) for edge in self._remove_edges],
        }
