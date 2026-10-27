"""Checked direct-write convenience methods over the native batch coordinator."""
from copy import deepcopy

from .batch import GraphBatch


class GraphWriter:
    """Each mutation stages one native batch and publishes exactly one state.

    Explicit checkpoints label the selected state without publishing a new one.
    Explicit restore_state selects retained state and retargets this session.

    Keep the owning graph for reads. Another writer advancing its head makes this
    session stale until explicitly refreshed. A private copy of the schema is
    applied on every call; legacy writes are rejected by the native dirty guard.
    """

    def __init__(self, graph, author="writer", schema=None):
        self._graph = graph
        self._author = author
        self._schema = deepcopy(schema)
        self._state_id = graph.current_state_id

    @property
    def state_id(self):
        return self._state_id

    def refresh(self):
        """Explicitly target the graph's latest published head on the next call."""
        self._state_id = self._graph.current_state_id
        return self._state_id

    def checkpoint(self, message="checkpoint"):
        """Label this published state without allocating another mutation version."""
        return self._graph.checkpoint(
            message, self._author, expected_state=self._state_id, schema=self._schema
        )

    def apply_batch(self, batch, message="write"):
        result = self._graph.apply_batch(
            batch, self._state_id, message, self._author, schema=self._schema
        )
        self._state_id = result["state_id"]
        return result

    def restore_state(self, state):
        """Select a retained published state; preserve history until explicit collection."""
        restored = self._graph.restore_state(
            state, expected_state=self._state_id, schema=self._schema
        )
        self._state_id = restored
        return restored

    def add_node(self, **attrs):
        batch = GraphBatch()
        batch.add_node(**attrs)
        return self.apply_batch(batch, "add_node")["added_nodes"][0]

    def add_nodes(self, count, *, effective_at=None):
        batch = GraphBatch(effective_at=effective_at)
        # One count crosses FFI; the Rust coordinator uses its bulk allocator.
        batch._add_nodes = count
        return self.apply_batch(batch, "add_nodes")["added_nodes"]

    def add_nodes_with_attrs(self, rows, *, effective_at=None):
        batch = GraphBatch(effective_at=effective_at)
        for row in rows:
            batch.add_node(**row)
        return self.apply_batch(batch, "add_nodes_with_attrs")["added_nodes"]

    def add_edge(self, source, target, **attrs):
        batch = GraphBatch()
        batch.add_edge(source, target, **attrs)
        return self.apply_batch(batch, "add_edge")["added_edges"][0]

    def add_edges(self, rows, *, effective_at=None):
        """Rows contain (source, target) or (source, target, properties)."""
        batch = GraphBatch(effective_at=effective_at)
        for row in rows:
            if len(row) == 2:
                batch.add_edge(row[0], row[1])
            elif len(row) == 3:
                batch.add_edge(row[0], row[1], **row[2])
            else:
                raise ValueError("Edge rows require two endpoints and optional properties")
        return self.apply_batch(batch, "add_edges")["added_edges"]

    def set_edge_endpoints(self, edge, source, target, *, effective_at=None):
        batch = GraphBatch()
        batch.set_edge_endpoints(edge, source, target, effective_at=effective_at)
        return self.apply_batch(batch, "set_edge_endpoints")["state_id"]

    def set_edge_endpoints_bulk(self, rows, *, effective_at=None):
        batch = GraphBatch(effective_at=effective_at)
        for edge, source, target in rows:
            batch.set_edge_endpoints(edge, source, target)
        return self.apply_batch(batch, "set_edge_endpoints")["state_id"]

    def set_node_attr(self, node, name, value, *, effective_at=None):
        batch = GraphBatch()
        batch.set_node_attr(node, name, value, effective_at=effective_at)
        return self.apply_batch(batch, "set_node_attr")["state_id"]

    def set_edge_attr(self, edge, name, value, *, effective_at=None):
        batch = GraphBatch()
        batch.set_edge_attr(edge, name, value, effective_at=effective_at)
        return self.apply_batch(batch, "set_edge_attr")["state_id"]

    def set_node_attrs(self, columns, *, effective_at=None):
        """Columns map names to {node: value} mappings or (node, value) pairs."""
        batch = GraphBatch(effective_at=effective_at)
        for name, column in columns.items():
            for node, value in column.items() if isinstance(column, dict) else column:
                batch.set_node_attr(node, name, value)
        return self.apply_batch(batch, "set_node_attrs")["state_id"]

    def set_edge_attrs(self, columns, *, effective_at=None):
        batch = GraphBatch(effective_at=effective_at)
        for name, column in columns.items():
            for edge, value in column.items() if isinstance(column, dict) else column:
                batch.set_edge_attr(edge, name, value)
        return self.apply_batch(batch, "set_edge_attrs")["state_id"]

    def unset_node_attr(self, node, name, *, effective_at=None):
        batch = GraphBatch()
        batch.unset_node_attr(node, name, effective_at=effective_at)
        return self.apply_batch(batch, "unset_node_attr")["state_id"]

    def unset_node_attrs(self, columns, *, effective_at=None):
        """Columns map property names to entity ID iterables."""
        batch = GraphBatch(effective_at=effective_at)
        for name, ids in columns.items():
            for node in ids:
                batch.unset_node_attr(node, name)
        return self.apply_batch(batch, "unset_node_attrs")["state_id"]

    def unset_edge_attr(self, edge, name, *, effective_at=None):
        batch = GraphBatch()
        batch.unset_edge_attr(edge, name, effective_at=effective_at)
        return self.apply_batch(batch, "unset_edge_attr")["state_id"]

    def unset_edge_attrs(self, columns, *, effective_at=None):
        """Columns map property names to entity ID iterables."""
        batch = GraphBatch(effective_at=effective_at)
        for name, ids in columns.items():
            for edge in ids:
                batch.unset_edge_attr(edge, name)
        return self.apply_batch(batch, "unset_edge_attrs")["state_id"]

    def set_node_labels(self, node, labels, *, effective_at=None):
        return self.set_node_attr(node, "labels", labels, effective_at=effective_at)

    def set_edge_relationship_type(self, edge, kind=None, *, effective_at=None):
        return self.set_edge_attr(edge, "relationship_type", kind, effective_at=effective_at)

    def remove_node(self, node, *, effective_at=None):
        return self.remove_nodes([node], effective_at=effective_at)

    def remove_nodes(self, nodes, *, effective_at=None):
        batch = GraphBatch(effective_at=effective_at)
        for node in nodes:
            batch.remove_node(node)
        return self.apply_batch(batch, "remove_nodes")["state_id"]

    def remove_edge(self, edge, *, effective_at=None):
        return self.remove_edges([edge], effective_at=effective_at)

    def remove_edges(self, edges, *, effective_at=None):
        batch = GraphBatch(effective_at=effective_at)
        for edge in edges:
            batch.remove_edge(edge)
        return self.apply_batch(batch, "remove_edges")["state_id"]

    def set_node_external_id(self, node, external_id, *, effective_at=None):
        batch = GraphBatch()
        batch.set_node_external_id(node, external_id, effective_at=effective_at)
        return self.apply_batch(batch, "set_node_external_id")["state_id"]

    def set_edge_external_id(self, edge, external_id, *, effective_at=None):
        batch = GraphBatch()
        batch.set_edge_external_id(edge, external_id, effective_at=effective_at)
        return self.apply_batch(batch, "set_edge_external_id")["state_id"]

    def upsert_nodes(self, rows, *, uid_key="external_id", effective_at=None):
        result = self._graph.publish_upsert_nodes(
            rows, self._state_id, author=self._author, uid_key=uid_key,
            schema=self._schema, effective_at=effective_at,
        )
        self._state_id = result["state_id"]
        return result["node_ids"]

    def upsert_edges(self, rows, *, uid_key="external_id", effective_at=None):
        result = self._graph.publish_upsert_edges(
            rows, self._state_id, author=self._author, uid_key=uid_key,
            schema=self._schema, effective_at=effective_at,
        )
        self._state_id = result["state_id"]
        return result["edge_ids"]
